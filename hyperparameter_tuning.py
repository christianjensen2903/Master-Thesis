"""
Hyperparameter Tuning Script for GNN Models

Uses Optuna for Bayesian optimization to tune:
- DualEncoderGNN (homogeneous graph) - tunes conv_type (GAT vs GraphSAGE)
- MLPBaseline (no graph structure)
- CaseLinkGNN (semantic graph) - uses GAT
- SymmetricGNN with semantic graph (caselink_symmetric) - tunes conv_type (GAT vs GraphSAGE)
"""

import argparse
import json
import os
import time
from datetime import datetime

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from gnn_evaluator import GNNEvaluator
from gnn_trainers import GNNTrainer
from models import DualEncoderGNN, MLPBaseline, CaseLinkGNN, SymmetricGNN
from preprocessing.graph_builder import HomogeneousGraphBuilder, SemanticGraphBuilder


# ============================================================================
# Configuration
# ============================================================================

TRAIN_CUTOFF_YEAR = 2015
VAL_CUTOFF_YEAR = 2018
INPUT_DIM = 1024
PREPROCESSED_DIR = "data/preprocessed_new"

# Training settings
MAX_TRIALS = 30
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
WARMUP_EPOCHS = 5

# Fixed model settings
FUSION_MODE = "cross_attention"
USE_LANGUAGE = False
LANGUAGE_EMBED_DIM = 16


# ============================================================================
# Model Creation
# ============================================================================


def create_gnn_model(trial: optuna.Trial) -> torch.nn.Module:
    return DualEncoderGNN(
        input_dim=INPUT_DIM,
        output_dim=INPUT_DIM,
        num_layers=trial.suggest_int("num_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        num_heads=trial.suggest_categorical("num_heads", [1, 2, 4]),
        fusion_mode=FUSION_MODE,
        language_embed_dim=LANGUAGE_EMBED_DIM,
        use_language=USE_LANGUAGE,
        use_case_metadata=True,
        conv_type=trial.suggest_categorical("conv_type", ["sage", "gat"]),
    )


def create_mlp_model(trial: optuna.Trial) -> torch.nn.Module:
    return MLPBaseline(
        input_dim=INPUT_DIM,
        output_dim=INPUT_DIM,
        num_layers=trial.suggest_int("num_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        num_heads=trial.suggest_categorical("num_heads", [1, 2, 4]),
        fusion_mode=FUSION_MODE,
        language_embed_dim=LANGUAGE_EMBED_DIM,
        use_language=USE_LANGUAGE,
        use_case_metadata=True,
    )


def create_caselink_model(trial: optuna.Trial) -> torch.nn.Module:
    return CaseLinkGNN(
        input_dim=INPUT_DIM,
        hidden_dim=INPUT_DIM,
        output_dim=INPUT_DIM,
        num_layers=trial.suggest_int("num_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        num_heads=trial.suggest_categorical("num_heads", [1, 2, 4]),
    )


def create_caselink_symmetric_model(trial: optuna.Trial) -> torch.nn.Module:
    return SymmetricGNN(
        input_dim=INPUT_DIM,
        output_dim=INPUT_DIM,
        num_layers=trial.suggest_int("num_layers", 1, 3),
        dropout=trial.suggest_float("dropout", 0.1, 0.5, step=0.1),
        num_heads=trial.suggest_categorical("num_heads", [1, 2, 4]),
        fusion_mode=FUSION_MODE,
        language_embed_dim=LANGUAGE_EMBED_DIM,
        use_language=USE_LANGUAGE,
        use_case_metadata=True,
        conv_type=trial.suggest_categorical("conv_type", ["sage", "gat"]),
    )


def get_training_params(trial: optuna.Trial, model_name: str) -> dict:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "temperature": trial.suggest_float("temperature", 0.03, 0.15, step=0.01),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512, 1024]),
        "warmup_epochs": WARMUP_EPOCHS,
        "epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }

    if model_name in ("caselink", "caselink_symmetric"):
        params["degree_reg_weight"] = trial.suggest_categorical(
            "degree_reg_weight", [0, 5e-4, 1e-3, 5e-3]
        )

    return params


# ============================================================================
# Objective Functions
# ============================================================================


def run_trial(
    trial: optuna.Trial,
    model: torch.nn.Module,
    graph_builder: HomogeneousGraphBuilder | SemanticGraphBuilder,
    model_name: str,
    num_hops: int,
) -> float:
    """Common trial execution logic for all models."""
    training_params = get_training_params(trial, model_name)

    trainer = GNNTrainer(
        graph_builder=graph_builder,
        output_path=f"checkpoints/tuning/{model_name}_trial_{trial.number}",
        num_hops=num_hops,
        wandb_project=None,
        eval_every_n_epochs=1,
        **training_params,
    )

    try:
        trained_model = trainer.train(
            model,
            train_cutoff_year=TRAIN_CUTOFF_YEAR,
            val_cutoff_year=VAL_CUTOFF_YEAR,
        )

        evaluator = GNNEvaluator(
            gnn_model=trained_model,
            graph_builder=graph_builder,
            par_to_par_path="data/par-to-par-cleaned.csv",
            train_cutoff_year=VAL_CUTOFF_YEAR,
            k_hops=num_hops,
            top_k=1000,
        )

        map_score = evaluator.run(k_values=[5, 10, 100])

        trial.report(map_score, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        return map_score

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def objective_gnn(trial: optuna.Trial, graph_builder: HomogeneousGraphBuilder) -> float:
    model = create_gnn_model(trial)
    num_layers = trial.params["num_layers"]
    return run_trial(trial, model, graph_builder, "gnn", num_hops=num_layers)


def objective_mlp(trial: optuna.Trial, graph_builder: HomogeneousGraphBuilder) -> float:
    model = create_mlp_model(trial)
    return run_trial(trial, model, graph_builder, "mlp", num_hops=0)


def objective_caselink(trial: optuna.Trial) -> float:
    # Tune semantic_max_neighbors (requires new graph builder per trial)
    semantic_max_neighbors = trial.suggest_categorical("semantic_max_neighbors", [3, 5])

    graph_builder = SemanticGraphBuilder(
        PREPROCESSED_DIR,
        "data/judgments_cleaned.json",
        semantic_threshold=0.0,
        semantic_max_neighbors=semantic_max_neighbors,
        include_article_nodes=False,
        semantic_cache_path="data/semantic_cache",
    )

    model = create_caselink_model(trial)
    num_layers = trial.params["num_layers"]
    return run_trial(trial, model, graph_builder, "caselink", num_hops=num_layers)


def objective_caselink_symmetric(trial: optuna.Trial) -> float:
    """CaseLink using SymmetricGNN architecture with semantic graph."""
    semantic_max_neighbors = trial.suggest_categorical("semantic_max_neighbors", [3, 5])

    graph_builder = SemanticGraphBuilder(
        PREPROCESSED_DIR,
        "data/judgments_cleaned.json",
        semantic_threshold=0.0,
        semantic_max_neighbors=semantic_max_neighbors,
        include_article_nodes=False,
        semantic_cache_path="data/semantic_cache",
    )

    model = create_caselink_symmetric_model(trial)
    num_layers = trial.params["num_layers"]
    return run_trial(
        trial, model, graph_builder, "caselink_symmetric", num_hops=num_layers
    )


# ============================================================================
# Main
# ============================================================================


def run_hyperparameter_search(
    model_name: str,
    n_trials: int = MAX_TRIALS,
    storage: str | None = None,
    output_dir: str = "results/tuning",
) -> optuna.Study:
    """Run hyperparameter search for a specific model."""
    os.makedirs(output_dir, exist_ok=True)

    study_name = f"{model_name}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_path = os.path.join(output_dir, f"{study_name}.json")

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=TPESampler(seed=42, n_startup_trials=5),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        storage=storage,
        load_if_exists=True,
    )

    # Create graph builder and objective
    if model_name == "gnn":
        builder = HomogeneousGraphBuilder(PREPROCESSED_DIR, include_only_citing=True)
        objective = lambda t: objective_gnn(t, builder)
    elif model_name == "mlp":
        builder = HomogeneousGraphBuilder(PREPROCESSED_DIR, include_only_citing=True)
        objective = lambda t: objective_mlp(t, builder)
    elif model_name == "caselink":
        # Graph builder created per trial (semantic_max_neighbors is tuned)
        objective = lambda t: objective_caselink(t)
    elif model_name == "caselink_symmetric":
        # SymmetricGNN with semantic graph (graph builder created per trial)
        objective = lambda t: objective_caselink_symmetric(t)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\n{'='*80}")
    print(f"Hyperparameter Search: {model_name.upper()}")
    print(f"{'='*80}")
    print(
        f"Trials: {n_trials} | Epochs: {EPOCHS} | Patience: {EARLY_STOPPING_PATIENCE}"
    )
    print(f"Results will be saved to: {output_path}")

    # Create callback for continuous saving
    save_callback = create_continuous_save_callback(output_path)

    start_time = time.time()
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=False, callbacks=[save_callback]
    )
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Complete: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Best MAP: {study.best_value:.4f}")
    print("Best params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"Time: {elapsed/3600:.2f} hours")
    print(f"All results saved to: {output_path}")

    return study


def save_all_trials(study: optuna.Study, output_path: str) -> None:
    """Save all trial results to a JSON file."""
    trials_data = []
    for trial in study.trials:
        trial_data = {
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "state": trial.state.name,
            "datetime_start": (
                trial.datetime_start.isoformat() if trial.datetime_start else None
            ),
            "datetime_complete": (
                trial.datetime_complete.isoformat() if trial.datetime_complete else None
            ),
        }
        trials_data.append(trial_data)

    results = {
        "study_name": study.study_name,
        "best_trial": study.best_trial.number if study.best_trial else None,
        "best_value": study.best_value if study.best_trial else None,
        "best_params": study.best_params if study.best_trial else None,
        "n_trials": len(study.trials),
        "trials": trials_data,
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)


def create_continuous_save_callback(output_path: str):
    """Create a callback that saves results after each trial."""

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        save_all_trials(study, output_path)
        print(f"Results saved to {output_path} (trial {trial.number})")

    return callback


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning")
    parser.add_argument(
        "--model",
        choices=["gnn", "mlp", "caselink", "caselink_symmetric", "all"],
        default="gnn",
    )
    parser.add_argument("--n_trials", type=int, default=MAX_TRIALS)
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/tuning")
    args = parser.parse_args()

    models = (
        ["gnn", "mlp", "caselink", "caselink_symmetric"]
        if args.model == "all"
        else [args.model]
    )

    for model in models:
        run_hyperparameter_search(model, args.n_trials, args.storage, args.output_dir)


if __name__ == "__main__":
    main()

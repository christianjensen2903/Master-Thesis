"""
Hyperparameter Tuning Script for GNN Models

This script performs systematic hyperparameter optimization for:
1. Homogeneous GNN (SymmetricGNN or DualEncoderGNN)
2. MLP Baseline
3. CaseLink GNN

Uses Optuna for Bayesian optimization with early pruning for efficiency.

Author: Generated for Master Thesis
"""

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch

from gnn_evaluator import GNNEvaluator
from gnn_trainers import GNNTrainer
from models import DualEncoderGNN, SymmetricGNN, MLPBaseline, CaseLinkGNN
from preprocessing.graph_builder import HomogeneousGraphBuilder, SemanticGraphBuilder


# ============================================================================
# Configuration
# ============================================================================

TRAIN_CUTOFF_YEAR = 2015
VAL_CUTOFF_YEAR = 2018
INPUT_DIM = 1024  # mE5-Large embedding dimension
PREPROCESSED_DIR = "data/preprocessed_new"

# Time budget settings
MAX_TRIALS_PER_MODEL = 30  # Adjust based on available time
EPOCHS_FOR_TUNING = 30  # Reduced epochs during tuning
EARLY_STOPPING_PATIENCE = 5


def create_gnn_model(trial: optuna.Trial) -> torch.nn.Module:
    """Create a GNN model with hyperparameters suggested by Optuna."""

    # Architecture hyperparameters
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4])
    fusion_mode = "cross_attention"

    output_dim = INPUT_DIM

    # Language embedding dimension
    language_embed_dim = 16

    return DualEncoderGNN(
        input_dim=INPUT_DIM,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        fusion_mode=fusion_mode,
        language_embed_dim=language_embed_dim,
        use_language=False,
        use_case_metadata=True,
    )


def create_mlp_model(trial: optuna.Trial) -> torch.nn.Module:
    """Create an MLP baseline with hyperparameters suggested by Optuna."""

    # MLP may benefit from more layers to compensate for lack of graph structure
    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4])
    fusion_mode = "cross_attention"

    output_dim = INPUT_DIM

    language_embed_dim = 16

    return MLPBaseline(
        input_dim=INPUT_DIM,
        output_dim=output_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
        fusion_mode=fusion_mode,
        language_embed_dim=language_embed_dim,
        use_language=False,
        use_case_metadata=True,
    )


def create_caselink_model(trial: optuna.Trial) -> torch.nn.Module:
    """Create a CaseLink GNN with hyperparameters suggested by Optuna."""

    num_layers = trial.suggest_int("num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
    num_heads = trial.suggest_categorical("num_heads", [1, 2, 4])

    # CaseLink uses hidden_dim, output_dim
    hidden_dim = 1024

    return CaseLinkGNN(
        input_dim=INPUT_DIM,
        hidden_dim=hidden_dim,
        output_dim=INPUT_DIM,  # Keep output same for fair comparison
        num_layers=num_layers,
        dropout=dropout,
        num_heads=num_heads,
    )


def get_training_params(trial: optuna.Trial, model_name: str) -> dict:
    """Get training hyperparameters from Optuna trial."""

    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    temperature = trial.suggest_float("temperature", 0.03, 0.15, step=0.01)
    batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
    warmup_epochs = 5

    params = {
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "temperature": temperature,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "epochs": EPOCHS_FOR_TUNING,
        "early_stopping_patience": EARLY_STOPPING_PATIENCE,
    }

    # CaseLink-specific: degree regularization
    if model_name == "caselink":
        degree_reg_weight = trial.suggest_categorical(
            "degree_reg_weight", [0, 5e-4, 1e-3, 5e-3]
        )
        params["degree_reg_weight"] = degree_reg_weight

    return params


def objective_gnn(
    trial: optuna.Trial,
    graph_builder: HomogeneousGraphBuilder | None = None,
) -> float:
    """Optuna objective function for GNN hyperparameter tuning."""

    if graph_builder is None:
        graph_builder = HomogeneousGraphBuilder(
            PREPROCESSED_DIR,
            include_only_citing=True,
        )

    model = create_gnn_model(trial)
    training_params = get_training_params(trial, "gnn")
    num_layers = trial.params["num_layers"]

    trainer = GNNTrainer(
        graph_builder=graph_builder,
        output_path=f"checkpoints/tuning/gnn_trial_{trial.number}",
        num_hops=num_layers,
        wandb_project=None,  # Disable wandb during tuning
        eval_every_n_epochs=1,
        **training_params,
    )

    try:
        trained_model = trainer.train(
            model,
            train_cutoff_year=TRAIN_CUTOFF_YEAR,
            val_cutoff_year=VAL_CUTOFF_YEAR,
        )

        # Evaluate on validation set
        evaluator = GNNEvaluator(
            gnn_model=trained_model,
            graph_builder=graph_builder,
            par_to_par_path="data/par-to-par-cleaned.csv",
            train_cutoff_year=VAL_CUTOFF_YEAR,  # Use val cutoff for incremental eval
            k_hops=num_layers,
            top_k=1000,
        )

        map_score = evaluator.run(k_values=[5, 10, 100])

        # Report intermediate value for pruning
        trial.report(map_score, step=1)

        if trial.should_prune():
            raise optuna.TrialPruned()

        return map_score

    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        return 0.0


def objective_mlp(
    trial: optuna.Trial,
    graph_builder: HomogeneousGraphBuilder | None = None,
) -> float:
    """Optuna objective function for MLP baseline tuning."""

    if graph_builder is None:
        graph_builder = HomogeneousGraphBuilder(
            PREPROCESSED_DIR,
            include_only_citing=True,
        )

    model = create_mlp_model(trial)
    training_params = get_training_params(trial, "mlp")

    trainer = GNNTrainer(
        graph_builder=graph_builder,
        output_path=f"checkpoints/tuning/mlp_trial_{trial.number}",
        num_hops=0,  # No neighbor sampling for MLP
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
            k_hops=0,
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


def objective_caselink(
    trial: optuna.Trial,
    graph_builder: SemanticGraphBuilder | None = None,
) -> float:
    """Optuna objective function for CaseLink tuning."""

    if graph_builder is None:
        graph_builder = SemanticGraphBuilder(
            PREPROCESSED_DIR,
            "data/judgments_cleaned.json",
            semantic_threshold=0.0,
            include_article_nodes=True,
        )

    model = create_caselink_model(trial)
    training_params = get_training_params(trial, "caselink")
    num_layers = trial.params["num_layers"]

    trainer = GNNTrainer(
        graph_builder=graph_builder,
        output_path=f"checkpoints/tuning/caselink_trial_{trial.number}",
        num_hops=num_layers,
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
            k_hops=num_layers,
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


def run_hyperparameter_search(
    model_name: str,
    n_trials: int = MAX_TRIALS_PER_MODEL,
    study_name: str | None = None,
    storage: str | None = None,
) -> optuna.Study:
    """Run hyperparameter search for a specific model."""

    if study_name is None:
        study_name = f"{model_name}_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Use TPE sampler with pruning for efficiency
    sampler = TPESampler(seed=42, n_startup_trials=5)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",  # Maximize MAP
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    # Select objective function based on model
    if model_name == "gnn":
        homo_builder = HomogeneousGraphBuilder(
            PREPROCESSED_DIR, include_only_citing=True
        )
        objective = lambda trial: objective_gnn(trial, homo_builder)
    elif model_name == "mlp":
        homo_builder = HomogeneousGraphBuilder(
            PREPROCESSED_DIR, include_only_citing=True
        )
        objective = lambda trial: objective_mlp(trial, homo_builder)
    elif model_name == "caselink":
        semantic_builder = SemanticGraphBuilder(
            PREPROCESSED_DIR,
            "data/judgments_cleaned.json",
        )
        objective = lambda trial: objective_caselink(trial, semantic_builder)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"\n{'='*80}")
    print(f"Starting Hyperparameter Search for {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")

    start_time = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    elapsed = time.time() - start_time

    print(f"\n{'='*80}")
    print(f"Hyperparameter Search Complete for {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best MAP: {study.best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Total time: {elapsed/3600:.2f} hours")

    return study


def save_results(
    studies: dict[str, optuna.Study], output_dir: str = "results/tuning"
) -> None:
    """Save tuning results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for model_name, study in studies.items():
        results[model_name] = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "best_trial_number": study.best_trial.number,
        }

    output_path = os.path.join(
        output_dir, f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for GNN Models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["gnn", "mlp", "caselink", "all"],
        default="gnn",
        help="Model to tune (default: gnn)",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=MAX_TRIALS_PER_MODEL,
        help=f"Number of trials per model (default: {MAX_TRIALS_PER_MODEL})",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///tuning.db) for persistence",
    )
    args = parser.parse_args()

    studies = {}

    if args.model in ["gnn", "all"]:
        studies["gnn"] = run_hyperparameter_search(
            "gnn", args.n_trials, storage=args.storage
        )

    if args.model in ["mlp", "all"]:
        studies["mlp"] = run_hyperparameter_search(
            "mlp", args.n_trials, storage=args.storage
        )

    if args.model in ["caselink", "all"]:
        studies["caselink"] = run_hyperparameter_search(
            "caselink", args.n_trials, storage=args.storage
        )

    if studies:
        save_results(studies)


if __name__ == "__main__":
    main()

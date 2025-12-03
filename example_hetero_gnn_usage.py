from gnn_trainers import GNNTrainer
from models import create_hetero_model, DualEncoderGNN
from preprocessing.graph_builder import HeterogeneousGraphBuilder
from gnn_evaluator import GNNEvaluator
import torch


def train_hetero_example() -> None:
    """Example: Train a heterogeneous GNN model."""
    print("\n" + "=" * 80)
    print("Training Heterogeneous GNN Model")
    print("=" * 80 + "\n")

    # Build graph to get dimensions
    print("Building graph to extract dimensions...")
    builder = HeterogeneousGraphBuilder(
        "data/preprocessed",
        include_articles=False,  # Start without articles
        include_citations=True,
    )
    sample_graph = builder.build_graph(train_cutoff_year=2018)

    print(f"Node types: {sample_graph.node_types}")
    print(f"Edge types: {sample_graph.edge_types}")

    # Create heterogeneous GNN using factory function
    layers = 3
    hidden_dim = 256
    model = create_hetero_model(
        graph_data=sample_graph,
        model_type="symmetric",  # or "dual" for separate query/doc encoders
        hidden_dim=hidden_dim,
        output_dim=hidden_dim,
        num_layers=layers,
        dropout=0.3,
        conv_type="sage",
        use_language=False,
        language_embed_dim=16,
    )

    print(f"\nModel embedding dimension: {model.embedding_dim}")

    graph_builder = HeterogeneousGraphBuilder(
        "data/preprocessed",
        include_articles=False,
        include_citations=True,
    )

    # Initialize trainer with heterogeneous graph type
    trainer = GNNTrainer(
        graph_builder=graph_builder,
        output_path="checkpoints/hetero_gnn",
        batch_size=512,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-3,
        temperature=0.05,
        num_hops=layers,
        warmup_epochs=3,
        early_stopping_patience=5,
        early_stopping_min_delta=1e-3,
        eval_every_n_epochs=1,
    )

    # Train on paragraph pairs
    cutoff_year = 2018
    val_cutoff_year = 2022
    trainer.train(model, cutoff_year, val_cutoff_year)

    print("\nTraining complete!")

    # Load best model
    model.load_state_dict(torch.load("checkpoints/hetero_gnn/best_model.pt"))

    evaluator = GNNEvaluator(
        gnn_model=model,
        graph_builder=graph_builder,
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=cutoff_year,
        k_hops=layers,
        top_k=1000,
    )
    evaluator.run(k_values=[5, 10, 100])


def train_homo_example() -> None:
    """Example: Train a homogeneous GNN model (original approach)."""
    print("\n" + "=" * 80)
    print("Training Homogeneous GNN Model")
    print("=" * 80 + "\n")
    from preprocessing.graph_builder import HomogeneousGraphBuilder

    in_channels = 1024

    layers = 1

    model = DualEncoderGNN(
        input_dim=in_channels,
        output_dim=in_channels,
        num_layers=layers,
        dropout=0.4,
        fusion_mode="cross_attention",
        use_language=False,
    )

    graph_builder = HomogeneousGraphBuilder("data/preprocessed_new")

    # Initialize trainer with homogeneous graph type
    trainer = GNNTrainer(
        graph_builder=graph_builder,
        output_path="checkpoints/homo_gnn",
        batch_size=256,
        epochs=50,
        learning_rate=1e-4,
        weight_decay=4e-5,
        temperature=0.03,
        num_hops=layers,
        checkpoint_interval=10,
        wandb_project="homo-gnn-training",
        # gradient_clip_val=3.0,
        eval_every_n_epochs=1,
        warmup_epochs=5,
        early_stopping_patience=5,
        early_stopping_min_delta=1e-3,
    )
    # Train on paragraph pairs
    cutoff_year = 2016
    trainer.train(model, cutoff_year, 2018)

    print("\nTraining complete!")

    # Load best model
    model.load_state_dict(torch.load("checkpoints/homo_gnn/best_model.pt"))

    evaluator = GNNEvaluator(
        gnn_model=model,
        graph_builder=graph_builder,
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,
        k_hops=layers,
        top_k=10000,
    )
    evaluator.run(k_values=[5, 10, 100])

    print("\nEvaluation complete!")


def train_caselink_example() -> None:
    """Example: Train a case-link GNN model."""
    print("\n" + "=" * 80)
    print("Training Case-Link GNN Model")
    print("=" * 80 + "\n")
    from preprocessing.graph_builder import SemanticGraphBuilder
    from models import CaseLinkGNN

    in_channels = 384
    layers = 1

    model = CaseLinkGNN(
        input_dim=in_channels,
        num_layers=layers,
        dropout=0.2,
        num_heads=1,
    )

    trainer = GNNTrainer(
        graph_builder=SemanticGraphBuilder(
            "data/preprocessed",
            "data/judgments_cleaned.json",
            semantic_cache_path="data/semantic_cache",
        ),
        output_path="checkpoints/caselink_gnn",
        batch_size=512,
        epochs=100,
        learning_rate=1e-4,
        weight_decay=1e-6,
        temperature=0.1,
        num_hops=layers,
        checkpoint_interval=10,
        wandb_project="caselink-gnn-training",
        degree_reg_weight=1e-3,
    )

    cutoff_year = 2018
    trainer.train(model, cutoff_year, 2022)

    print("\nTraining complete!")


def train_mlp_baseline_example() -> None:
    """Example: Train an MLP baseline (no graph structure) for comparison."""
    print("\n" + "=" * 80)
    print("Training MLP Baseline (No Graph Structure)")
    print("=" * 80 + "\n")
    from preprocessing.graph_builder import HomogeneousGraphBuilder
    from models import MLPBaseline

    in_channels = 384
    layers = 2  # MLP depth

    model = MLPBaseline(
        input_dim=in_channels,
        output_dim=in_channels,
        num_layers=layers,
        dropout=0.3,
        fusion_mode="scalar",
    )

    # Uses the same graph builder for data loading, but model ignores edges
    trainer = GNNTrainer(
        graph_builder=HomogeneousGraphBuilder("data/preprocessed"),
        output_path="checkpoints/mlp_baseline",
        batch_size=512,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-3,
        temperature=0.05,
        num_hops=0,  # No neighbor sampling needed for MLP
        checkpoint_interval=10,
        wandb_project="mlp-baseline-training",
        eval_every_n_epochs=1,
        warmup_epochs=3,
        early_stopping_patience=5,
    )

    cutoff_year = 2018
    trainer.train(model, cutoff_year, 2022)

    print("\nTraining complete!")


if __name__ == "__main__":
    # Train heterogeneous GNN (uses all edge types)
    # train_hetero_example()

    # Or train homogeneous GNN (citation edges only)
    train_homo_example()
    # train_caselink_example()

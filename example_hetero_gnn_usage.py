from gnn_trainers import GNNTrainer
from models import HeteroGNN, DualEncoderGNN
from preprocessing.graph_builder import HeterogeneousGraphBuilder
from gnn_evaluator import GNNEvaluator
import torch


def train_hetero_example() -> None:
    """Example: Train a heterogeneous GNN model."""
    print("\n" + "=" * 80)
    print("Training Heterogeneous GNN Model")
    print("=" * 80 + "\n")

    in_channels = 384  # mE5-Small

    # Build graph to get metadata
    print("Building graph to extract metadata...")
    builder = HeterogeneousGraphBuilder("data/preprocessed")
    sample_graph = builder.build_graph(train_cutoff_year=2018)

    # Extract metadata (node types and edge types)
    metadata = (
        list(sample_graph.node_types),
        list(sample_graph.edge_types),
    )
    print(f"Node types: {metadata[0]}")
    print(f"Edge types: {metadata[1]}")

    # Get input dimensions per node type
    # Paragraph/article nodes: paragraph embeddings (384)
    # Case nodes: date(3) + language(23) + subject_matter(384) + keywords(384) + case_law_about(384)
    # Legal act nodes: averaged article embeddings (384)
    node_input_dims = {
        node_type: sample_graph[node_type].x.shape[1]
        for node_type in sample_graph.node_types
    }
    print(f"Node input dimensions: {node_input_dims}")

    # Initialize heterogeneous GNN
    layers = 2
    model = HeteroGNN(
        input_dim=in_channels,
        hidden_dim=in_channels,
        output_dim=in_channels,
        num_layers=layers,
        metadata=metadata,
        node_input_dims=node_input_dims,
    )

    graph_builder = HeterogeneousGraphBuilder("data/preprocessed")

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

    in_channels = 384

    layers = 1

    model = DualEncoderGNN(
        input_dim=in_channels,
        output_dim=in_channels,
        num_layers=layers,
        dropout=0.3,
        fusion_mode="cross_attention",
        use_language=False,
    )

    # Initialize trainer with homogeneous graph type
    trainer = GNNTrainer(
        graph_builder=HomogeneousGraphBuilder("data/preprocessed"),
        output_path="checkpoints/homo_gnn",
        batch_size=512,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=1e-3,
        temperature=0.05,
        num_hops=layers,
        checkpoint_interval=10,
        wandb_project="homo-gnn-training",
        # gradient_clip_val=3.0,
        eval_every_n_epochs=1,
        warmup_epochs=3,
        early_stopping_patience=5,
    )
    # Train on paragraph pairs
    cutoff_year = 2018
    trainer.train(model, cutoff_year, 2022)

    print("\nTraining complete!")


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
    train_hetero_example()

    # Or train homogeneous GNN (citation edges only)
    # train_homo_example()
    # train_caselink_example()

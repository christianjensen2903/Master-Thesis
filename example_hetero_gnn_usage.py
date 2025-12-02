from gnn_trainers import GNNTrainer
from models import HeteroGNN, DualEncoderGNN
from preprocessing.graph_builder import HeterogeneousGraphBuilder


def train_hetero_example() -> None:
    """Example: Train a heterogeneous GNN model."""
    print("\n" + "=" * 80)
    print("Training Heterogeneous GNN Model")
    print("=" * 80 + "\n")

    in_channels = 384  # mE5-Small

    # Build graph to get metadata
    print("Building graph to extract metadata...")
    builder = HeterogeneousGraphBuilder("data/preprocessed")
    sample_graph = builder.build_graph(train_cutoff_year=2018, include_only_citing=True)

    # Extract metadata (node types and edge types)
    metadata = (
        list(sample_graph.node_types),
        list(sample_graph.edge_types),
    )
    print(f"Node types: {metadata[0]}")
    print(f"Edge types: {metadata[1]}")

    # Initialize heterogeneous GNN
    model = HeteroGNN(
        input_dim=in_channels,
        hidden_dim=in_channels,
        output_dim=in_channels,
        num_layers=2,
        metadata=metadata,
    )

    # Initialize trainer with heterogeneous graph type
    trainer = GNNTrainer(
        preprocessed_dir="data/preprocessed",
        output_path="checkpoints/hetero_gnn",
        batch_size=2**10,
        epochs=400,
        learning_rate=5e-5,
        weight_decay=1e-4,
        temperature=0.07,
        num_hops=2,
        graph_type="heterogeneous",  # Use heterogeneous graph
    )

    # Train on paragraph pairs
    cutoff_year = 2018
    trainer.train(model, cutoff_year)

    print("\nTraining complete!")


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
        fusion_mode="scalar",
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
            "data/preprocessed", "data/judgments_cleaned.json"
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


if __name__ == "__main__":
    # Train heterogeneous GNN (uses all edge types)
    # train_hetero_example()

    # Or train homogeneous GNN (citation edges only)
    train_homo_example()
    # train_caselink_example()

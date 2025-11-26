from gnn_trainers import GNNTrainer
from models import HeteroGNN, CitationGNN
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

    in_channels = 384
    hidden_dim = 384
    out_channels = 384

    # Initialize homogeneous GNN
    model = CitationGNN(
        input_dim=384,
        output_dim=384,
        num_layers=2,
        dropout=0.3,
    )

    # Initialize trainer with homogeneous graph type
    trainer = GNNTrainer(
        preprocessed_dir="data/preprocessed",
        output_path="checkpoints/homo_gnn",
        batch_size=2048,
        epochs=50,
        learning_rate=1e-3,
        weight_decay=5e-4,
        temperature=0.05,
        num_hops=2,
        checkpoint_interval=10,
        graph_type="homogeneous",  # Use homogeneous graph
        wandb_project="homo-gnn-training",
        # gradient_clip_val=3.0,
        eval_every_n_epochs=2,
        warmup_epochs=3,
        x_similarity_threshold=0.8,
    )
    # Train on paragraph pairs
    cutoff_year = 2018
    trainer.train(model, cutoff_year, 2022)

    print("\nTraining complete!")


if __name__ == "__main__":
    # Train heterogeneous GNN (uses all edge types)
    # train_hetero_example()

    # Or train homogeneous GNN (citation edges only)
    train_homo_example()

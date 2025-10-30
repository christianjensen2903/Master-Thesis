"""
Improved GNN Training with better hyperparameters and validation.

Key improvements:
1. Proper validation-based early stopping
2. Better hyperparameters (more conservative)
3. Smaller model to reduce overfitting
4. Comparison with base encoder
"""

import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from gnn_trainers import GNNTrainer
from torch_geometric.nn import GAT, GraphSAGE


def train_improved_gat() -> None:
    """Train GAT with improved hyperparameters."""
    print("\n" + "=" * 80)
    print("Training Improved GAT Model")
    print("=" * 80 + "\n")

    # Initialize text encoder
    encoding_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_encoder = SentenceTransformer(encoding_model)
    embedding_dim = text_encoder.get_sentence_embedding_dimension()

    # Improved GAT configuration
    model = GAT(
        in_channels=embedding_dim,
        hidden_channels=192,  # Smaller to reduce overfitting
        out_channels=embedding_dim,  # Keep same dimension as input!
        num_layers=2,  # Fewer layers to prevent over-smoothing
        dropout=0.3,  # Higher dropout for regularization
        v2=True,
        heads=4,  # More heads for attention
        concat=True,
    )

    trainer = GNNTrainer(
        text_encoder_name=encoding_model,
        output_path="checkpoints/gnn_improved",
        batch_size=64,  # Larger batch for more stable gradients
        epochs=100,  # More epochs OK with early stopping
        eval_every_n_epochs=2,  # Evaluate more frequently
        learning_rate=1e-4,  # Lower LR for stability
        weight_decay=5e-4,  # Stronger regularization
        temperature=0.07,  # Standard temperature (not too aggressive)
        num_negatives=8,  # Fewer negatives
        validation_split=0.1,
        use_wandb=True,
        embeddings_cache_dir="artifacts/embeddings_cache",
    )

    cutoff_date = pd.Timestamp("2018-01-01")
    trainer.train(model, "data/par-to-par-og.csv", cutoff_date)

    print("\n✓ Training complete!")


def train_graphsage() -> None:
    """Try GraphSAGE as alternative to GAT."""
    print("\n" + "=" * 80)
    print("Training GraphSAGE Model")
    print("=" * 80 + "\n")

    encoding_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_encoder = SentenceTransformer(encoding_model)
    embedding_dim = text_encoder.get_sentence_embedding_dimension()

    # GraphSAGE often works better for retrieval tasks
    model = GraphSAGE(
        in_channels=embedding_dim,
        hidden_channels=256,
        out_channels=embedding_dim,  # Keep same dimension
        num_layers=2,
        dropout=0.3,
        aggr="mean",  # Mean aggregation
    )

    trainer = GNNTrainer(
        text_encoder_name=encoding_model,
        output_path="checkpoints/gnn_graphsage",
        batch_size=64,
        epochs=100,
        eval_every_n_epochs=2,
        learning_rate=1e-4,
        weight_decay=5e-4,
        temperature=0.07,
        num_negatives=8,
        validation_split=0.1,
        use_wandb=True,
        embeddings_cache_dir="artifacts/embeddings_cache",
    )

    cutoff_date = pd.Timestamp("2018-01-01")
    trainer.train(model, "data/par-to-par-og.csv", cutoff_date)

    print("\n✓ Training complete!")


def train_shallow_gnn() -> None:
    """Train very shallow GNN - sometimes 1 layer is enough!"""
    print("\n" + "=" * 80)
    print("Training Shallow GAT (1 layer)")
    print("=" * 80 + "\n")

    encoding_model = "sentence-transformers/all-MiniLM-L6-v2"
    text_encoder = SentenceTransformer(encoding_model)
    embedding_dim = text_encoder.get_sentence_embedding_dimension()

    # Very shallow model - just refine embeddings slightly
    model = GAT(
        in_channels=embedding_dim,
        hidden_channels=embedding_dim,
        out_channels=embedding_dim,
        num_layers=1,  # Just 1 layer!
        dropout=0.2,
        v2=True,
        heads=4,
        concat=False,  # Average heads instead of concat
    )

    trainer = GNNTrainer(
        text_encoder_name=encoding_model,
        output_path="checkpoints/gnn_shallow",
        batch_size=64,
        epochs=100,
        eval_every_n_epochs=2,
        learning_rate=2e-4,  # Can use higher LR with shallow model
        weight_decay=5e-4,
        temperature=0.07,
        num_negatives=8,
        validation_split=0.1,
        use_wandb=True,
        embeddings_cache_dir="artifacts/embeddings_cache",
    )

    cutoff_date = pd.Timestamp("2018-01-01")
    trainer.train(model, "data/par-to-par-og.csv", cutoff_date)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        model_type = sys.argv[1]
        if model_type == "gat":
            train_improved_gat()
        elif model_type == "graphsage":
            train_graphsage()
        elif model_type == "shallow":
            train_shallow_gnn()
        else:
            print(f"Unknown model type: {model_type}")
            print("Usage: python improved_gnn_training.py [gat|graphsage|shallow]")
    else:
        # Default: train improved GAT
        train_improved_gat()

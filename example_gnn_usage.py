import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from gnn_trainers import GNNTrainer
from torch_geometric.nn import GAT


def train_example() -> None:
    """Example: Train a GNN model with embeddings caching."""
    print("\n" + "=" * 80)
    print("Training GNN Model")
    print("=" * 80 + "\n")

    # Initialize text encoder
    encoding_model = "checkpoints/simcse_citation_model"
    text_encoder = SentenceTransformer(encoding_model)

    in_channels = text_encoder.get_sentence_embedding_dimension()

    # Option 1: Optimized GraphSAGE
    model = GAT(
        in_channels=in_channels,
        hidden_channels=in_channels,
        out_channels=in_channels,
        num_layers=2,
        dropout=0.2,
        v2=True,
        heads=2,
        concat=True,
    )

    trainer = GNNTrainer(
        text_encoder_name=encoding_model,
        output_path="checkpoints/gnn",
        batch_size=32,
        epochs=50,
        eval_every_n_epochs=1,
        learning_rate=3e-4,
        weight_decay=1e-4,
        temperature=0.05,
        num_negatives=12,
        validation_split=0.1,
        use_wandb=False,
        embeddings_cache_dir="artifacts/embeddings_cache",
    )

    # Train on paragraph pairs (pass model to train method)
    cutoff_year = 2018
    trainer.train(model, "data/par-to-par-cleaned.csv", cutoff_year)

    print("\nTraining complete!")


if __name__ == "__main__":
    train_example()

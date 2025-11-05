import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from gnn_trainers import GNNTrainer
from torch_geometric.nn import GATConv, GCN
import torch.nn as nn
import torch.nn.functional as F


class CitationGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        # GNN layers for documents in the graph
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4)
        self.conv2 = GATConv(hidden_dim * 4, output_dim, heads=1)
        self.dropout = nn.Dropout(0.1)

        # Query projection layer (for paragraphs NOT in graph)
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index):
        """Process documents through GNN."""
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

    def encode_query(self, x):
        """Process queries (not in graph) through projection."""
        return self.query_proj(x)


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
    # model = GCN(
    #     in_channels=in_channels,
    #     hidden_channels=in_channels,
    #     out_channels=in_channels,
    #     num_layers=2,
    #     dropout=0.2,
    #     # v2=True,
    #     # heads=2,
    #     # concat=True,
    # )

    model = CitationGNN(in_channels)

    trainer = GNNTrainer(
        text_encoder_name=encoding_model,
        output_path="checkpoints/gnn",
        batch_size=256,
        epochs=50,
        eval_every_n_epochs=1,
        learning_rate=3e-4,
        weight_decay=1e-2,
        temperature=0.05,
        num_negatives=5,
        validation_split=0.1,
        use_wandb=False,
        embeddings_cache_dir="artifacts/embeddings_cache",
        max_citations_per_anchor=5,
    )

    # Train on paragraph pairs (pass model to train method)
    cutoff_year = 2018
    trainer.train(model, "data/par-to-par-cleaned.csv", cutoff_year)

    print("\nTraining complete!")


if __name__ == "__main__":
    train_example()

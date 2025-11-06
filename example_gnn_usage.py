import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from gnn_trainers import GNNTrainer
from torch_geometric.nn import SAGEConv, GCNConv
import torch.nn as nn
import torch.nn.functional as F
import torch


class CitationGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=None, num_layers=3):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        # Keep same dimensions for residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(GCNConv(input_dim, input_dim))
            self.norms.append(nn.LayerNorm(input_dim))

        # Optional projection at the end only
        self.final_proj = nn.Sequential(
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, output_dim),
        )

        self.dropout = nn.Dropout(0.1)

        # Query projection with residual connection
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, edge_index):
        x_orig = x

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        # Concat original embeddings with GNN output
        x = torch.cat([x_orig, x], dim=1)
        return self.final_proj(x)

    def encode_query(self, x):
        """Process queries (not in graph) through projection."""
        return x
        # return self.query_proj(x)


def train_example() -> None:
    """Example: Train a GNN model with embeddings caching."""
    print("\n" + "=" * 80)
    print("Training GNN Model")
    print("=" * 80 + "\n")

    # Initialize text encoder
    encoding_model = "checkpoints/simcse_citation_model"
    text_encoder = SentenceTransformer(encoding_model)

    in_channels = text_encoder.get_sentence_embedding_dimension()

    model = CitationGNN(
        in_channels, hidden_dim=512, output_dim=in_channels, num_layers=3
    )

    trainer = GNNTrainer(
        text_encoder_name=encoding_model,
        output_path="checkpoints/gnn",
        batch_size=1024,
        epochs=400,
        eval_every_n_epochs=20,
        learning_rate=3e-4,
        weight_decay=1e-2,
        temperature=0.05,
        validation_split=0.1,
        embeddings_cache_dir="artifacts/embeddings_cache",
        num_hops=3,
    )

    # Train on paragraph pairs (pass model to train method)
    cutoff_year = 2018
    trainer.train(model, "data/par-to-par-cleaned.csv", cutoff_year)

    print("\nTraining complete!")


if __name__ == "__main__":
    train_example()

import pandas as pd  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from gnn_trainers import GNNTrainer
from torch_geometric.nn import SAGEConv
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
            self.convs.append(SAGEConv(input_dim, input_dim))
            self.norms.append(nn.LayerNorm(input_dim))

        # Learnable residual weight
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        self.dropout = nn.Dropout(0.1)

        # Simplified query projection
        self.query_proj = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

    def forward(self, x, edge_index):
        x_orig = x

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        alpha = torch.sigmoid(self.residual_weight)
        x = alpha * x_orig + (1 - alpha) * x

        return x

    def encode_query(self, x):
        """Process queries (not in graph) through projection."""
        # return x
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

    model = CitationGNN(
        in_channels, hidden_dim=in_channels, output_dim=in_channels, num_layers=2
    )

    trainer = GNNTrainer(
        text_encoder_name=encoding_model,
        output_path="checkpoints/gnn",
        batch_size=2**10,
        epochs=400,
        eval_every_n_epochs=20,
        learning_rate=5e-5,
        weight_decay=1e-4,
        temperature=0.07,
        validation_split=0.1,
        embeddings_cache_dir="artifacts/embeddings_cache",
        num_hops=2,
    )

    # Train on paragraph pairs (pass model to train method)
    cutoff_year = 2018
    trainer.train(model, "data/par-to-par-cleaned.csv", cutoff_year)

    print("\nTraining complete!")


if __name__ == "__main__":
    train_example()

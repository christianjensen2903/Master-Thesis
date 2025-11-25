import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv  # type: ignore


class CitationGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.5,
        date_feature_dim: int = 1,
        num_heads: int = 4,
        edge_dim: int = 16,  # Dimension for edge feature embedding
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.num_heads = num_heads

        self.date_projection = nn.Sequential(
            nn.Linear(date_feature_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim),
            nn.GELU(),
        )

        # Edge type embedding (2 types: citing=0, cited_by=1)
        self.edge_embedding = nn.Embedding(2, edge_dim)

        # Keep same dimensions for residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            if i < num_layers - 1:
                self.convs.append(
                    GATConv(
                        in_channels=input_dim,
                        out_channels=input_dim // num_heads,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        add_self_loops=False,  # We handle edge types explicitly
                        concat=True,  # Concatenate head outputs
                    )
                )
            else:
                self.convs.append(
                    GATConv(
                        input_dim,
                        input_dim,
                        heads=num_heads,
                        dropout=dropout,
                        edge_dim=edge_dim,
                        add_self_loops=False,
                        concat=False,
                    )
                )
            self.norms.append(nn.LayerNorm(input_dim))

        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim * 2),
            nn.GELU(),
            nn.Linear(output_dim * 2, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:

        # date_feature = self.date_projection(date_feature)
        # x = x + date_feature

        for i, conv in enumerate(self.convs):
            # Convert edge type indices to embeddings
            edge_emb = self.edge_embedding(edge_attr)  # [E, edge_dim]
            x_new = conv(x, edge_index, edge_attr=edge_emb)
            x_new = self.norms[i](x_new)
            if i < len(self.convs) - 1:
                x_new = F.gelu(x_new)
                x_new = self.dropout(x_new)
            x = x + x_new

        x = self.projector(x)

        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        return x

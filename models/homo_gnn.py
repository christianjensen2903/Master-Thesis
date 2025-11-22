import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # type: ignore


class CitationGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.5,
        date_feature_dim: int = 1,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.date_projection = nn.Sequential(
            nn.Linear(date_feature_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim),
            nn.GELU(),
        )

        # Keep same dimensions for residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(input_dim, input_dim, aggr="sum"))
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
    ) -> torch.Tensor:

        date_feature = self.date_projection(date_feature)
        x = x + date_feature

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.norms[i](x_new)
            if i < len(self.convs) - 1:
                x_new = F.gelu(x_new)
                x_new = self.dropout(x_new)
            x = F.layer_norm(x, [x.size(-1)]) + x_new

        # x = self.projector(x)

        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # type: ignore


class CitationGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        num_layers: int = 3,
    ):
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_orig = x

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.norms[i](x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        return x

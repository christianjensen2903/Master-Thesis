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
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        # Keep same dimensions for residual
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(input_dim, input_dim, aggr="sum"))
            self.norms.append(nn.LayerNorm(input_dim))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:

        for i, conv in enumerate(self.convs):
            x_new = conv(x, edge_index)
            x_new = self.norms[i](x_new)
            if i < len(self.convs) - 1:
                x_new = F.gelu(x_new)
                x_new = self.dropout(x_new)
            x = F.layer_norm(x, [x.size(-1)]) + x_new

        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        return x

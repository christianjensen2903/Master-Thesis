import torch
import torch.nn as nn
from torch_geometric.nn import (  # type: ignore
    GATv2Conv,
    SAGEConv,
    GCNConv,
)


class BaseGNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError


class GATv2Encoder(BaseGNNEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        norm_dims: list[int] = []
        prev_dim = hidden_dim
        for i in range(num_layers):
            in_channels = prev_dim
            is_last = i == num_layers - 1
            out_channels = hidden_dim if not is_last else output_dim
            heads = num_heads if not is_last else 1
            concat = not is_last

            self.gat_layers.append(
                GATv2Conv(
                    in_channels,
                    out_channels,
                    heads=heads,
                    dropout=dropout,
                    concat=concat,
                )
            )

            layer_out_dim = out_channels * heads if concat else out_channels
            norm_dims.append(layer_out_dim)
            prev_dim = layer_out_dim

        self.layer_norms = nn.ModuleList([nn.LayerNorm(d) for d in norm_dims])

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_proj(x)
        x = torch.relu(x)

        for i, (gat, norm) in enumerate(zip(self.gat_layers, self.layer_norms)):
            x_new = gat(x, edge_index)
            x_new = norm(x_new)

            if i < self.num_layers - 1:
                x_new = torch.relu(x_new)
                x_new = torch.dropout(x_new, p=self.dropout, train=self.training)
                x = x + x_new if x_new.shape == x.shape else x_new
            else:
                x = x_new

        return x


class GraphSAGEEncoder(BaseGNNEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims: list[int] = [hidden_dim] * (num_layers - 1) + [output_dim]
        prev_dim = hidden_dim
        for i, out_dim in enumerate(dims):
            self.convs.append(SAGEConv(prev_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            prev_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_proj(x)
        x = torch.relu(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            if i < self.num_layers - 1:
                x_new = torch.relu(x_new)
                x_new = torch.dropout(x_new, p=self.dropout, train=self.training)
                x = x + x_new if x_new.shape == x.shape else x_new
            else:
                x = x_new
        return x


class GCNEncoder(BaseGNNEncoder):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        dims: list[int] = [hidden_dim] * (num_layers - 1) + [output_dim]
        prev_dim = hidden_dim
        for i, out_dim in enumerate(dims):
            self.convs.append(GCNConv(prev_dim, out_dim))
            self.norms.append(nn.LayerNorm(out_dim))
            prev_dim = out_dim

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.input_proj(x)
        x = torch.relu(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            if i < self.num_layers - 1:
                x_new = torch.relu(x_new)
                x_new = torch.dropout(x_new, p=self.dropout, train=self.training)
                x = x + x_new if x_new.shape == x.shape else x_new
            else:
                x = x_new
        return x

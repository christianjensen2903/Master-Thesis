import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero  # type: ignore
from torch_geometric.data import HeteroData  # type: ignore


class HeteroGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        num_layers: int = 3,
        metadata: tuple | None = None,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Create a homogeneous base model first
        self.base_model = BaseGNN(input_dim, hidden_dim, output_dim, num_layers)

        # Convert to heterogeneous if metadata provided
        if metadata is not None:
            self.model = to_hetero(self.base_model, metadata, aggr="mean")
        else:
            self.model = None

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        if self.model is None:
            raise ValueError(
                "Model not initialized with metadata. "
                "Please provide metadata in __init__"
            )

        # Extract node features and edge indices from HeteroData
        x_dict = data.x_dict
        edge_index_dict = data.edge_index_dict

        # Run heterogeneous model
        out_dict = self.model(x_dict, edge_index_dict)

        return out_dict


class BaseGNN(nn.Module):
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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # First layer: input_dim -> hidden_dim
        self.convs.append(SAGEConv(input_dim, hidden_dim))
        self.norms.append(nn.LayerNorm(hidden_dim))

        # Middle layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))

        # Last layer: hidden_dim -> output_dim
        self.convs.append(SAGEConv(hidden_dim, output_dim))
        self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = nn.Dropout(0.1)

        # Projection to match dimensions for residual
        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim)
        else:
            self.input_proj = None

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x_orig = x

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.norms[i](x)

            # Apply ReLU to all but last layer
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = self.dropout(x)

        # Add residual connection
        if self.input_proj is not None:
            x_orig = self.input_proj(x_orig)

        x = x + 0.1 * x_orig

        # Normalize embeddings
        x = F.normalize(x, p=2, dim=1)

        return x

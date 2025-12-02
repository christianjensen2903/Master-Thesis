import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero, HeteroConv  # type: ignore
from torch_geometric.data import HeteroData  # type: ignore


class HeteroGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        num_layers: int = 3,
        metadata: tuple | None = None,
        node_input_dims: dict[str, int] | None = None,
    ):
        """
        Heterogeneous GNN that handles different input dimensions per node type.

        Args:
            input_dim: Default input dimension (used if node_input_dims not specified)
            hidden_dim: Hidden dimension for GNN layers
            output_dim: Output dimension (defaults to input_dim)
            num_layers: Number of GNN layers
            metadata: Tuple of (node_types, edge_types)
            node_input_dims: Dict mapping node type to input dimension.
                             If not provided, assumes all node types have input_dim.
        """
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.metadata = metadata

        # Create input projections for each node type
        self.input_projections = nn.ModuleDict()
        if metadata is not None:
            node_types = metadata[0]
            for node_type in node_types:
                in_dim = input_dim
                if node_input_dims is not None and node_type in node_input_dims:
                    in_dim = node_input_dims[node_type]
                self.input_projections[node_type] = nn.Linear(in_dim, hidden_dim)

        # Create a homogeneous base model with hidden_dim as input
        # (all node types are projected to hidden_dim first)
        self.base_model = BaseGNN(hidden_dim, hidden_dim, output_dim, num_layers)

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

        # Project each node type to hidden_dim
        projected_x_dict = {}
        for node_type, x in x_dict.items():
            if node_type in self.input_projections:
                projected_x_dict[node_type] = self.input_projections[node_type](x)
            else:
                projected_x_dict[node_type] = x

        # Run heterogeneous model
        out_dict = self.model(projected_x_dict, edge_index_dict)

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
        self.input_proj: nn.Linear | None = None
        if input_dim != output_dim:
            self.input_proj = nn.Linear(input_dim, output_dim)

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

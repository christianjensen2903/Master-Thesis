import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class CitationGNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.5,
        date_feature_dim: int = 1,
        num_heads: int = 4,
        edge_dim: int = 16,
        degree_embed_dim: int = 32,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.num_heads = num_heads
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        # Query encoder - no graph structure, just MLP
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, output_dim),
        )

        # Candidate encoder - uses graph structure
        self.degree_encoder = nn.Sequential(
            nn.Linear(2, degree_embed_dim),
            nn.GELU(),
            nn.Linear(degree_embed_dim, input_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            self.convs.append(SAGEConv(input_dim, input_dim, aggr="sum"))
            self.norms.append(nn.LayerNorm(input_dim))

        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def compute_degree_features(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute in-degree and out-degree for each node."""
        in_degree = torch.zeros(num_nodes, device=edge_index.device)
        out_degree = torch.zeros(num_nodes, device=edge_index.device)

        ones = torch.ones(edge_index.size(1), device=edge_index.device)
        in_degree.scatter_add_(0, edge_index[1], ones)
        out_degree.scatter_add_(0, edge_index[0], ones)

        return in_degree, out_degree

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        """Encode queries without graph structure."""
        x = self.query_encoder(x)
        x = F.normalize(x, p=2, dim=1)
        return x

    def encode_candidates(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode candidates using graph structure."""
        num_nodes = x.size(0)

        # Compute degree features
        # in_deg, out_deg = self.compute_degree_features(edge_index, num_nodes)

        # degree_feats = torch.stack([in_deg, out_deg], dim=1)
        # degree_feats = torch.log1p(degree_feats)

        # degree_embedding = self.degree_encoder(degree_feats)
        # x = x + degree_embedding

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x_new = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x_new = F.gelu(x_new)
                x_new = self.dropout(x_new)
            x = x + x_new

        # x = self.projector(x)
        x = F.normalize(x, p=2, dim=1)

        return x

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Default forward - encode as candidates."""
        return self.encode_candidates(x, edge_index, date_feature, edge_attr)

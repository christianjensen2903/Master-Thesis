import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class SinusoidalDateEncoder(nn.Module):
    """Encodes a normalized date scalar [0,1] using sinusoidal positional encoding."""

    freqs: torch.Tensor

    def __init__(self, embed_dim: int, max_freq: float = 10.0):
        """
        Args:
            embed_dim: Dimension of the output embedding (must be even)
            max_freq: Maximum frequency multiplier for the [0,1] input range
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim

        # Log-spaced frequencies from 1 to max_freq (for [0,1] normalized input)
        half_dim = embed_dim // 2
        freqs = torch.linspace(0, math.log(max_freq), half_dim).exp() * (2 * math.pi)
        self.register_buffer("freqs", freqs)

    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dates: Tensor of shape (N,) or (N, 1) with normalized dates in [0, 1]
        Returns:
            Tensor of shape (N, embed_dim) with sinusoidal embeddings
        """
        # Handle both (N,) and (N, 1) input shapes
        if dates.dim() == 2:
            dates = dates.squeeze(-1)
        # dates: (N,) -> (N, 1)
        dates = dates.unsqueeze(-1)
        # Compute angles: (N, half_dim)
        angles = dates * self.freqs
        # Sinusoidal encoding: (N, embed_dim)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class DualEncoderGNN(nn.Module):
    """Dual encoder with separate query encoder (MLP) and document encoder (GNN)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.5,
        num_heads: int = 4,
        degree_embed_dim: int = 32,
        num_edge_types: int = 3,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Shared date encoder for both query and document
        # Use same dim as input so we can add directly (preserves relative-time dot product)
        self.date_encoder = SinusoidalDateEncoder(input_dim)
        # Learnable scale - starts small so date is subtle, model learns to amplify
        self.date_scale = nn.Parameter(torch.tensor(0.1))

        # Query encoder: MLP (no graph structure needed since edges are masked)
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

        # Document encoder: GNN (uses graph structure)
        self.degree_encoder = nn.Sequential(
            nn.Linear(2, degree_embed_dim),
            nn.GELU(),
            nn.Linear(degree_embed_dim, input_dim),
        )

        self.edge_type_embedding = nn.Embedding(num_edge_types, input_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(input_dim, input_dim, aggr="mean"))
            self.norms.append(nn.LayerNorm(input_dim))

        self.doc_projector = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def compute_degree_features(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        in_degree = torch.zeros(num_nodes, device=edge_index.device)
        out_degree = torch.zeros(num_nodes, device=edge_index.device)

        ones = torch.ones(edge_index.size(1), device=edge_index.device)
        in_degree.scatter_add_(0, edge_index[1], ones)
        out_degree.scatter_add_(0, edge_index[0], ones)

        return in_degree, out_degree

    def _encode_date(self, date_feature: torch.Tensor | None) -> torch.Tensor | None:
        """Shared date encoding for both query and document."""
        if date_feature is None:
            return None
        date_embed = self.date_encoder(date_feature)
        # Scale but don't project - preserves cos(Δt) relative time in dot products
        return self.date_scale * date_embed

    def _encode_node(
        self, x: torch.Tensor, date_feature: torch.Tensor | None = None
    ) -> torch.Tensor:
        if date_feature is not None:
            x = x + self._encode_date(date_feature)
        return x

    def encode_query(
        self, x: torch.Tensor, date_feature: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode query nodes using MLP (no graph structure)."""
        x = self._encode_node(x, date_feature)
        return F.normalize(x, p=2, dim=1)

    def encode_document(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode document nodes using GNN (with graph structure)."""
        x = self._encode_node(x, date_feature)

        x = self.dropout(x)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x_new = conv(x, edge_index)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        return F.normalize(x, p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass encoding all nodes as documents (for compatibility)."""
        return self.encode_document(x, edge_index, date_feature, edge_attr)

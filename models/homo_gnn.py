import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


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
        degree_embed_dim: int = 32,  # Embedding dim for degree features
        num_edge_types: int = 3,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.num_heads = num_heads

        # Embed in-degree and out-degree counts
        # Using a small MLP to project [in_degree, out_degree] -> degree_embed_dim
        self.degree_encoder = nn.Sequential(
            nn.Linear(2, degree_embed_dim),
            nn.GELU(),
            nn.Linear(degree_embed_dim, input_dim),
        )

        self.edge_type_embedding = nn.Embedding(num_edge_types, input_dim)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            # self.convs.append(SAGEConv(input_dim, input_dim, aggr="mean"))
            self.convs.append(
                GATConv(
                    input_dim,
                    input_dim // num_heads,
                    heads=num_heads,
                    add_self_loops=False,
                )
            )
            self.norms.append(nn.LayerNorm(input_dim))

        self.projector = nn.Sequential(
            nn.Linear(output_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def compute_degree_features(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute in-degree and out-degree for each node."""
        # edge_index[0] = source nodes, edge_index[1] = target nodes
        # in-degree: count of edges where node is target
        # out-degree: count of edges where node is source

        in_degree = torch.zeros(num_nodes, device=edge_index.device)
        out_degree = torch.zeros(num_nodes, device=edge_index.device)

        ones = torch.ones(edge_index.size(1), device=edge_index.device)
        in_degree.scatter_add_(0, edge_index[1], ones)
        out_degree.scatter_add_(0, edge_index[0], ones)

        return in_degree, out_degree

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)

        # Compute degree features
        in_deg, out_deg = self.compute_degree_features(edge_index, num_nodes)

        # Stack and optionally log-transform (helps with skewed distributions)
        degree_feats = torch.stack([in_deg, out_deg], dim=1)  # [num_nodes, 2]
        degree_feats = torch.log1p(degree_feats)  # log(1 + x) for stability

        # Encode and add to node features
        degree_embedding = self.degree_encoder(degree_feats)  # [num_nodes, input_dim]
        x = x + degree_embedding

        for i, conv in enumerate(self.convs):
            edge_type_embedding = self.edge_type_embedding(edge_attr)
            x = self.norms[i](x)
            x_new = conv(x, edge_index, edge_attr=edge_type_embedding)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        x = self.projector(x)
        x = F.normalize(x, p=2, dim=1)

        return x


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

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        """Encode query nodes using MLP (no graph structure)."""
        # out = self.query_encoder(x)
        return F.normalize(x, p=2, dim=1)

    def encode_document(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode document nodes using GNN (with graph structure)."""

        x = self.dropout(x)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x_new = conv(x, edge_index)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        # x = self.doc_projector(x)
        return F.normalize(x, p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass encoding all nodes as documents (for compatibility)."""
        return self.encode_document(x, edge_index, edge_attr)

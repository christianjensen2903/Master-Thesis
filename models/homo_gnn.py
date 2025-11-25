import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv  # type: ignore


class CitationGNN(nn.Module):
    """
    A dual-encoder architecture where:
    - Query encoder: MLP that only uses node features (no message passing)
    - Document encoder: GNN that uses graph structure for context

    This allows queries to be encoded without needing the graph structure,
    while documents benefit from their neighborhood context.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.5,
        date_feature_dim: int = 1,
        query_encoder_layers: int = 2,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Date feature projection (shared or can be separate)
        self.date_projection = nn.Sequential(
            nn.Linear(date_feature_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim),
            nn.GELU(),
        )

        # Query encoder: MLP only (no edges)
        query_layers = []
        for i in range(query_encoder_layers):
            query_layers.extend(
                [
                    nn.Linear(input_dim, input_dim),
                    nn.LayerNorm(input_dim),
                    nn.GELU(),
                    (
                        nn.Dropout(dropout)
                        if i < query_encoder_layers - 1
                        else nn.Identity()
                    ),
                ]
            )
        self.query_encoder = nn.Sequential(*query_layers)

        # Query projector
        self.query_projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        # Document encoder: GNN with message passing
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(input_dim, input_dim, aggr="mean"))
            self.norms.append(nn.LayerNorm(input_dim))

        # Document projector
        self.doc_projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode query nodes using only their features (no graph structure).

        Args:
            x: Node features [num_nodes, input_dim]

        Returns:
            Query embeddings [num_nodes, output_dim]
        """
        # h = self.query_encoder(x)
        # h = self.query_projector(h)
        h = x
        h = F.normalize(h, p=2, dim=1)
        return h

    def encode_documents(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Encode document nodes using GNN with graph structure.

        Args:
            x: Node features [num_nodes, input_dim]
            edge_index: Graph connectivity [2, num_edges]
            date_feature: Optional date features [num_nodes, date_feature_dim]

        Returns:
            Document embeddings [num_nodes, output_dim]
        """
        h = x

        for i, conv in enumerate(self.convs):
            identity = h
            h = conv(h, edge_index)
            h = self.norms[i](h)
            h = F.gelu(h)
            if i < len(self.convs) - 1:
                h = self.dropout(h)
            h = h + identity  # Residual connection

        h = self.doc_projector(h)
        h = F.normalize(h, p=2, dim=1)
        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        return_both: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        Forward pass that returns document embeddings by default.

        For training, you typically want to:
        1. Use encode_query() for anchor nodes
        2. Use encode_documents() for the full graph (positive targets)

        Args:
            x: Node features
            edge_index: Graph connectivity
            date_feature: Optional date features
            return_both: If True, returns both query and document embeddings

        Returns:
            Document embeddings, or dict with both if return_both=True
        """
        doc_emb = self.encode_documents(x, edge_index, date_feature)

        if return_both:
            query_emb = self.encode_query(x)
            return {"query": query_emb, "document": doc_emb}

        return doc_emb

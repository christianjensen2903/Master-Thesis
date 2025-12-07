import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class SimpleHomoGNN(nn.Module):
    """
    Simple homogeneous GNN for legal case retrieval.

    Key features:
    - Residual connections
    - Degree-aware embeddings
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # GNN layers - use separate convolutions per edge type or shared
        self.convs = nn.ModuleList()

        self.convs.append(
            SAGEConv(
                input_dim,
                hidden_dim,
                aggr="mean",
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                    aggr="mean",
                )
            )
        self.convs.append(
            SAGEConv(
                hidden_dim,
                output_dim,
                aggr="mean",
            )
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor,
        edge_attr: torch.Tensor,
        language: torch.Tensor,
        subject_matter: torch.Tensor,
        keywords: torch.Tensor,
        case_law_about: torch.Tensor,
    ) -> torch.Tensor:

        in_feat = x  # Store input for residual connection
        h = x

        # Apply all layers with activation and normalization
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)

            # Apply activation and norm for all layers except the last
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)

        h = h + in_feat
        h = F.normalize(h, p=2, dim=1)

        return h


class SimpleDualHomoGNN(nn.Module):
    """
    Simple homogeneous GNN for legal case retrieval.

    Key features:
    - Residual connections
    - Degree-aware embeddings
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # GNN layers - use separate convolutions per edge type or shared
        self.convs = nn.ModuleList()

        self.convs.append(
            SAGEConv(
                input_dim,
                hidden_dim,
                aggr="mean",
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                SAGEConv(
                    hidden_dim,
                    hidden_dim,
                    aggr="mean",
                )
            )
        self.convs.append(
            SAGEConv(
                hidden_dim,
                output_dim,
                aggr="mean",
            )
        )

        self.dropout = nn.Dropout(dropout)

        self.query_proj = nn.Linear(input_dim, output_dim)
        self.doc_proj = nn.Linear(input_dim, output_dim)

    def encode_query(
        self,
        x: torch.Tensor,
        date_feature: torch.Tensor,
        language: torch.Tensor,
        subject_matter: torch.Tensor,
        keywords: torch.Tensor,
        case_law_about: torch.Tensor,
    ) -> torch.Tensor:
        # x = self.query_proj(x)

        return F.normalize(x, p=2, dim=1)

    def encode_document(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor,
        edge_attr: torch.Tensor,
        language: torch.Tensor,
        subject_matter: torch.Tensor,
        keywords: torch.Tensor,
        case_law_about: torch.Tensor,
    ) -> torch.Tensor:

        in_feat = x  # Store input for residual connection
        h = x

        # Apply all layers with activation and normalization
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)

            # Apply activation and norm for all layers except the last
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = self.dropout(h)

        h = h + in_feat

        h = F.normalize(h, p=2, dim=1)

        return h

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor,
        edge_attr: torch.Tensor,
        language: torch.Tensor,
        subject_matter: torch.Tensor,
        keywords: torch.Tensor,
        case_law_about: torch.Tensor,
    ) -> torch.Tensor:
        return self.encode_document(
            x,
            edge_index,
            date_feature,
            edge_attr,
            language,
            subject_matter,
            keywords,
            case_law_about,
        )

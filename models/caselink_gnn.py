import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv


class CaseLinkGNN(nn.Module):
    """
    CaseLink-inspired GNN for legal case retrieval.

    Key features:
    - Multiple edge type handling
    - Residual connections (like CaseLink)
    - Degree-aware embeddings
    - Optional attention mechanism
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
            GATConv(
                input_dim,
                hidden_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )

        self.convs.append(
            GATConv(
                hidden_dim,
                output_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
        )

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

        h = h + in_feat
        h = F.normalize(h, p=2, dim=1)

        return h

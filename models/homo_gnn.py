import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

from preprocessing.graph_builder import NUM_LANGUAGES


class SinusoidalDateEncoder(nn.Module):
    """Encodes normalized date scalars [0,1] using sinusoidal positional encoding."""

    freqs: torch.Tensor

    def __init__(self, embed_dim: int, num_dates: int = 1, max_freq: float = 10.0):
        """
        Args:
            embed_dim: Dimension of the output embedding per date (must be even)
            num_dates: Number of date features to encode (e.g., 2 for judgment + application)
            max_freq: Maximum frequency multiplier for the [0,1] input range
        """
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.num_dates = num_dates

        # Log-spaced frequencies from 1 to max_freq (for [0,1] normalized input)
        half_dim = embed_dim // 2
        freqs = torch.linspace(0, math.log(max_freq), half_dim).exp() * (2 * math.pi)
        self.register_buffer("freqs", freqs)

    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dates: Tensor of shape (N,), (N, 1), or (N, num_dates) with normalized dates
        Returns:
            Tensor of shape (N, embed_dim * num_dates) with sinusoidal embeddings
        """
        # Handle (N,) input shape -> (N, 1)
        if dates.dim() == 1:
            dates = dates.unsqueeze(-1)

        # dates: (N, num_dates) -> process each date column
        embeddings = []
        for i in range(dates.size(-1)):
            date_col = dates[:, i : i + 1]  # (N, 1)
            angles = date_col * self.freqs  # (N, half_dim)
            emb = torch.cat(
                [torch.sin(angles), torch.cos(angles)], dim=-1
            )  # (N, embed_dim)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=-1)  # (N, embed_dim * num_dates)


class DualEncoderGNN(nn.Module):
    """Dual encoder with separate query encoder (MLP) and document encoder (GNN).

    Language is handled via concatenation: [content_emb, language_emb].
    This is self-contained in the embedding space:

        dot([content, lang], [content', lang']) = content_sim + lang_sim

    The model learns language embeddings where:
    - Same-language pairs have high dot product (homophily)
    - Some languages may have higher norms (citation hubs)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.5,
        num_heads: int = 4,
        num_date_features: int = 3,
        use_language: bool = True,
        language_embed_dim: int = 16,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_date_features = num_date_features
        self.use_language = use_language
        self.language_embed_dim = language_embed_dim if use_language else 0

        # Separate date encoder for each date type (judgment_date, application_date)
        # Each encodes to input_dim to preserve relative-time property in dot products
        self.date_encoder = SinusoidalDateEncoder(input_dim, num_dates=1)
        # Learnable scales for each date type - starts small, model learns to amplify
        # [judgment_date, application_date, duration]
        self.date_scales = nn.Parameter(torch.tensor([0.1, 0.0, 0.0]))

        # Language embedding - concatenated to content (not added)
        if use_language:
            self.language_doc_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)
            self.language_query_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        # Query encoder: MLP (no graph structure needed since edges are masked)
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(input_dim, input_dim, aggr="mean"))
            self.norms.append(nn.LayerNorm(input_dim))

        self.doc_projector = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def _encode_date(self, date_feature: torch.Tensor | None) -> torch.Tensor | None:
        """Encode dates and sum them with learnable scales. Preserves relative-time property."""
        if date_feature is None:
            return None

        # Handle (N,) -> (N, 1) for backward compatibility
        if date_feature.dim() == 1:
            date_feature = date_feature.unsqueeze(-1)

        # Encode each date separately and sum with scales
        # This preserves cos(Δt) property in dot products for each date type
        result = torch.zeros(
            date_feature.size(0), self.input_dim, device=date_feature.device
        )

        for i in range(min(date_feature.size(-1), self.num_date_features)):
            date_col = date_feature[:, i]  # (N,)
            date_emb = self.date_encoder(date_col)  # (N, input_dim)
            scaled_emb = self.date_scales[i] * date_emb
            result = result + scaled_emb

        return result

    def _encode_node(
        self,
        x: torch.Tensor,
        date_feature: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode content features with optional date encoding."""
        if date_feature is not None:
            date_emb = self._encode_date(date_feature)
            assert date_emb is not None
            x = x + date_emb

        return x

    def encode_query(
        self,
        x: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode query nodes. Returns [content, language] normalized."""
        x = self._encode_node(x, date_feature)
        if self.use_language and language is not None:
            # Language is already multihot encoded (N, NUM_LANGUAGES)
            lang_emb = self.language_query_proj(language)  # (N, lang_dim)
            x = torch.cat([x, lang_emb], dim=1)  # (N, content_dim + lang_dim)
        return F.normalize(x, p=2, dim=1)

    def encode_document(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode document nodes using GNN. Returns [content, language] normalized."""
        x = self._encode_node(x, date_feature)
        x = self.dropout(x)

        # GNN layers operate on content (384 dims)
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x_new = conv(x, edge_index)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        if self.use_language and language is not None:
            # Language is already multihot encoded (N, NUM_LANGUAGES)
            lang_emb = self.language_doc_proj(language)  # (N, lang_dim)
            x = torch.cat([x, lang_emb], dim=1)  # (N, content_dim + lang_dim)

        return F.normalize(x, p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass encoding all nodes as documents (for compatibility)."""
        return self.encode_document(x, edge_index, date_feature, edge_attr, language)

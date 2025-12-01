import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

from preprocessing.graph_builder import NUM_LANGUAGES


class WeightedEmbeddingFusion(nn.Module):
    """Fuses multiple embeddings using learned weights.

    Supports two modes:
    - 'scalar': Simple learned scalar weights (w1, w2, w3, w4)
    - 'attention': Weights derived from embeddings via attention mechanism
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_embeddings: int = 4,
        mode: str = "attention",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_embeddings = num_embeddings
        self.mode = mode

        # Project each embedding to output_dim (if needed)
        self.needs_projection = input_dim != output_dim
        if self.needs_projection:
            self.projections = nn.ModuleList(
                [nn.Linear(input_dim, output_dim) for _ in range(num_embeddings)]
            )

        if mode == "scalar":
            # Learned scalar weights initialized uniformly
            self.weights = nn.Parameter(torch.ones(num_embeddings) / num_embeddings)
        elif mode == "attention":
            # Attention-based: derive weights from embeddings
            # Query vector learns what to attend to
            self.query = nn.Parameter(torch.randn(output_dim))
            # Key projection for each embedding type
            self.key_proj = nn.Linear(output_dim, output_dim)
            # Temperature for softmax sharpness
            self.temperature = nn.Parameter(torch.tensor(1.0))
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'scalar' or 'attention'")

    def forward(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            *embeddings: Variable number of embeddings, each (N, input_dim)
        Returns:
            Fused embedding (N, output_dim)
        """
        assert len(embeddings) == self.num_embeddings

        # Project embeddings if needed
        if self.needs_projection:
            projected = [proj(emb) for proj, emb in zip(self.projections, embeddings)]
        else:
            projected = list(embeddings)

        # Stack: (N, num_embeddings, output_dim)
        stacked = torch.stack(projected, dim=1)

        if self.mode == "scalar":
            # Softmax weights for proper mixing
            weights = F.softmax(self.weights, dim=0)  # (num_embeddings,)
            # Weighted sum: (N, output_dim)
            output = torch.einsum("nkd,k->nd", stacked, weights)
        else:  # attention
            # Compute keys: (N, num_embeddings, output_dim)
            keys = self.key_proj(stacked)
            # Attention scores: query · keys
            # (output_dim,) · (N, num_embeddings, output_dim) -> (N, num_embeddings)
            scores = torch.einsum("d,nkd->nk", self.query, keys) / self.temperature
            weights = F.softmax(scores, dim=1)  # (N, num_embeddings)
            # Weighted sum: (N, output_dim)
            output = torch.einsum("nkd,nk->nd", stacked, weights)

        return output

    def get_weights(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """Return the current weights (for analysis/logging)."""
        if self.mode == "scalar":
            return F.softmax(self.weights, dim=0).detach()
        else:
            # Need to compute attention weights
            if self.needs_projection:
                projected = [
                    proj(emb) for proj, emb in zip(self.projections, embeddings)
                ]
            else:
                projected = list(embeddings)
            stacked = torch.stack(projected, dim=1)
            keys = self.key_proj(stacked)
            scores = torch.einsum("d,nkd->nk", self.query, keys) / self.temperature
            return F.softmax(scores, dim=1).detach()


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
    Case metadata (subject_matter, keywords, case_law_about) is also concatenated.

    Final embedding structure: [content, language, metadata]

    This concatenation approach preserves the original content signal while
    adding auxiliary information as separate dimensions. The dot product
    becomes: content_sim + lang_sim + metadata_sim
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        num_heads: int = 4,
        num_date_features: int = 3,
        use_language: bool = True,
        language_embed_dim: int = 16,
        use_case_metadata: bool = True,
        fusion_mode: str = "attention",
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
        self.use_case_metadata = use_case_metadata
        self.fusion_mode = fusion_mode

        # Separate date encoder for each date type (judgment_date, application_date)
        # Each encodes to input_dim to preserve relative-time property in dot products
        self.date_encoder = SinusoidalDateEncoder(output_dim, num_dates=1)
        # Learnable scales for each date type - starts small, model learns to amplify
        # [judgment_date, application_date, duration]
        self.date_scales = nn.Parameter(torch.tensor([0.1, 0.0, 0.0]))

        # Weighted fusion: output = w1*text + w2*keywords + w3*subject + w4*caselaw
        # Preserves embedding space structure since all inputs from same encoder
        self.embedding_fusion = WeightedEmbeddingFusion(
            input_dim=input_dim,
            output_dim=output_dim,
            num_embeddings=4,
            mode=fusion_mode,
        )

        # Language embedding - concatenated to content (not added)
        if use_language:
            self.language_doc_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)
            self.language_query_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        # Query encoder: MLP (no graph structure needed since edges are masked)
        self.query_encoder = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(output_dim, output_dim, aggr="mean"))
            self.norms.append(nn.LayerNorm(output_dim))

        self.doc_projector = nn.Linear(output_dim, output_dim)
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
            date_feature.size(0), self.output_dim, device=date_feature.device
        )

        for i in range(min(date_feature.size(-1), self.num_date_features)):
            date_col = date_feature[:, i]  # (N,)
            date_emb = self.date_encoder(date_col)  # (N, output_dim)
            scaled_emb = self.date_scales[i] * date_emb
            result = result + scaled_emb

        return result

    def _encode_node(
        self,
        x: torch.Tensor,
        date_feature: torch.Tensor,
        subject_matter: torch.Tensor,
        keywords: torch.Tensor,
        case_law_about: torch.Tensor,
    ) -> torch.Tensor:
        """Encode content features with weighted fusion.

        Uses learned weights: output = w1*text + w2*keywords + w3*subject + w4*caselaw
        """
        # Weighted sum preserves embedding space structure
        x = self.embedding_fusion(x, keywords, subject_matter, case_law_about)

        if date_feature is not None:
            date_emb = self._encode_date(date_feature)
            assert date_emb is not None
            x = x + date_emb

        return x

    def get_fusion_weights(
        self,
        x: torch.Tensor,
        keywords: torch.Tensor,
        subject_matter: torch.Tensor,
        case_law_about: torch.Tensor,
    ) -> torch.Tensor:
        """Get current fusion weights for analysis."""
        return self.embedding_fusion.get_weights(
            x, keywords, subject_matter, case_law_about
        )

    def encode_query(
        self,
        x: torch.Tensor,
        date_feature: torch.Tensor,
        language: torch.Tensor,
        subject_matter: torch.Tensor,
        keywords: torch.Tensor,
        case_law_about: torch.Tensor,
    ) -> torch.Tensor:
        """Encode query nodes. Returns [content, language, metadata] normalized."""
        x = self._encode_node(x, date_feature, subject_matter, keywords, case_law_about)

        # Concatenate language embedding
        if self.use_language:
            lang_emb = self.language_query_proj(language)  # (N, lang_dim)
            x = torch.cat([x, lang_emb], dim=1)

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
        """Encode document nodes using GNN. Returns [content, language, metadata] normalized."""
        x = self._encode_node(x, date_feature, subject_matter, keywords, case_law_about)
        x = self.dropout(x)

        # GNN layers operate on content only (preserves semantic signal)
        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x_new = conv(x, edge_index)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        # Concatenate language embedding after GNN
        if self.use_language and language is not None:
            lang_emb = self.language_doc_proj(language)  # (N, lang_dim)
            x = torch.cat([x, lang_emb], dim=1)

        return F.normalize(x, p=2, dim=1)

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
        """Forward pass encoding all nodes as documents (for compatibility)."""
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

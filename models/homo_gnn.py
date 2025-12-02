import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

from preprocessing.graph_builder import NUM_LANGUAGES


class CrossAttentionFusion(nn.Module):
    """Fuses text embedding with metadata using cross-attention.

    Text embedding attends to metadata embeddings (keywords, subject, caselaw),
    allowing the model to learn which metadata is most relevant for each node.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_embeddings: int = 4,  # text + 3 metadata
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads

        # Project text (query) and metadata (key/value) to output_dim
        self.text_proj = nn.Linear(input_dim, output_dim)
        self.metadata_proj = nn.Linear(input_dim, output_dim)

        # Cross-attention: text attends to metadata
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer norms and FFN for post-attention processing
        self.norm1 = nn.LayerNorm(output_dim)
        self.norm2 = nn.LayerNorm(output_dim)

        self.ffn = nn.Sequential(
            nn.Linear(output_dim, output_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 4, output_dim),
            nn.Dropout(dropout),
        )

        # Store last attention weights for analysis
        self._last_attn_weights: torch.Tensor | None = None

    def forward(
        self,
        text_emb: torch.Tensor,
        keywords_emb: torch.Tensor,
        subject_emb: torch.Tensor,
        caselaw_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            text_emb: (N, input_dim) - main text embedding
            keywords_emb: (N, input_dim) - keywords embedding
            subject_emb: (N, input_dim) - subject matter embedding
            caselaw_emb: (N, input_dim) - case law about embedding
        Returns:
            Fused embedding (N, output_dim)
        """
        # Project text as query: (N, output_dim) -> (N, 1, output_dim)
        query = self.text_proj(text_emb).unsqueeze(1)

        # Project and stack metadata as key/value: (N, 3, output_dim)
        metadata = torch.stack(
            [
                self.metadata_proj(keywords_emb),
                self.metadata_proj(subject_emb),
                self.metadata_proj(caselaw_emb),
            ],
            dim=1,
        )

        # Cross-attention: text attends to metadata
        attn_out, attn_weights = self.cross_attn(
            query=query,
            key=metadata,
            value=metadata,
        )
        # attn_out: (N, 1, output_dim), attn_weights: (N, 1, 3)

        # Store attention weights for analysis
        self._last_attn_weights = attn_weights.squeeze(1).detach()  # (N, 3)

        # Residual connection and norm
        x = self.norm1(query + attn_out)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        # Remove sequence dimension: (N, 1, output_dim) -> (N, output_dim)
        return x.squeeze(1)

    def get_weights(self, *embeddings: torch.Tensor) -> torch.Tensor:
        """Return last attention weights (N, 3) for analysis."""
        if self._last_attn_weights is not None:
            return self._last_attn_weights
        # Return uniform if no forward pass yet
        return torch.ones(3) / 3


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

    embedding_fusion: CrossAttentionFusion | WeightedEmbeddingFusion

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
        fusion_mode: str = "scalar",
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

        # Embedding fusion: combine text, keywords, subject, caselaw
        if fusion_mode == "cross_attention":
            self.embedding_fusion = CrossAttentionFusion(
                input_dim=input_dim,
                output_dim=output_dim,
                num_embeddings=4,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(output_dim, output_dim, aggr="mean"))
            self.norms.append(nn.LayerNorm(output_dim))

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


class SymmetricGNN(nn.Module):
    """Symmetric GNN encoder - uses the same GNN architecture for both query and document.

    Unlike DualEncoderGNN which uses MLP for queries and GNN for documents,
    this model applies GNN message passing to both query and document nodes.
    The GNN weights are shared between query and document encoders.
    """

    embedding_fusion: CrossAttentionFusion | WeightedEmbeddingFusion

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
        fusion_mode: str = "scalar",
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

        # Date encoder (shared)
        self.date_encoder = SinusoidalDateEncoder(output_dim, num_dates=1)
        self.date_scales = nn.Parameter(torch.tensor([0.1, 0.0, 0.0]))

        # Embedding fusion (shared between query and document)
        if fusion_mode == "cross_attention":
            self.embedding_fusion = CrossAttentionFusion(
                input_dim=input_dim,
                output_dim=output_dim,
                num_embeddings=4,
                num_heads=num_heads,
                dropout=dropout,
            )
        else:
            self.embedding_fusion = WeightedEmbeddingFusion(
                input_dim=input_dim,
                output_dim=output_dim,
                num_embeddings=4,
                mode=fusion_mode,
            )

        # Language embedding (shared projection)
        if use_language:
            self.language_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        # Shared GNN layers for both query and document
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.convs.append(SAGEConv(output_dim, output_dim, aggr="mean"))
            self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = nn.Dropout(dropout)

    def _encode_date(self, date_feature: torch.Tensor | None) -> torch.Tensor | None:
        """Encode dates and sum them with learnable scales."""
        if date_feature is None:
            return None

        if date_feature.dim() == 1:
            date_feature = date_feature.unsqueeze(-1)

        result = torch.zeros(
            date_feature.size(0), self.output_dim, device=date_feature.device
        )

        for i in range(min(date_feature.size(-1), self.num_date_features)):
            date_col = date_feature[:, i]
            date_emb = self.date_encoder(date_col)
            scaled_emb = self.date_scales[i] * date_emb
            result = result + scaled_emb

        return result

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
        """Forward pass encoding all nodes using GNN."""
        x = self.embedding_fusion(x, keywords, subject_matter, case_law_about)

        if date_feature is not None:
            date_emb = self._encode_date(date_feature)
            assert date_emb is not None
            x = x + date_emb

        x = self.dropout(x)

        for i, conv in enumerate(self.convs):
            x = self.norms[i](x)
            x_new = conv(x, edge_index)
            x_new = F.gelu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        if self.use_language and language is not None:
            lang_emb = self.language_proj(language)
            x = torch.cat([x, lang_emb], dim=1)

        return F.normalize(x, p=2, dim=1)

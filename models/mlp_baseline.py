import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing.graph_builder import NUM_LANGUAGES
from .homo_gnn import (
    CrossAttentionFusion,
    WeightedEmbeddingFusion,
    SinusoidalDateEncoder,
)


class SymmetricMLPBaseline(nn.Module):
    """Symmetric MLP baseline without graph structure for comparison with GNN models."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.3,
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
        self.num_layers = num_layers

        # Date encoder (same as GNN)
        self.date_encoder = SinusoidalDateEncoder(output_dim, num_dates=1)
        self.date_scales = nn.Parameter(torch.tensor([0.1, 0.0, 0.0]))

        # Embedding fusion (same as GNN)
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

        # Language embedding (same as GNN)
        if use_language:
            self.language_doc_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)
            self.language_query_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        # MLP layers instead of GNN layers
        self.mlp_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(output_dim, output_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(output_dim * 2, output_dim),
                )
            )
            self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = nn.Dropout(dropout)

    def _encode_date(self, date_feature: torch.Tensor | None) -> torch.Tensor | None:
        """Encode dates with learnable scales (same as GNN)."""
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

    def _encode_node(
        self,
        x: torch.Tensor,
        date_feature: torch.Tensor | None,
        subject_matter: torch.Tensor | None,
        keywords: torch.Tensor | None,
        case_law_about: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode node features with metadata fusion (same as GNN)."""
        # Handle missing metadata with zeros
        if subject_matter is None:
            subject_matter = torch.zeros_like(x)
        if keywords is None:
            keywords = torch.zeros_like(x)
        if case_law_about is None:
            case_law_about = torch.zeros_like(x)

        x = self.embedding_fusion(x, keywords, subject_matter, case_law_about)

        if date_feature is not None:
            date_emb = self._encode_date(date_feature)
            if date_emb is not None:
                date_emb = date_emb - (date_emb * x).sum(dim=-1, keepdim=True) * x
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

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
        language: torch.Tensor | None,
        subject_matter: torch.Tensor | None,
        keywords: torch.Tensor | None,
        case_law_about: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass encoding all nodes as documents."""
        x = self._encode_node(x, date_feature, subject_matter, keywords, case_law_about)
        # Normalize the output
        return F.normalize(x, p=2, dim=1)


class MLPBaseline(nn.Module):
    """MLP baseline without graph structure for comparison with GNN models.

    Uses the same feature processing (date encoding, language, metadata fusion)
    but replaces GNN layers with MLP layers. This isolates the contribution
    of graph structure in the GNN models.

    Implements dual encoder interface for compatibility with existing trainer/evaluator.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.3,
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
        self.num_layers = num_layers

        # Date encoder (same as GNN)
        self.date_encoder = SinusoidalDateEncoder(output_dim, num_dates=1)
        self.date_scales = nn.Parameter(torch.tensor([0.1, 0.0, 0.0]))

        # Embedding fusion (same as GNN)
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

        # Language embedding (same as GNN)
        if use_language:
            self.language_doc_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)
            self.language_query_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        # MLP layers instead of GNN layers
        self.mlp_layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        for _ in range(num_layers):
            self.mlp_layers.append(
                nn.Sequential(
                    nn.Linear(output_dim, output_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(output_dim * 2, output_dim),
                )
            )
            self.norms.append(nn.LayerNorm(output_dim))

        self.dropout = nn.Dropout(dropout)

    def _encode_date(self, date_feature: torch.Tensor | None) -> torch.Tensor | None:
        """Encode dates with learnable scales (same as GNN)."""
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

    def _encode_node(
        self,
        x: torch.Tensor,
        date_feature: torch.Tensor | None,
        subject_matter: torch.Tensor | None,
        keywords: torch.Tensor | None,
        case_law_about: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode node features with metadata fusion (same as GNN)."""
        # Handle missing metadata with zeros
        if subject_matter is None:
            subject_matter = torch.zeros_like(x)
        if keywords is None:
            keywords = torch.zeros_like(x)
        if case_law_about is None:
            case_law_about = torch.zeros_like(x)

        x = self.embedding_fusion(x, keywords, subject_matter, case_law_about)

        if date_feature is not None:
            date_emb = self._encode_date(date_feature)
            if date_emb is not None:
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
        date_feature: torch.Tensor | None,
        language: torch.Tensor | None,
        subject_matter: torch.Tensor | None,
        keywords: torch.Tensor | None,
        case_law_about: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode query nodes (no MLP layers, same as GNN query encoder)."""
        x = self._encode_node(x, date_feature, subject_matter, keywords, case_law_about)

        if self.use_language and language is not None:
            lang_emb = self.language_query_proj(language)
            x = torch.cat([x, lang_emb], dim=1)

        return F.normalize(x, p=2, dim=1)

    def encode_document(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,  # Ignored - no graph structure
        date_feature: torch.Tensor | None,
        edge_attr: torch.Tensor | None,  # Ignored
        language: torch.Tensor | None,
        subject_matter: torch.Tensor | None,
        keywords: torch.Tensor | None,
        case_law_about: torch.Tensor | None,
    ) -> torch.Tensor:
        """Encode document nodes using MLP (no graph structure)."""
        x = self._encode_node(x, date_feature, subject_matter, keywords, case_law_about)
        x = self.dropout(x)

        # MLP layers instead of GNN
        for i, mlp in enumerate(self.mlp_layers):
            x = self.norms[i](x)
            x_new = mlp(x)
            x_new = self.dropout(x_new)
            x = x + x_new

        if self.use_language and language is not None:
            lang_emb = self.language_doc_proj(language)
            x = torch.cat([x, lang_emb], dim=1)

        return F.normalize(x, p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None,
        edge_attr: torch.Tensor | None,
        language: torch.Tensor | None,
        subject_matter: torch.Tensor | None,
        keywords: torch.Tensor | None,
        case_law_about: torch.Tensor | None,
    ) -> torch.Tensor:
        """Forward pass encoding all nodes as documents."""
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

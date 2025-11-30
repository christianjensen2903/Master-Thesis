import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv

from preprocessing.graph_builder import NUM_LANGUAGES


class CaseMetadataEncoder(nn.Module):
    """Encodes case-level metadata (subject_matter, keywords, case_law_about) via concatenation.

    Projects each metadata type to a smaller dimension and concatenates them.
    This keeps the original content embedding intact and adds metadata as separate dimensions,
    similar to how language embeddings are handled.

    Output: [content, metadata_proj] where metadata_proj is the concatenated projected metadata.
    """

    def __init__(
        self,
        metadata_dim: int = 384,
        output_dim_per_type: int = 128,
    ):
        """
        Args:
            metadata_dim: Dimension of each input metadata embedding
            output_dim_per_type: Output dimension for each metadata type (total = 3 * this)
        """
        super().__init__()
        self.metadata_dim = metadata_dim
        self.output_dim_per_type = output_dim_per_type
        self.total_output_dim = output_dim_per_type * 3

        # Project each metadata type to smaller dimension
        self.subject_matter_proj = nn.Linear(metadata_dim, output_dim_per_type)
        self.keywords_proj = nn.Linear(metadata_dim, output_dim_per_type)
        self.case_law_about_proj = nn.Linear(metadata_dim, output_dim_per_type)

    def forward(
        self,
        subject_matter: torch.Tensor | None = None,
        keywords: torch.Tensor | None = None,
        case_law_about: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """
        Encode and concatenate metadata embeddings.

        Args:
            subject_matter: Subject matter embeddings (N, metadata_dim) or None
            keywords: Keywords embeddings (N, metadata_dim) or None
            case_law_about: Case law about embeddings (N, metadata_dim) or None

        Returns:
            Concatenated metadata embeddings (N, total_output_dim) or None if all inputs are None
        """
        if subject_matter is None and keywords is None and case_law_about is None:
            return None

        # Get batch size from first non-None input
        batch_size = (
            subject_matter
            if subject_matter is not None
            else keywords if keywords is not None else case_law_about
        ).size(0)
        device = (
            subject_matter
            if subject_matter is not None
            else keywords if keywords is not None else case_law_about
        ).device

        # Project each metadata type (use zeros if None)
        if subject_matter is not None:
            sm_proj = self.subject_matter_proj(subject_matter)
        else:
            sm_proj = torch.zeros(batch_size, self.output_dim_per_type, device=device)

        if keywords is not None:
            kw_proj = self.keywords_proj(keywords)
        else:
            kw_proj = torch.zeros(batch_size, self.output_dim_per_type, device=device)

        if case_law_about is not None:
            cla_proj = self.case_law_about_proj(case_law_about)
        else:
            cla_proj = torch.zeros(batch_size, self.output_dim_per_type, device=device)

        # Concatenate all projected metadata
        return torch.cat([sm_proj, kw_proj, cla_proj], dim=-1)


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
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.5,
        num_heads: int = 4,
        num_date_features: int = 3,
        use_language: bool = True,
        language_embed_dim: int = 16,
        use_case_metadata: bool = True,
        metadata_dim: int = 384,
        metadata_output_dim_per_type: int = 128,
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
        self.metadata_output_dim = (
            metadata_output_dim_per_type * 3 if use_case_metadata else 0
        )

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

        # Case metadata encoder - projects and concatenates metadata
        if use_case_metadata:
            self.metadata_encoder = CaseMetadataEncoder(
                metadata_dim=metadata_dim,
                output_dim_per_type=metadata_output_dim_per_type,
            )

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
        subject_matter: torch.Tensor | None = None,
        keywords: torch.Tensor | None = None,
        case_law_about: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode query nodes. Returns [content, language, metadata] normalized."""
        x = self._encode_node(x, date_feature)

        # Concatenate language embedding
        if self.use_language and language is not None:
            lang_emb = self.language_query_proj(language)  # (N, lang_dim)
            x = torch.cat([x, lang_emb], dim=1)

        # Concatenate metadata embedding (keeps content intact)
        if self.use_case_metadata:
            metadata_emb = self.metadata_encoder(
                subject_matter=subject_matter,
                keywords=keywords,
                case_law_about=case_law_about,
            )
            if metadata_emb is not None:
                x = torch.cat([x, metadata_emb], dim=1)
            else:
                # Pad with zeros if no metadata
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            x.size(0), self.metadata_output_dim, device=x.device
                        ),
                    ],
                    dim=1,
                )

        return F.normalize(x, p=2, dim=1)

    def encode_document(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
        subject_matter: torch.Tensor | None = None,
        keywords: torch.Tensor | None = None,
        case_law_about: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode document nodes using GNN. Returns [content, language, metadata] normalized."""
        x = self._encode_node(x, date_feature)
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

        # Concatenate metadata embedding after GNN (keeps content intact)
        if self.use_case_metadata:
            metadata_emb = self.metadata_encoder(
                subject_matter=subject_matter,
                keywords=keywords,
                case_law_about=case_law_about,
            )
            if metadata_emb is not None:
                x = torch.cat([x, metadata_emb], dim=1)
            else:
                # Pad with zeros if no metadata
                x = torch.cat(
                    [
                        x,
                        torch.zeros(
                            x.size(0), self.metadata_output_dim, device=x.device
                        ),
                    ],
                    dim=1,
                )

        return F.normalize(x, p=2, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        language: torch.Tensor | None = None,
        subject_matter: torch.Tensor | None = None,
        keywords: torch.Tensor | None = None,
        case_law_about: torch.Tensor | None = None,
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

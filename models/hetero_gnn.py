import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, HeteroConv, HGTConv
from torch_geometric.data import HeteroData

from preprocessing.graph_builder import NUM_LANGUAGES


class SinusoidalDateEncoder(nn.Module):
    """Encodes normalized date scalars [0,1] using sinusoidal positional encoding."""

    freqs: torch.Tensor

    def __init__(self, embed_dim: int, num_dates: int = 1, max_freq: float = 10.0):
        super().__init__()
        assert embed_dim % 2 == 0, "embed_dim must be even"
        self.embed_dim = embed_dim
        self.num_dates = num_dates

        half_dim = embed_dim // 2
        freqs = torch.linspace(0, torch.log(torch.tensor(max_freq)), half_dim).exp() * (
            2 * torch.pi
        )
        self.register_buffer("freqs", freqs)

    def forward(self, dates: torch.Tensor) -> torch.Tensor:
        if dates.dim() == 1:
            dates = dates.unsqueeze(-1)

        embeddings = []
        for i in range(dates.size(-1)):
            date_col = dates[:, i : i + 1]
            angles = date_col * self.freqs
            emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=-1)


class CaseFeatureEncoder(nn.Module):
    """Encodes case-level features with learned fusion of metadata embeddings.

    Case features from graph_builder: [date(3), language(23), subject_matter, keywords, case_law_about]
    """

    def __init__(
        self,
        metadata_embed_dim: int,
        output_dim: int,
        date_embed_dim: int = 64,
        language_embed_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.metadata_embed_dim = metadata_embed_dim
        self.output_dim = output_dim

        # Date encoding (3 date features)
        self.date_encoder = SinusoidalDateEncoder(date_embed_dim, num_dates=3)
        self.date_scales = nn.Parameter(torch.tensor([0.1, 0.0, 0.0]))

        # Language projection
        self.language_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        # Metadata projections (each to output_dim for fusion)
        self.subject_matter_proj = nn.Linear(metadata_embed_dim, output_dim)
        self.keywords_proj = nn.Linear(metadata_embed_dim, output_dim)
        self.case_law_about_proj = nn.Linear(metadata_embed_dim, output_dim)

        # Learnable attention weights for metadata fusion
        self.metadata_query = nn.Parameter(torch.randn(output_dim))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # Final projection combining date + language + fused metadata
        combined_dim = date_embed_dim + language_embed_dim + output_dim
        self.output_proj = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, case_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            case_features: (N, 3 + 23 + 3*metadata_embed_dim) case feature tensor
        Returns:
            (N, output_dim) encoded case features
        """
        # Split case features into components
        date_features = case_features[:, :3]
        language_features = case_features[:, 3 : 3 + NUM_LANGUAGES]
        metadata_start = 3 + NUM_LANGUAGES

        subject_matter = case_features[
            :, metadata_start : metadata_start + self.metadata_embed_dim
        ]
        keywords = case_features[
            :,
            metadata_start
            + self.metadata_embed_dim : metadata_start
            + 2 * self.metadata_embed_dim,
        ]
        case_law_about = case_features[
            :, metadata_start + 2 * self.metadata_embed_dim :
        ]

        # Encode dates with sinusoidal encoding and learnable scales
        date_emb = torch.zeros(
            date_features.size(0),
            self.date_encoder.embed_dim,
            device=case_features.device,
        )
        for i in range(3):
            single_date_emb = self.date_encoder(date_features[:, i : i + 1].squeeze(-1))
            # Only use first date_encoder.embed_dim // 3 dims per date
            date_emb = date_emb + self.date_scales[i] * single_date_emb

        # Encode language
        lang_emb = self.language_proj(language_features)

        # Project metadata embeddings
        sm_proj = self.subject_matter_proj(subject_matter)
        kw_proj = self.keywords_proj(keywords)
        cla_proj = self.case_law_about_proj(case_law_about)

        # Attention-based fusion of metadata
        metadata_stack = torch.stack(
            [sm_proj, kw_proj, cla_proj], dim=1
        )  # (N, 3, output_dim)
        scores = (
            torch.einsum("d,nkd->nk", self.metadata_query, metadata_stack)
            / self.temperature
        )
        weights = F.softmax(scores, dim=1)  # (N, 3)
        fused_metadata = torch.einsum(
            "nkd,nk->nd", metadata_stack, weights
        )  # (N, output_dim)

        # Combine all components
        combined = torch.cat([date_emb, lang_emb, fused_metadata], dim=-1)

        return self.output_proj(combined)


class HeteroGNNLayer(nn.Module):
    """Single heterogeneous message passing layer using HeteroConv."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        conv_type: str = "sage",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Build convolutions for each edge type
        convs = {}

        if conv_type == "gat":
            # Use GAT for all edge types
            for edge_type in [
                ("paragraph", "cites", "paragraph"),
                ("paragraph", "next", "paragraph"),
            ]:
                convs[edge_type] = GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                )
            # Bipartite edges need tuple input dims
            convs[("paragraph", "belongs_to", "case")] = SAGEConv(
                (hidden_dim, hidden_dim), hidden_dim, aggr="mean"
            )
            convs[("case", "contains", "paragraph")] = SAGEConv(
                (hidden_dim, hidden_dim), hidden_dim, aggr="mean"
            )
        else:
            # Default: SAGEConv for all edge types
            convs[("paragraph", "cites", "paragraph")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean"
            )
            convs[("paragraph", "next", "paragraph")] = SAGEConv(
                hidden_dim, hidden_dim, aggr="mean"
            )
            convs[("paragraph", "belongs_to", "case")] = SAGEConv(
                (hidden_dim, hidden_dim), hidden_dim, aggr="mean"
            )
            convs[("case", "contains", "paragraph")] = SAGEConv(
                (hidden_dim, hidden_dim), hidden_dim, aggr="mean"
            )

        self.conv = HeteroConv(convs, aggr="sum")

        # Layer norms for each node type
        self.norms = nn.ModuleDict(
            {
                "paragraph": nn.LayerNorm(hidden_dim),
                "case": nn.LayerNorm(hidden_dim),
            }
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Apply pre-normalization
        x_normed = {k: self.norms[k](v) for k, v in x_dict.items() if k in self.norms}

        # Message passing
        out_dict = self.conv(x_normed, edge_index_dict)

        # Apply activation and dropout, then residual
        result = {}
        for node_type in x_dict:
            if node_type in out_dict:
                h = F.gelu(out_dict[node_type])
                h = self.dropout(h)
                result[node_type] = x_dict[node_type] + h
            else:
                result[node_type] = x_dict[node_type]

        return result


class HGTLayer(nn.Module):
    """Heterogeneous Graph Transformer layer.

    HGT uses type-specific transformations and relation-aware attention,
    making it theoretically better suited for heterogeneous graphs with
    semantically different edge types (citations vs sequential vs hierarchical).
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        metadata: tuple[list[str], list[tuple[str, str, str]]] | None = None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Default metadata if not provided
        if metadata is None:
            metadata = (
                ["paragraph", "case"],
                [
                    ("paragraph", "cites", "paragraph"),
                    ("paragraph", "next", "paragraph"),
                    ("paragraph", "belongs_to", "case"),
                    ("case", "contains", "paragraph"),
                ],
            )

        self.conv = HGTConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            metadata=metadata,
            heads=num_heads,
        )

        # Layer norms for each node type
        self.norms = nn.ModuleDict(
            {node_type: nn.LayerNorm(hidden_dim) for node_type in metadata[0]}
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Apply pre-normalization
        x_normed = {k: self.norms[k](v) for k, v in x_dict.items() if k in self.norms}

        # HGT message passing
        out_dict = self.conv(x_normed, edge_index_dict)

        # Apply activation and dropout, then residual
        result = {}
        for node_type in x_dict:
            if node_type in out_dict:
                h = F.gelu(out_dict[node_type])
                h = self.dropout(h)
                result[node_type] = x_dict[node_type] + h
            else:
                result[node_type] = x_dict[node_type]

        return result


class HeteroDualEncoderGNN(nn.Module):
    """Heterogeneous dual encoder GNN for legal citation prediction.

    Architecture:
    - Paragraph nodes: text embeddings projected to hidden_dim
    - Case nodes: metadata (date, language, subject_matter, keywords, case_law_about)
      encoded and projected to hidden_dim
    - Message passing through heterogeneous edges
    - Language concatenated to final paragraph embedding

    Dual encoder: separate query (MLP) and document (GNN) encoders.
    """

    def __init__(
        self,
        paragraph_input_dim: int,
        case_metadata_dim: int,  # dimension of each metadata embedding (e.g., 1024)
        hidden_dim: int = 256,
        output_dim: int | None = None,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        conv_type: str = "sage",
        use_language: bool = True,
        language_embed_dim: int = 16,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim

        self.paragraph_input_dim = paragraph_input_dim
        self.case_metadata_dim = case_metadata_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_language = use_language
        self.language_embed_dim = language_embed_dim if use_language else 0

        # Paragraph input projection
        self.paragraph_proj = nn.Sequential(
            nn.Linear(paragraph_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Case feature encoder
        self.case_encoder = CaseFeatureEncoder(
            metadata_embed_dim=case_metadata_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # Heterogeneous GNN layers
        self.layers = nn.ModuleList()
        self.conv_type = conv_type

        for _ in range(num_layers):
            if conv_type == "hgt":
                self.layers.append(
                    HGTLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )
            else:
                self.layers.append(
                    HeteroGNNLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        conv_type=conv_type,
                    )
                )

        # Output projection for paragraphs
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Language projections (separate for query and document)
        if use_language:
            self.language_query_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)
            self.language_doc_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _get_language_from_case(
        self, data: HeteroData, paragraph_indices: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Extract language features for paragraphs from their parent cases."""
        if not self.use_language:
            return None

        case_features = data["case"].x
        language_features = case_features[:, 3 : 3 + NUM_LANGUAGES]

        # Get paragraph -> case mapping
        if ("paragraph", "belongs_to", "case") not in data.edge_types:
            return None

        par_to_case = data["paragraph", "belongs_to", "case"].edge_index
        num_paragraphs = data["paragraph"].x.size(0)

        # Vectorized: scatter language features from cases to paragraphs
        par_language = torch.zeros(
            num_paragraphs, NUM_LANGUAGES, device=case_features.device
        )
        par_language[par_to_case[0]] = language_features[par_to_case[1]]

        if paragraph_indices is not None:
            return par_language[paragraph_indices]
        return par_language

    @property
    def embedding_dim(self) -> int:
        """Return the effective output embedding dimension."""
        return self.output_dim + self.language_embed_dim

    def encode_query(
        self,
        data: HeteroData,
        paragraph_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode query paragraphs (no message passing, just projection).

        For queries, we use x_query embeddings and only apply MLP.
        """
        x = (
            data["paragraph"].x_query
            if hasattr(data["paragraph"], "x_query")
            else data["paragraph"].x
        )

        if paragraph_indices is not None:
            x = x[paragraph_indices]

        # Project to hidden dim and then output dim
        h = self.paragraph_proj(x)
        h = self.output_proj(h)

        # Concatenate language embedding
        if self.use_language:
            lang = self._get_language_from_case(data, paragraph_indices)
            if lang is not None:
                lang_emb = self.language_query_proj(lang)
                h = torch.cat([h, lang_emb], dim=-1)

        return F.normalize(h, p=2, dim=-1)

    def encode_document(self, data: HeteroData) -> torch.Tensor:
        """Encode document paragraphs using GNN message passing."""
        # Project inputs
        x_dict = {
            "paragraph": self.paragraph_proj(data["paragraph"].x),
            "case": self.case_encoder(data["case"].x),
        }

        # Get edge indices (only use edges that exist in the batch)
        edge_index_dict = {}
        for edge_type in [
            ("paragraph", "cites", "paragraph"),
            ("paragraph", "next", "paragraph"),
            ("paragraph", "belongs_to", "case"),
            ("case", "contains", "paragraph"),
        ]:
            if edge_type in data.edge_types:
                edge_index_dict[edge_type] = data[edge_type].edge_index

        # Apply GNN layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        # Output projection for paragraphs
        h = self.output_proj(x_dict["paragraph"])

        # Concatenate language embedding
        if self.use_language:
            lang = self._get_language_from_case(data)
            if lang is not None:
                lang_emb = self.language_doc_proj(lang)
                h = torch.cat([h, lang_emb], dim=-1)

        return F.normalize(h, p=2, dim=-1)

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Forward pass - encodes all paragraphs as documents.

        Returns dict with 'paragraph' embeddings for compatibility with trainer.
        """
        return {"paragraph": self.encode_document(data)}


class HeteroSymmetricGNN(nn.Module):
    """Symmetric heterogeneous GNN - uses same GNN for both query and document.

    Unlike HeteroDualEncoderGNN, this applies GNN message passing to both
    query and document paragraphs. Weights are shared.
    """

    def __init__(
        self,
        paragraph_input_dim: int,
        case_metadata_dim: int,
        hidden_dim: int = 256,
        output_dim: int | None = None,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.3,
        conv_type: str = "sage",
        use_language: bool = True,
        language_embed_dim: int = 16,
    ):
        super().__init__()
        if output_dim is None:
            output_dim = hidden_dim

        self.paragraph_input_dim = paragraph_input_dim
        self.case_metadata_dim = case_metadata_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_language = use_language
        self.language_embed_dim = language_embed_dim if use_language else 0

        # Paragraph input projection
        self.paragraph_proj = nn.Sequential(
            nn.Linear(paragraph_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Case feature encoder
        self.case_encoder = CaseFeatureEncoder(
            metadata_embed_dim=case_metadata_dim,
            output_dim=hidden_dim,
            dropout=dropout,
        )

        # Heterogeneous GNN layers
        self.layers = nn.ModuleList()
        self.conv_type = conv_type

        for _ in range(num_layers):
            if conv_type == "hgt":
                self.layers.append(
                    HGTLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                    )
                )
            else:
                self.layers.append(
                    HeteroGNNLayer(
                        hidden_dim=hidden_dim,
                        num_heads=num_heads,
                        dropout=dropout,
                        conv_type=conv_type,
                    )
                )

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        # Language projection (shared)
        if use_language:
            self.language_proj = nn.Linear(NUM_LANGUAGES, language_embed_dim)

        self.dropout = nn.Dropout(dropout)

    def _get_language_from_case(
        self, data: HeteroData, paragraph_indices: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        """Extract language features for paragraphs from their parent cases."""
        if not self.use_language:
            return None

        case_features = data["case"].x
        language_features = case_features[:, 3 : 3 + NUM_LANGUAGES]

        if ("paragraph", "belongs_to", "case") not in data.edge_types:
            return None

        par_to_case = data["paragraph", "belongs_to", "case"].edge_index
        num_paragraphs = data["paragraph"].x.size(0)

        # Vectorized: scatter language features from cases to paragraphs
        par_language = torch.zeros(
            num_paragraphs, NUM_LANGUAGES, device=case_features.device
        )
        par_language[par_to_case[0]] = language_features[par_to_case[1]]

        if paragraph_indices is not None:
            return par_language[paragraph_indices]
        return par_language

    @property
    def embedding_dim(self) -> int:
        """Return the effective output embedding dimension."""
        return self.output_dim + self.language_embed_dim

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        """Forward pass - encodes all paragraphs using GNN."""
        # Project inputs
        x_dict = {
            "paragraph": self.paragraph_proj(data["paragraph"].x),
            "case": self.case_encoder(data["case"].x),
        }

        # Get edge indices
        edge_index_dict = {}
        for edge_type in [
            ("paragraph", "cites", "paragraph"),
            ("paragraph", "next", "paragraph"),
            ("paragraph", "belongs_to", "case"),
            ("case", "contains", "paragraph"),
        ]:
            if edge_type in data.edge_types:
                edge_index_dict[edge_type] = data[edge_type].edge_index

        # Apply GNN layers
        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        # Output projection
        h = self.output_proj(x_dict["paragraph"])

        # Concatenate language
        if self.use_language:
            lang = self._get_language_from_case(data)
            if lang is not None:
                lang_emb = self.language_proj(lang)
                h = torch.cat([h, lang_emb], dim=-1)

        return {"paragraph": F.normalize(h, p=2, dim=-1)}


def create_hetero_model(
    graph_data: HeteroData,
    model_type: str = "dual",
    hidden_dim: int = 256,
    output_dim: int | None = None,
    num_layers: int = 3,
    num_heads: int = 4,
    dropout: float = 0.3,
    conv_type: str = "sage",
    use_language: bool = True,
    language_embed_dim: int = 16,
) -> HeteroDualEncoderGNN | HeteroSymmetricGNN:
    """Factory function to create heterogeneous GNN model.

    Args:
        graph_data: HeteroData object to infer dimensions from
        model_type: "dual" or "symmetric"
        hidden_dim: Hidden dimension for GNN layers
        output_dim: Output embedding dimension (defaults to hidden_dim)
        num_layers: Number of GNN layers
        num_heads: Number of attention heads (for GAT/HGT)
        dropout: Dropout rate
        conv_type: Convolution type:
            - "sage": GraphSAGE (simple, fast)
            - "gat": Graph Attention Network (attention over neighbors)
            - "hgt": Heterogeneous Graph Transformer (type-specific attention,
                     best theoretical fit for heterogeneous graphs)
        use_language: Whether to concatenate language embedding
        language_embed_dim: Dimension of language embedding

    Returns:
        Configured heterogeneous GNN model
    """
    paragraph_input_dim = graph_data["paragraph"].x.size(1)
    case_feature_dim = graph_data["case"].x.size(1)

    # Calculate metadata embedding dimension
    # Case features: [date(3), language(23), subject_matter, keywords, case_law_about]
    case_metadata_dim = (case_feature_dim - 3 - NUM_LANGUAGES) // 3

    print(f"Inferred dimensions:")
    print(f"  Paragraph input dim: {paragraph_input_dim}")
    print(f"  Case feature dim: {case_feature_dim}")
    print(f"  Case metadata dim (per field): {case_metadata_dim}")

    if model_type == "dual":
        return HeteroDualEncoderGNN(
            paragraph_input_dim=paragraph_input_dim,
            case_metadata_dim=case_metadata_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            conv_type=conv_type,
            use_language=use_language,
            language_embed_dim=language_embed_dim,
        )
    elif model_type == "symmetric":
        return HeteroSymmetricGNN(
            paragraph_input_dim=paragraph_input_dim,
            case_metadata_dim=case_metadata_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            conv_type=conv_type,
            use_language=use_language,
            language_embed_dim=language_embed_dim,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'dual' or 'symmetric'")

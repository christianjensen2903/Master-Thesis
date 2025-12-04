import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn
import torch
from torch_geometric.data import Data, HeteroData  # type: ignore
from torch_geometric.utils import add_self_loops  # type: ignore

# Fix OpenMP conflict on macOS (FAISS and PyTorch may use different OpenMP runtimes)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss

# Set FAISS to single-threaded mode to avoid segmentation faults
faiss.omp_set_num_threads(1)

# Language vocabulary for authentic language feature
# Based on frequency analysis of training data (year < 2018):
# Only includes languages with >= 500 training samples (sufficient to learn embeddings)
# UNKNOWN: missing, invalid, or rare languages (<500 samples: MLT, HRV)
LANGUAGE_VOCAB = [
    "UNKNOWN",  # Index 0: Unknown/None/rare languages
    "MULTI",  # Index 1: Multi-language cases (~1.5%)
    "DEU",  # Index 2: German (23.06%)
    "FRA",  # Index 3: French (18.22%)
    "ENG",  # Index 4: English (14.27%)
    "NLD",  # Index 5: Dutch (10.87%)
    "ITA",  # Index 6: Italian (10.43%)
    "SPA",  # Index 7: Spanish (5.56%)
    "ELL",  # Index 8: Greek (3.15%)
    "POR",  # Index 9: Portuguese (2.12%)
    "DAN",  # Index 10: Danish (1.68%)
    "POL",  # Index 11: Polish (1.47%)
    "SWE",  # Index 12: Swedish (1.37%)
    "FIN",  # Index 13: Finnish (1.21%)
    "HUN",  # Index 14: Hungarian (1.18%)
    "BUL",  # Index 15: Bulgarian (0.93%)
    "RON",  # Index 16: Romanian (0.60%)
    "CES",  # Index 17: Czech (0.52%)
    "LAV",  # Index 18: Latvian (0.51%)
    "LIT",  # Index 19: Lithuanian (0.44%)
    "SLK",  # Index 20: Slovak (0.35%)
    "SLV",  # Index 21: Slovenian (0.25%)
    "EST",  # Index 22: Estonian (0.24%)
]
LANGUAGE_TO_IDX = {lang: idx for idx, lang in enumerate(LANGUAGE_VOCAB)}
NUM_LANGUAGES = len(LANGUAGE_VOCAB)


def encode_language(authentic_language: list[str] | str | None) -> np.ndarray:
    """Encode authentic language to multihot vector.

    Args:
        authentic_language: List of language codes, single code, or None

    Returns:
        Multihot vector of shape (NUM_LANGUAGES,) where UNKNOWN is all zeros
    """
    multihot = np.zeros(NUM_LANGUAGES, dtype=np.float32)

    if authentic_language is None:
        return multihot  # UNKNOWN: all zeros

    # Handle string input (single language)
    if isinstance(authentic_language, str):
        idx = LANGUAGE_TO_IDX.get(authentic_language)
        if idx is not None and idx != LANGUAGE_TO_IDX["UNKNOWN"]:
            multihot[idx] = 1.0
        return multihot

    # Handle list input
    if not authentic_language:  # Empty list
        return multihot  # UNKNOWN: all zeros

    # For multiple languages, set all of them to 1
    for lang in authentic_language:
        idx = LANGUAGE_TO_IDX.get(lang)
        if idx is not None and idx != LANGUAGE_TO_IDX["UNKNOWN"]:
            multihot[idx] = 1.0

    return multihot


class BaseGraphBuilder(ABC):
    """Base class for graph builders."""

    def __init__(self, preprocessed_dir: str):
        """
        Initialize graph builder with preprocessed data.

        Args:
            preprocessed_dir: Directory containing preprocessed embeddings and metadata
        """
        self.preprocessed_dir = Path(preprocessed_dir)

        # Load paragraph data (both document and query embeddings)
        doc_emb_path = self.preprocessed_dir / "paragraph_embeddings_doc.npy"
        query_emb_path = self.preprocessed_dir / "paragraph_embeddings_query.npy"
        legacy_emb_path = self.preprocessed_dir / "paragraph_embeddings.npy"

        if doc_emb_path.exists() and query_emb_path.exists():
            self.par_embeddings_doc = np.load(doc_emb_path)
            self.par_embeddings_query = np.load(query_emb_path)
            print("Loaded separate document and query embeddings")
        elif legacy_emb_path.exists():
            # Fallback to legacy format (use same embeddings for both)
            self.par_embeddings_doc = np.load(legacy_emb_path)
            self.par_embeddings_query = self.par_embeddings_doc
            print("Using legacy embeddings (same for doc and query)")
        else:
            raise FileNotFoundError(
                f"Could not find paragraph embeddings in {self.preprocessed_dir}"
            )

        with open(self.preprocessed_dir / "paragraph_metadata.pkl", "rb") as f:
            self.par_metadata = pickle.load(f)

        # Load article data
        self.art_embeddings = np.load(self.preprocessed_dir / "article_embeddings.npy")
        with open(self.preprocessed_dir / "article_metadata.pkl", "rb") as f:
            self.art_metadata = pickle.load(f)

        # Load citations
        with open(self.preprocessed_dir / "citations.pkl", "rb") as f:
            self.citations = pickle.load(f)

        # Load case-level metadata embeddings
        self._load_case_metadata_embeddings()

        # Create ID mappings
        self.par_id_to_idx = {m["id"]: i for i, m in enumerate(self.par_metadata)}
        self.art_id_to_idx = {m["id"]: i for i, m in enumerate(self.art_metadata)}

        print(
            f"Loaded {len(self.par_metadata)} paragraphs, {len(self.art_metadata)} articles"
        )
        print(f"Loaded {len(self.citations)} citation edges")

    def _load_case_metadata_embeddings(self):
        """Load case-level metadata embeddings and create CELEX mapping."""
        # Check if case metadata embeddings exist
        subject_matter_path = (
            self.preprocessed_dir / "case_embeddings_subject_matter.npy"
        )
        keywords_path = self.preprocessed_dir / "case_embeddings_keywords.npy"
        case_law_about_path = (
            self.preprocessed_dir / "case_embeddings_case_law_about.npy"
        )
        case_metadata_path = self.preprocessed_dir / "case_metadata.pkl"

        if all(
            p.exists()
            for p in [
                subject_matter_path,
                keywords_path,
                case_law_about_path,
                case_metadata_path,
            ]
        ):
            self.case_embeddings_subject_matter = np.load(subject_matter_path)
            self.case_embeddings_keywords = np.load(keywords_path)
            self.case_embeddings_case_law_about = np.load(case_law_about_path)

            with open(case_metadata_path, "rb") as f:
                case_metadata = pickle.load(f)

            # Create CELEX to case index mapping
            self.celex_to_case_idx = {
                m["celex"]: i for i, m in enumerate(case_metadata)
            }

            self.has_case_metadata = True
            print(
                f"Loaded case metadata embeddings for {len(self.celex_to_case_idx)} cases"
            )
            print(
                f"  - Subject matter embedding dim: {self.case_embeddings_subject_matter.shape[1]}"
            )
            print(
                f"  - Keywords embedding dim: {self.case_embeddings_keywords.shape[1]}"
            )
            print(
                f"  - Case law about embedding dim: {self.case_embeddings_case_law_about.shape[1]}"
            )
        else:
            self.has_case_metadata = False
            self.celex_to_case_idx = {}
            print("Case metadata embeddings not found, skipping metadata concatenation")

    def _get_case_metadata_embedding(self, celex: str) -> np.ndarray | None:
        """Get concatenated case metadata embedding for a CELEX."""
        if not self.has_case_metadata:
            return None

        if celex not in self.celex_to_case_idx:
            return None

        case_idx = self.celex_to_case_idx[celex]

        # Concatenate all three metadata embeddings
        metadata_emb = np.concatenate(
            [
                self.case_embeddings_subject_matter[case_idx],
                self.case_embeddings_keywords[case_idx],
                self.case_embeddings_case_law_about[case_idx],
            ]
        )

        return metadata_emb

    @abstractmethod
    def build_graph(self, train_cutoff_year: int | None = None):
        """Build and return graph data structure.

        Args:
            train_cutoff_year: Only include paragraphs before this year
        """
        pass

    def mask_edges_for_training(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        anchor_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Mask edges to prevent information leakage during training.

        Override in subclasses to implement graph-specific masking strategies.

        Args:
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge attributes/types (optional)
            anchor_count: Number of anchor/input nodes (first anchor_count nodes)

        Returns:
            Tuple of (masked_edge_index, masked_edge_attr)
        """
        raise NotImplementedError("Subclasses must implement mask_edges_for_training")

    def _date_to_timestamp(self, date_str: str | None) -> int:
        """Convert ISO date string to Unix timestamp (seconds since epoch)."""
        if not date_str:
            return 0
        try:
            dt = datetime.fromisoformat(date_str)
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            return 0

    def _filter_paragraphs(
        self, include_only_citing: bool, train_cutoff_year: int | None
    ) -> list[int]:
        """Filter which paragraphs to include."""
        selected_pars = []

        citing_pars = set()
        if include_only_citing:
            for src_id, tgt_id in self.citations:
                if src_id.startswith("par:") and tgt_id.startswith("par:"):
                    citing_pars.add(src_id)
                    citing_pars.add(tgt_id)

        for i, meta in enumerate(self.par_metadata):
            # Filter by citing status
            if include_only_citing and meta["id"] not in citing_pars:
                continue

            # Filter by year
            if train_cutoff_year and meta.get("year"):
                if meta["year"] >= train_cutoff_year:
                    continue

            selected_pars.append(i)

        return selected_pars

    def _normalize_date(self, date_str: str | None) -> float:
        """Normalize date to [0, 1] range based on days since 1954-01-01."""
        if not date_str:
            return 0.0
        try:
            dt = datetime.fromisoformat(date_str)
            base_date = datetime(1954, 1, 1)
            max_date = datetime(2025, 12, 31)
            days_since_base = (dt - base_date).days
            max_days = (max_date - base_date).days
            return max(0.0, min(1.0, days_since_base / max_days))
        except (ValueError, AttributeError):
            return 0.0

    def _extract_date_features(
        self, date_str: str | None, application_date_str: str | None = None
    ) -> np.ndarray:
        """Extract date features: [judgment_date, application_date, duration] normalized."""
        judgment_norm = self._normalize_date(date_str)
        application_norm = self._normalize_date(application_date_str)

        # Duration between application and judgment (case processing time)
        duration_norm = 0.0
        if date_str and application_date_str:
            try:
                judgment_dt = datetime.fromisoformat(date_str)
                application_dt = datetime.fromisoformat(application_date_str)
                duration_days = (judgment_dt - application_dt).days
                # Normalize: 0 days = 0, ~5 years (1825 days) = 1, clamp to [0, 1]
                duration_norm = max(0.0, min(1.0, duration_days / 1825))
            except (ValueError, AttributeError):
                pass

        return np.array(
            [judgment_norm, application_norm, duration_norm], dtype=np.float32
        )

    def _compute_relative_positions(self, selected_pars: list[int]) -> dict[int, float]:
        """Compute relative paragraph positions within each case.

        Positions are computed relative to ALL paragraphs in each case,
        not just the selected/filtered ones.
        """
        # First, collect ALL paragraphs for each case to get total counts
        case_to_all_pars: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for par_idx, meta in enumerate(self.par_metadata):
            celex = meta["celex"]
            par_num = meta.get("paragraph_number", 0)
            case_to_all_pars[celex].append((par_idx, par_num))

        # Build mapping from paragraph index to its position in the full case
        par_idx_to_position: dict[int, float] = {}
        for celex, all_pars in case_to_all_pars.items():
            # Sort all paragraphs by paragraph number
            sorted_pars = sorted(all_pars, key=lambda x: x[1])
            total_pars = len(sorted_pars)

            if total_pars == 1:
                # Single paragraph case
                par_idx_to_position[sorted_pars[0][0]] = 0.5
            else:
                # Normalize position to [0, 1] based on position in full case
                for i, (par_idx, _) in enumerate(sorted_pars):
                    par_idx_to_position[par_idx] = i / (total_pars - 1)

        # Return positions only for selected paragraphs
        relative_positions: dict[int, float] = {}
        for par_idx in selected_pars:
            relative_positions[par_idx] = par_idx_to_position.get(par_idx, 0.5)

        return relative_positions


def parse_celex(celex):
    """Parse CELEX into components (CJ only)"""
    match = re.match(r"(\d)(\d{4})CJ(\d+)", celex)
    if match:
        sector, year, number = match.groups()
        return {"sector": int(sector), "year": int(year), "number": int(number)}
    raise ValueError(f"Invalid CELEX format (expected CJ type): {celex}")


def encode_celex(celex, paragraph):
    """Encode CELEX + paragraph to tensor (CJ only)"""
    parsed = parse_celex(celex)

    tensor = torch.tensor(
        [parsed["sector"], parsed["year"], parsed["number"], paragraph]
    )

    return tensor


def decode_celex(tensor):
    """Decode tensor back to (CELEX, paragraph)"""
    sector, year, number, paragraph = tensor.tolist()
    celex = f"{sector}{year:04d}CJ{number:04d}"

    return celex, paragraph


class HomogeneousGraphBuilder(BaseGraphBuilder):
    """
    Homogeneous graph builder with citation edges only.

    Edge types:
    - 0: "cites" (forward direction: src cites tgt)
    - 1: "cited_by" (reverse direction)

    Returns PyTorch Geometric Data object.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        include_only_citing: bool = True,
        include_self_loops: bool = False,
        add_reverse_edges: bool = True,
    ):
        """
        Initialize homogeneous graph builder.

        Args:
            preprocessed_dir: Directory containing preprocessed embeddings and metadata
            include_only_citing: Only include paragraphs involved in citations
            include_self_loops: Whether to add self loops to all nodes
            add_reverse_edges: Whether to add reverse edges with different edge type
        """
        super().__init__(preprocessed_dir)
        self.include_only_citing = include_only_citing
        self.include_self_loops = include_self_loops
        self.add_reverse_edges = add_reverse_edges

    def mask_edges_for_training(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        anchor_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Mask edges to prevent leakage during training.

        Masking strategy:
        - Mask outgoing edges from anchor nodes (prevent seeing their predictions)
        - Mask incoming citation edges to anchors (prevent seeing what cites them)

        This ensures anchors don't see the citation relationships we're training on.
        """
        src, tgt = edge_index
        outgoing_from_anchor = src < anchor_count
        incoming_to_anchor = tgt < anchor_count

        if edge_attr is not None:
            # Only mask incoming edges that are citations (not reverse edges)
            is_citation = (edge_attr == 0) | (edge_attr == 1)
            leakage_mask = outgoing_from_anchor | (incoming_to_anchor & is_citation)
        else:
            # Without edge types, mask all incoming/outgoing
            leakage_mask = outgoing_from_anchor | incoming_to_anchor

        keep_mask = ~leakage_mask
        masked_edge_index = edge_index[:, keep_mask]
        masked_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None

        return masked_edge_index, masked_edge_attr

    def build_graph(self, train_cutoff_year: int | None = None) -> Data:
        """Build homogeneous citation graph.

        Args:
            train_cutoff_year: Only include paragraphs before this year
        """
        selected_pars = self._filter_paragraphs(
            self.include_only_citing, train_cutoff_year
        )
        print(f"Selected {len(selected_pars)} paragraphs")

        # Build node mappings
        node_id_to_idx: dict[str, int] = {}
        doc_embeddings_list = []
        query_embeddings_list = []
        date_features_list = []
        language_multihot_list = []
        node_times = []
        node_ids = []

        # Case metadata lists
        subject_matter_list = []
        keywords_list = []
        case_law_about_list = []

        for par_idx in selected_pars:
            meta = self.par_metadata[par_idx]
            node_id = meta["id"]
            current_idx = len(node_id_to_idx)

            node_id_to_idx[node_id] = current_idx
            doc_embeddings_list.append(self.par_embeddings_doc[par_idx])
            query_embeddings_list.append(self.par_embeddings_query[par_idx])

            case_meta = meta.get("meta", {})
            date_features_list.append(
                self._extract_date_features(
                    meta.get("date"), case_meta.get("application_date")
                )
            )
            node_ids.append(encode_celex(meta["celex"], meta["paragraph_number"]))
            language_multihot_list.append(
                encode_language(case_meta.get("authentic_language"))
            )
            node_times.append(self._date_to_timestamp(meta.get("date")))

            if self.has_case_metadata:
                celex = meta["celex"]
                if celex in self.celex_to_case_idx:
                    case_idx = self.celex_to_case_idx[celex]
                    subject_matter_list.append(
                        self.case_embeddings_subject_matter[case_idx]
                    )
                    keywords_list.append(self.case_embeddings_keywords[case_idx])
                    case_law_about_list.append(
                        self.case_embeddings_case_law_about[case_idx]
                    )
                else:
                    subject_matter_list.append(
                        np.zeros_like(self.case_embeddings_subject_matter[0])
                    )
                    keywords_list.append(
                        np.zeros_like(self.case_embeddings_keywords[0])
                    )
                    case_law_about_list.append(
                        np.zeros_like(self.case_embeddings_case_law_about[0])
                    )

        # Build citation edges
        edge_list = []
        edge_attr_list = []

        for src_id, tgt_id in self.citations:
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                tgt_idx = node_id_to_idx[tgt_id]

                edge_list.append([src_idx, tgt_idx])
                edge_attr_list.append(0)  # cites

                if self.add_reverse_edges:
                    edge_list.append([tgt_idx, src_idx])
                    edge_attr_list.append(1)  # cited_by

        # Create tensors
        x_doc = torch.tensor(np.array(doc_embeddings_list), dtype=torch.float32)
        x_query = torch.tensor(np.array(query_embeddings_list), dtype=torch.float32)
        date_features = torch.tensor(np.array(date_features_list), dtype=torch.float32)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        if self.include_self_loops:
            num_nodes = len(doc_embeddings_list)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=num_nodes
            )

        node_times_tensor = torch.tensor(node_times, dtype=torch.long)
        node_ids_tensor = torch.stack(node_ids)
        language_tensor = torch.tensor(
            np.array(language_multihot_list), dtype=torch.float32
        )

        # Case metadata tensors
        case_metadata_kwargs = {}
        if self.has_case_metadata and subject_matter_list:
            case_metadata_kwargs["subject_matter"] = torch.tensor(
                np.array(subject_matter_list), dtype=torch.float32
            )
            case_metadata_kwargs["keywords"] = torch.tensor(
                np.array(keywords_list), dtype=torch.float32
            )
            case_metadata_kwargs["case_law_about"] = torch.tensor(
                np.array(case_law_about_list), dtype=torch.float32
            )

        graph_data = Data(
            x=x_doc,
            x_query=x_query,
            date_feature=date_features,
            language=language_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(doc_embeddings_list),
            time=node_times_tensor,
            node_id_hash=node_ids_tensor,
            **case_metadata_kwargs,
        )

        cites_count = (edge_attr == 0).sum().item()
        cited_by_count = (edge_attr == 1).sum().item()
        print(
            f"Built homogeneous graph: {len(doc_embeddings_list)} nodes, "
            f"{edge_index.shape[1]} edges"
        )
        print(f"  Edge types: 0=cites ({cites_count}), 1=cited_by ({cited_by_count})")

        return graph_data


class SemanticGraphBuilder(BaseGraphBuilder):
    """
    Semantic graph builder with TF-IDF similarity edges (CaseLink-style).

    Uses TF-IDF semantic similarity for edges instead of citations.
    Optionally includes article nodes with embedding-based similarity edges.

    Edge types:
    - 0: "similar_to" - TF-IDF semantic similarity between paragraphs
    - 1: "references_article" - paragraph references article (if article nodes enabled)
    - 2: "referenced_by_article" - reverse of above
    - 3: "article_similar" - article to article similarity

    Citations are stored separately in `citation_pairs` for contrastive loss computation.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        judgments_path: str,
        include_only_citing: bool = True,
        semantic_max_neighbors: int = 10,
        use_temporal_constraint: bool = True,
        include_article_nodes: bool = True,
        article_threshold: float = 0.9,
        semantic_cache_path: str | None = None,
    ):
        """
        Initialize semantic graph builder.

        Args:
            preprocessed_dir: Directory containing preprocessed embeddings and metadata
            judgments_path: Path to judgments JSON file (required for TF-IDF)
            include_only_citing: Only include paragraphs involved in citations
            semantic_max_neighbors: Max neighbors per node for semantic edges
            use_temporal_constraint: Only link to earlier paragraphs (always True for TF-IDF)
            include_article_nodes: Include article nodes in the graph
            article_threshold: Cosine similarity threshold for article edges
            semantic_cache_path: Path to cache semantic edges (skips recomputation if exists)
        """
        super().__init__(preprocessed_dir)
        self.judgments_path = judgments_path
        self.include_only_citing = include_only_citing
        self.semantic_max_neighbors = semantic_max_neighbors
        self.use_temporal_constraint = use_temporal_constraint
        self.include_article_nodes = include_article_nodes
        self.article_threshold = article_threshold
        self.semantic_cache_path = semantic_cache_path

        # TF-IDF state (lazy initialization)
        self.tfidf_vectorizer: TfidfVectorizer | None = None
        self.par_texts: dict[str, str] | None = None

    def mask_edges_for_training(
        self,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None,
        anchor_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Mask edges to prevent leakage during training.

        Masking strategy (CaseLink-style):
        - Only mask outgoing edges from anchor nodes
        - Semantic similarity edges are kept for message passing
        - Citations are NOT in the graph (stored in citation_pairs), so no need to mask them

        This allows the GNN to use semantic neighbors for embeddings while preventing
        the query from influencing its own neighborhood.
        """
        src, _ = edge_index
        outgoing_from_anchor = src < anchor_count

        keep_mask = ~outgoing_from_anchor
        masked_edge_index = edge_index[:, keep_mask]
        masked_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None

        return masked_edge_index, masked_edge_attr

    # =========================================================================
    # Semantic Edge Computation (TF-IDF and FAISS-based)
    # =========================================================================

    def _load_paragraph_texts(self) -> dict[str, str]:
        """Load paragraph texts from judgments file for TF-IDF computation."""
        print("Loading paragraph texts for TF-IDF...")
        with open(self.judgments_path) as f:
            judgments = json.load(f)

        par_texts = {}
        for celex, judgment in judgments.items():
            for par_num, text in judgment.get("paragraphs", {}).items():
                par_id = f"par:{celex}:{par_num}"
                par_texts[par_id] = text

        print(f"Loaded texts for {len(par_texts)} paragraphs")
        return par_texts

    def _get_cache_key(self, par_ids: list[str], use_temporal: bool) -> str:
        """Generate a cache key based on parameters."""
        import hashlib

        # Include all parameters that affect the edges
        # Note: semantic_threshold removed - paragraphs no longer use threshold
        params = f"{self.semantic_max_neighbors}_{use_temporal}"
        par_ids_hash = hashlib.md5("_".join(sorted(par_ids)).encode()).hexdigest()[:16]
        return f"semantic_edges_{params}_{par_ids_hash}"

    def _load_cached_edges(self, cache_key: str) -> list[tuple[int, int]] | None:
        """Load cached semantic edges if available."""
        if self.semantic_cache_path is None:
            return None

        cache_dir = Path(self.semantic_cache_path)
        cache_file = cache_dir / f"{cache_key}.pkl"

        if cache_file.exists():
            print(f"  Loading cached semantic edges from {cache_file}")
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cached_edges(self, cache_key: str, edges: list[tuple[int, int]]) -> None:
        """Save computed semantic edges to cache."""
        if self.semantic_cache_path is None:
            return

        cache_dir = Path(self.semantic_cache_path)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{cache_key}.pkl"

        print(f"  Saving semantic edges to cache: {cache_file}")
        with open(cache_file, "wb") as f:
            pickle.dump(edges, f)

    def _compute_tfidf_semantic_edges(
        self,
        par_ids: list[str],
        times: np.ndarray,
        batch_size: int = 512,
    ) -> list[tuple[int, int]]:
        """Compute semantic similarity edges using TF-IDF with temporal constraints."""
        # Always use temporal for TF-IDF semantic edges
        use_temporal = True
        cache_key = self._get_cache_key(par_ids, use_temporal)

        # Try to load from cache
        cached_edges = self._load_cached_edges(cache_key)
        if cached_edges is not None:
            print(f"  Loaded {len(cached_edges)} cached TF-IDF semantic edges")
            return cached_edges

        if self.par_texts is None:
            self.par_texts = self._load_paragraph_texts()

        n = len(par_ids)
        print(f"  Computing TF-IDF semantic edges for {n} paragraphs...")

        # Get texts for the selected paragraphs
        texts = [self.par_texts.get(par_id, "") for par_id in par_ids]

        # Fit TF-IDF vectorizer if not already fitted
        if self.tfidf_vectorizer is None:
            print("  Fitting TF-IDF vectorizer...")
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words="english",
                strip_accents="ascii",
                norm="l2",
                max_features=50000,
            )
            self.tfidf_vectorizer.fit(texts)

        # Transform texts to TF-IDF vectors
        print("  Transforming texts to TF-IDF vectors...")
        tfidf_matrix = self.tfidf_vectorizer.transform(texts)

        # Always use temporal constraints for TF-IDF semantic edges
        edges = self._compute_tfidf_edges_temporal(tfidf_matrix, times, batch_size)

        print(f"  Found {len(edges)} TF-IDF semantic similarity edges")

        # Save to cache
        self._save_cached_edges(cache_key, edges)

        return edges

    def _compute_tfidf_edges_no_temporal(
        self,
        tfidf_matrix: csr_matrix,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        """Compute TF-IDF edges without temporal constraints using sparse_dot_topn."""
        n = tfidf_matrix.shape[0]

        # Use sparse_dot_topn to find top-k similar documents efficiently
        # Request extra neighbors to account for self-similarity filtering
        top_k = min(self.semantic_max_neighbors + 1, n)
        topn_matrix = sp_matmul_topn(
            tfidf_matrix,
            tfidf_matrix.T,
            top_n=top_k,
            threshold=0.0,  # No threshold filtering
            sort=True,
        )

        # Extract edges from sparse result, filtering out self-loops
        edges = []
        coo = topn_matrix.tocoo()
        for row_idx, col_idx in zip(coo.row, coo.col):
            if row_idx != col_idx:  # Skip self-similarity
                edges.append((row_idx, col_idx))

        return edges

    def _compute_tfidf_edges_temporal(
        self,
        tfidf_matrix: csr_matrix,
        times: np.ndarray,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        """Compute TF-IDF edges with temporal constraints using sparse_dot_topn."""
        n = tfidf_matrix.shape[0]

        # Group indices by time
        time_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(n):
            time_to_indices[times[idx]].append(idx)

        unique_times = sorted(time_to_indices.keys())
        print(f"  Processing {len(unique_times)} time groups chronologically...")

        edges = []
        accumulated_indices: list[int] = []
        nodes_processed = 0

        for time_idx, t in tqdm(
            enumerate(unique_times),
            total=len(unique_times),
            desc="Processing time groups chronologically",
        ):
            group_indices = time_to_indices[t]
            group_size = len(group_indices)

            if accumulated_indices and group_size > 0:
                # Get TF-IDF vectors for current group and accumulated (earlier) nodes
                group_tfidf = tfidf_matrix[group_indices]
                accumulated_tfidf = tfidf_matrix[accumulated_indices]

                # Use sparse_dot_topn to find top-k similar documents efficiently
                # This keeps everything sparse and avoids dense matrix conversion
                top_k = min(self.semantic_max_neighbors, len(accumulated_indices))
                topn_matrix = sp_matmul_topn(
                    group_tfidf,
                    accumulated_tfidf.T,
                    top_n=top_k,
                    threshold=0.0,  # No threshold filtering
                    sort=True,
                )

                # Extract edges from sparse result matrix
                coo = topn_matrix.tocoo()
                for row_idx, col_idx in zip(coo.row, coo.col):
                    orig_idx = group_indices[row_idx]
                    neighbor_orig_idx = accumulated_indices[col_idx]
                    edges.append((orig_idx, neighbor_orig_idx))

            # Add current group to accumulated indices
            accumulated_indices.extend(group_indices)
            nodes_processed += group_size

            if (time_idx + 1) % max(1, len(unique_times) // 10) == 0:
                pct = 100 * (time_idx + 1) / len(unique_times)
        print(
            f"    {pct:.0f}% complete ({nodes_processed}/{n} nodes, {len(edges)} edges)"
        )

        return edges

    def _compute_faiss_semantic_edges(
        self,
        embeddings: np.ndarray,
        times: np.ndarray | None = None,
        batch_size: int = 1024,
    ) -> list[tuple[int, int]]:
        """Compute semantic similarity edges using FAISS with temporal constraints."""
        n, d = embeddings.shape
        print(f"  Computing semantic edges for {n} nodes using FAISS...")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        embeddings_normalized = (embeddings / norms).astype(np.float32)

        # Ensure contiguous array for FAISS
        if not embeddings_normalized.flags["C_CONTIGUOUS"]:
            embeddings_normalized = np.ascontiguousarray(embeddings_normalized)

        if times is None:
            edges = self._compute_faiss_edges_no_temporal(
                embeddings_normalized, batch_size
            )
        else:
            edges = self._compute_faiss_edges_temporal(
                embeddings_normalized, times, batch_size
            )

        print(f"  Found {len(edges)} semantic similarity edges")
        return edges

    def _compute_faiss_edges_no_temporal(
        self,
        embeddings: np.ndarray,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        """Compute edges without temporal constraints using FAISS."""
        n, d = embeddings.shape

        index = faiss.IndexFlatIP(d)
        if not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings)
        index.add(embeddings)

        edges = []

        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_embs = embeddings[start:end]

            k = min(self.semantic_max_neighbors + 1, n)
            similarities, neighbors = index.search(batch_embs, k)

            for i, orig_idx in enumerate(range(start, end)):
                for j in range(k):
                    neighbor_idx = neighbors[i, j]
                    sim = similarities[i, j]

                    if neighbor_idx == orig_idx:
                        continue

                    if sim >= self.article_threshold:
                        edges.append((orig_idx, neighbor_idx))

        return edges

    def _compute_faiss_edges_temporal(
        self,
        embeddings: np.ndarray,
        times: np.ndarray,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        """Compute edges with temporal constraints using FAISS."""
        n, d = embeddings.shape

        # Group indices by time
        time_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(n):
            time_to_indices[times[idx]].append(idx)

        unique_times = sorted(time_to_indices.keys())
        print(f"  Processing {len(unique_times)} time groups chronologically 2...")

        index = faiss.IndexFlatIP(d)
        faiss_to_orig: list[int] = []

        edges = []
        nodes_processed = 0

        for time_idx, t in tqdm(
            enumerate(unique_times),
            total=len(unique_times),
            desc="Processing time groups chronologically",
        ):
            group_indices = time_to_indices[t]
            group_size = len(group_indices)

            if index.ntotal > 0 and group_size > 0:
                group_embs = embeddings[group_indices]

                if not group_embs.flags["C_CONTIGUOUS"]:
                    group_embs = np.ascontiguousarray(group_embs)

                for batch_start in range(0, group_size, batch_size):
                    batch_end = min(batch_start + batch_size, group_size)
                    batch_indices = group_indices[batch_start:batch_end]
                    batch_embs = group_embs[batch_start:batch_end]

                    if not batch_embs.flags["C_CONTIGUOUS"]:
                        batch_embs = np.ascontiguousarray(batch_embs)

                    k = min(self.semantic_max_neighbors, index.ntotal)
                    similarities, faiss_neighbors = index.search(batch_embs, k)

                    for i, orig_idx in enumerate(batch_indices):
                        for j in range(k):
                            sim = similarities[i, j]
                            if sim >= self.article_threshold:
                                faiss_idx = faiss_neighbors[i, j]
                                neighbor_orig_idx = faiss_to_orig[faiss_idx]
                                edges.append((orig_idx, neighbor_orig_idx))

            # Add current group to the index for future searches
            if group_size > 0:
                group_embs = embeddings[group_indices]

                if not group_embs.flags["C_CONTIGUOUS"]:
                    group_embs = np.ascontiguousarray(group_embs)

                index.add(group_embs)
                faiss_to_orig.extend(group_indices)

            nodes_processed += group_size

            if (time_idx + 1) % max(1, len(unique_times) // 10) == 0:
                pct = 100 * (time_idx + 1) / len(unique_times)
        print(
            f"    {pct:.0f}% complete ({nodes_processed}/{n} nodes, {len(edges)} edges)"
        )

        return edges

    def build_graph(self, train_cutoff_year: int | None = None) -> Data:
        """
        Build semantic similarity graph.

        Args:
            train_cutoff_year: Only include paragraphs before this year

        Returns:
            graph_data: PyTorch Geometric Data object with:
                - citation_pairs: [2, N] tensor of citation pairs (for training loss)
                - num_par_nodes: Number of paragraph nodes
                - num_art_nodes: Number of article nodes
        """
        selected_pars = self._filter_paragraphs(
            self.include_only_citing, train_cutoff_year
        )
        print(f"Selected {len(selected_pars)} paragraphs")

        # Build paragraph node mappings
        par_node_id_to_idx: dict[str, int] = {}
        doc_embeddings_list = []
        query_embeddings_list = []
        date_features_list = []
        par_times = []
        node_ids = []

        for par_idx in selected_pars:
            meta = self.par_metadata[par_idx]
            node_id = meta["id"]
            current_idx = len(par_node_id_to_idx)

            par_node_id_to_idx[node_id] = current_idx
            doc_embeddings_list.append(self.par_embeddings_doc[par_idx])
            query_embeddings_list.append(self.par_embeddings_query[par_idx])

            case_meta = meta.get("meta", {})
            date_features_list.append(
                self._extract_date_features(
                    meta.get("date"), case_meta.get("application_date")
                )
            )
            node_ids.append(encode_celex(meta["celex"], meta["paragraph_number"]))
            par_times.append(self._date_to_timestamp(meta.get("date")))

        num_par_nodes = len(par_node_id_to_idx)

        # Build article nodes if enabled
        art_node_id_to_idx: dict[str, int] = {}
        art_embeddings_list = []

        if self.include_article_nodes:
            referenced_articles = set()
            for src_id, tgt_id in self.citations:
                if src_id in par_node_id_to_idx and tgt_id in self.art_id_to_idx:
                    referenced_articles.add(tgt_id)

            for art_id in referenced_articles:
                art_orig_idx = self.art_id_to_idx[art_id]
                current_idx = len(art_node_id_to_idx)
                art_node_id_to_idx[art_id] = current_idx
                art_embeddings_list.append(self.art_embeddings[art_orig_idx])

            print(f"Selected {len(art_node_id_to_idx)} articles")

        num_art_nodes = len(art_node_id_to_idx)
        total_nodes = num_par_nodes + num_art_nodes

        # Build edges
        edge_list = []
        edge_attr_list = []

        # Collect citation pairs for loss computation (not as graph edges)
        citation_src_list = []
        citation_tgt_list = []
        for src_id, tgt_id in self.citations:
            if src_id in par_node_id_to_idx and tgt_id in par_node_id_to_idx:
                citation_src_list.append(par_node_id_to_idx[src_id])
                citation_tgt_list.append(par_node_id_to_idx[tgt_id])

        print(f"  Citation pairs for training: {len(citation_src_list)}")

        # TF-IDF semantic similarity edges
        print("Computing TF-IDF semantic similarity edges...")
        par_ids_in_order = [
            self.par_metadata[selected_pars[idx]]["id"] for idx in range(num_par_nodes)
        ]
        par_times_array = np.array(par_times)

        # Always use temporal constraints for TF-IDF semantic edges
        semantic_edges = self._compute_tfidf_semantic_edges(
            par_ids_in_order,
            times=par_times_array,
        )
        for src_idx, tgt_idx in semantic_edges:
            edge_list.append([src_idx, tgt_idx])
            edge_attr_list.append(0)  # similar_to
            edge_list.append([tgt_idx, src_idx])
            edge_attr_list.append(0)

        print(f"  Semantic edges: {len(semantic_edges)} (bidirectional)")

        # Article edges if enabled
        if self.include_article_nodes and len(art_node_id_to_idx) > 0:
            # Paragraph -> Article edges
            par_art_edges = 0
            for src_id, tgt_id in self.citations:
                if src_id in par_node_id_to_idx and tgt_id in art_node_id_to_idx:
                    par_idx = par_node_id_to_idx[src_id]
                    art_idx = num_par_nodes + art_node_id_to_idx[tgt_id]
                    edge_list.append([par_idx, art_idx])
                    edge_attr_list.append(1)  # references_article
                    edge_list.append([art_idx, par_idx])
                    edge_attr_list.append(2)  # referenced_by_article
                    par_art_edges += 1

            print(f"  Paragraph-article edges: {par_art_edges} (bidirectional)")

            # Article-article similarity edges
            if len(art_node_id_to_idx) > 1:
                print("Computing article similarity edges...")
                art_emb_array = np.array(art_embeddings_list)

                article_edges = self._compute_faiss_semantic_edges(
                    art_emb_array,
                    times=None,
                )
                for src_idx, tgt_idx in article_edges:
                    global_src = num_par_nodes + src_idx
                    global_tgt = num_par_nodes + tgt_idx
                    edge_list.append([global_src, global_tgt])
                    edge_attr_list.append(3)  # article_similar

                print(f"  Article-article edges: {len(article_edges)}")

        # Create embeddings tensors
        all_doc_embeddings = doc_embeddings_list.copy()
        all_query_embeddings = query_embeddings_list.copy()

        if self.include_article_nodes and len(art_embeddings_list) > 0:
            par_emb_dim = self.par_embeddings_doc.shape[1]
            art_emb_dim = self.art_embeddings.shape[1]
            art_emb_array = np.array(art_embeddings_list)

            if art_emb_dim != par_emb_dim:
                if art_emb_dim < par_emb_dim:
                    padding = np.zeros(
                        (len(art_embeddings_list), par_emb_dim - art_emb_dim)
                    )
                    art_emb_array = np.concatenate([art_emb_array, padding], axis=1)
                else:
                    art_emb_array = art_emb_array[:, :par_emb_dim]

            for emb in art_emb_array:
                all_doc_embeddings.append(emb)
                all_query_embeddings.append(emb)

        # Create tensors
        x_doc = torch.tensor(np.array(all_doc_embeddings), dtype=torch.float32)
        x_query = torch.tensor(np.array(all_query_embeddings), dtype=torch.float32)
        date_features = torch.tensor(np.array(date_features_list), dtype=torch.float32)

        # Extend date features for articles (zeros)
        if self.include_article_nodes and num_art_nodes > 0:
            art_date_features = np.zeros(
                (num_art_nodes, date_features.shape[1]), dtype=np.float32
            )
            date_features = torch.cat(
                [date_features, torch.tensor(art_date_features)], dim=0
            )

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        # Time tensor (articles get time 0)
        all_times = par_times.copy()
        if self.include_article_nodes and num_art_nodes > 0:
            all_times.extend([0] * num_art_nodes)
        node_times_tensor = torch.tensor(all_times, dtype=torch.long)

        node_ids_tensor = torch.stack(node_ids)

        # Node type: 0 = paragraph, 1 = article
        node_type = torch.zeros(total_nodes, dtype=torch.long)
        if self.include_article_nodes and num_art_nodes > 0:
            node_type[num_par_nodes:] = 1

        # Citation pairs for training
        if citation_src_list:
            citation_pairs = torch.tensor(
                [citation_src_list, citation_tgt_list], dtype=torch.long
            )
        else:
            citation_pairs = torch.empty((2, 0), dtype=torch.long)

        graph_data = Data(
            x=x_doc,
            x_query=x_query,
            date_feature=date_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=total_nodes,
            time=node_times_tensor,
            node_id_hash=node_ids_tensor,
            node_type=node_type,
            num_par_nodes=num_par_nodes,
            num_art_nodes=num_art_nodes,
            citation_pairs=citation_pairs,
        )

        print(f"\nBuilt semantic graph:")
        print(
            f"  Nodes: {total_nodes} ({num_par_nodes} paragraphs, {num_art_nodes} articles)"
        )
        print(f"  Edges: {edge_index.shape[1]}")
        print(f"  Citation pairs: {citation_pairs.shape[1]}")

        return graph_data


class HeterogeneousGraphBuilder(BaseGraphBuilder):
    """
    Heterogeneous graph builder with case/act nodes and rich connectivity.

    Node types: paragraph, article, case, legal_act
    Edge types:
    - (paragraph, cites, paragraph): Citation edges
    - (paragraph, next, paragraph): Sequential edges to next paragraph
    - (paragraph, belongs_to, case): Paragraph to its case
    - (article, belongs_to, legal_act): Article to its legal act

    Returns PyTorch Geometric HeteroData object.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        include_only_citing: bool = False,
        include_articles: bool = True,
        include_citations: bool = True,
    ):
        """
        Initialize heterogeneous graph builder.

        Args:
            preprocessed_dir: Directory containing preprocessed embeddings and metadata
            include_only_citing: Only include paragraphs involved in citations
            include_articles: Include article and legal_act nodes in the graph
            include_citations: Include citation edges between paragraphs
        """
        super().__init__(preprocessed_dir)
        self.include_only_citing = include_only_citing
        self.include_articles = include_articles
        self.include_citations = include_citations

    def build_graph(self, train_cutoff_year: int | None = None) -> HeteroData:
        """Build heterogeneous graph with case and act nodes.

        Args:
            train_cutoff_year: Only include paragraphs before this year
        """
        data = HeteroData()

        # Filter paragraphs
        selected_pars = self._filter_paragraphs(
            self.include_only_citing, train_cutoff_year
        )

        # Build paragraph nodes
        par_node_id_to_idx: dict[str, int] = {}
        par_idx_to_metadata: dict[int, dict] = {}
        par_doc_embeddings_list = []
        par_query_embeddings_list = []
        par_times = []
        par_node_ids = []
        case_to_par_indices: dict[str, list[int]] = defaultdict(list)
        # Track case-level info for case nodes
        case_info: dict[str, dict] = {}

        for par_idx in selected_pars:
            meta = self.par_metadata[par_idx]
            node_id = meta["id"]
            current_idx = len(par_node_id_to_idx)

            par_node_id_to_idx[node_id] = current_idx
            par_idx_to_metadata[current_idx] = meta
            par_doc_embeddings_list.append(self.par_embeddings_doc[par_idx])
            par_query_embeddings_list.append(self.par_embeddings_query[par_idx])
            par_node_ids.append(encode_celex(meta["celex"], meta["paragraph_number"]))

            # Convert date to Unix timestamp
            date_str = meta.get("date")
            par_times.append(self._date_to_timestamp(date_str))

            # Track which paragraphs belong to each case
            celex = meta["celex"]
            case_to_par_indices[celex].append(current_idx)

            # Store case-level info (only once per case)
            if celex not in case_info:
                case_meta = meta.get("meta", {})
                case_info[celex] = {
                    "date": meta.get("date"),
                    "application_date": case_meta.get("application_date"),
                    "authentic_language": case_meta.get("authentic_language"),
                }

        # Build article nodes (all articles) - only if include_articles is True
        art_node_id_to_idx: dict[str, int] = {}
        art_idx_to_metadata: dict[int, dict] = {}
        art_embeddings_list: list[np.ndarray] = []
        art_times: list[int] = []
        act_to_art_indices: dict[str, list[int]] = defaultdict(list)

        if self.include_articles:
            for art_idx, meta in enumerate(self.art_metadata):
                node_id = meta["id"]
                current_idx = len(art_node_id_to_idx)

                art_node_id_to_idx[node_id] = current_idx
                art_idx_to_metadata[current_idx] = meta
                art_embeddings_list.append(self.art_embeddings[art_idx])

                # Articles don't have dates, use 0
                art_times.append(0)

                # Track which articles belong to each legal act
                celex = meta["celex"]
                act_to_art_indices[celex].append(current_idx)

        # Build case nodes with case-level metadata
        case_node_id_to_idx: dict[str, int] = {}
        case_idx_to_metadata: dict[int, dict] = {}
        case_features_list = []
        case_times = []

        # Determine case feature dimension
        if self.has_case_metadata:
            # Features: date(3) + language(NUM_LANGUAGES) + subject_matter + keywords + case_law_about
            case_emb_dim = (
                3
                + NUM_LANGUAGES
                + self.case_embeddings_subject_matter.shape[1]
                + self.case_embeddings_keywords.shape[1]
                + self.case_embeddings_case_law_about.shape[1]
            )
        else:
            # Fallback: date(3) + language(NUM_LANGUAGES)
            case_emb_dim = 3 + NUM_LANGUAGES

        for celex, par_indices in case_to_par_indices.items():
            current_idx = len(case_node_id_to_idx)
            case_node_id_to_idx[f"case:{celex}"] = current_idx

            # Get case info
            info = case_info.get(celex, {})

            # Date features (3 dims)
            date_features = self._extract_date_features(
                info.get("date"), info.get("application_date")
            )

            # Language multihot (NUM_LANGUAGES dims)
            language_features = encode_language(info.get("authentic_language"))

            # Build case feature vector
            if self.has_case_metadata and celex in self.celex_to_case_idx:
                case_idx = self.celex_to_case_idx[celex]
                case_features = np.concatenate(
                    [
                        date_features,
                        language_features,
                        self.case_embeddings_subject_matter[case_idx],
                        self.case_embeddings_keywords[case_idx],
                        self.case_embeddings_case_law_about[case_idx],
                    ]
                )
            else:
                # Fallback: just date and language features, padded with zeros
                case_features = np.zeros(case_emb_dim, dtype=np.float32)
                case_features[: 3 + NUM_LANGUAGES] = np.concatenate(
                    [date_features, language_features]
                )

            case_features_list.append(case_features)

            # Use earliest paragraph's date timestamp
            case_timestamp = min(par_times[i] for i in par_indices)
            case_times.append(case_timestamp)

            case_idx_to_metadata[current_idx] = {
                "id": f"case:{celex}",
                "type": "case",
                "celex": celex,
                "num_paragraphs": len(par_indices),
                "timestamp": case_timestamp,
            }

        # Build legal act nodes (average of articles) - only if include_articles is True
        act_node_id_to_idx: dict[str, int] = {}
        act_idx_to_metadata: dict[int, dict] = {}
        act_embeddings_list: list[np.ndarray] = []
        act_times: list[int] = []

        if self.include_articles:
            for celex, art_indices in act_to_art_indices.items():
                current_idx = len(act_node_id_to_idx)
                act_node_id_to_idx[f"act:{celex}"] = current_idx

                # Average embeddings of all articles in act
                act_emb = np.mean([art_embeddings_list[i] for i in art_indices], axis=0)
                act_embeddings_list.append(act_emb)

                act_times.append(0)  # Acts don't have temporal info

                act_idx_to_metadata[current_idx] = {
                    "id": f"act:{celex}",
                    "type": "legal_act",
                    "celex": celex,
                    "num_articles": len(art_indices),
                }

        # Add node features
        # Paragraph nodes: just embeddings (no case-level info)
        x_doc = torch.tensor(np.array(par_doc_embeddings_list), dtype=torch.float32)
        x_query = torch.tensor(np.array(par_query_embeddings_list), dtype=torch.float32)
        par_node_ids_tensor = torch.stack(par_node_ids)

        data["paragraph"].x = x_doc  # Document embeddings
        data["paragraph"].x_query = x_query  # Query embeddings
        data["paragraph"].time = torch.tensor(par_times, dtype=torch.long)
        data["paragraph"].node_id_hash = par_node_ids_tensor  # Hashed node IDs

        if self.include_articles:
            data["article"].x = torch.tensor(
                np.array(art_embeddings_list), dtype=torch.float32
            )
            data["article"].time = torch.tensor(art_times, dtype=torch.long)

        # Case nodes: case-level metadata (date, language, subject_matter, keywords, case_law_about)
        data["case"].x = torch.tensor(np.array(case_features_list), dtype=torch.float32)
        data["case"].time = torch.tensor(case_times, dtype=torch.long)

        if self.include_articles:
            data["legal_act"].x = torch.tensor(
                np.array(act_embeddings_list), dtype=torch.float32
            )
            data["legal_act"].time = torch.tensor(act_times, dtype=torch.long)

        # Build edges: citation (bidirectional) - only if include_citations is True
        citation_edges = []
        if self.include_citations:
            for src_id, tgt_id in self.citations:
                if src_id in par_node_id_to_idx and tgt_id in par_node_id_to_idx:
                    src_idx = par_node_id_to_idx[src_id]
                    tgt_idx = par_node_id_to_idx[tgt_id]
                    citation_edges.append([src_idx, tgt_idx])
                    citation_edges.append([tgt_idx, src_idx])

            if citation_edges:
                edge_index = torch.tensor(citation_edges, dtype=torch.long).t()
                data["paragraph", "cites", "paragraph"].edge_index = edge_index

        # Build edges: sequential (prev/next paragraph in same case)
        sequential_edges = []
        for celex, par_indices in case_to_par_indices.items():
            # Sort paragraphs by paragraph number
            sorted_pars = sorted(
                par_indices,
                key=lambda idx: par_idx_to_metadata[idx]["paragraph_number"],
            )
            # Connect consecutive paragraphs
            for i in range(len(sorted_pars) - 1):
                sequential_edges.append([sorted_pars[i], sorted_pars[i + 1]])
                sequential_edges.append([sorted_pars[i + 1], sorted_pars[i]])

        if sequential_edges:
            edge_index = torch.tensor(sequential_edges, dtype=torch.long).t()
            data["paragraph", "next", "paragraph"].edge_index = edge_index

        # Build edges: paragraph -> case
        par_to_case_edges = []
        for celex, par_indices in case_to_par_indices.items():
            case_idx = case_node_id_to_idx[f"case:{celex}"]
            for par_idx in par_indices:
                par_to_case_edges.append([par_idx, case_idx])

        if par_to_case_edges:
            edge_index = torch.tensor(par_to_case_edges, dtype=torch.long).t()
            data["paragraph", "belongs_to", "case"].edge_index = edge_index
            # Add reverse edge
            data["case", "contains", "paragraph"].edge_index = edge_index.flip([0])

        # Build edges: article -> legal_act - only if include_articles is True
        if self.include_articles:
            art_to_act_edges = []
            for celex, art_indices in act_to_art_indices.items():
                act_idx = act_node_id_to_idx[f"act:{celex}"]
                for art_idx in art_indices:
                    art_to_act_edges.append([art_idx, act_idx])

            if art_to_act_edges:
                edge_index = torch.tensor(art_to_act_edges, dtype=torch.long).t()
                data["article", "belongs_to", "legal_act"].edge_index = edge_index
                # Add reverse edge
                data["legal_act", "contains", "article"].edge_index = edge_index.flip(
                    [0]
                )

        print(f"Built heterogeneous graph:")
        print(
            f"  Paragraphs: {len(par_doc_embeddings_list)} (embedding dim: {x_doc.shape[1]})"
        )
        if self.include_articles:
            print(f"  Articles: {len(art_embeddings_list)}")
        print(
            f"  Cases: {len(case_features_list)} (feature dim: {data['case'].x.shape[1]})"
        )
        if self.include_articles:
            print(f"  Legal acts: {len(act_embeddings_list)}")
        print(f"  Edge types: {len(data.edge_types)}")
        print(f"  Include citations: {self.include_citations}")
        print(f"  Include articles: {self.include_articles}")

        return data

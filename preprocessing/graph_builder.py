import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData  # type: ignore
from torch_geometric.utils import add_self_loops  # type: ignore

# Fix OpenMP conflict on macOS (FAISS and PyTorch may use different OpenMP runtimes)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss

# Set FAISS to single-threaded mode to avoid segmentation faults
faiss.omp_set_num_threads(1)


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
    def build_graph(
        self, train_cutoff_year: int | None = None, include_only_citing: bool = False
    ):
        """Build and return graph data structure."""
        pass

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

    def _extract_date_features(self, date_str: str | None) -> np.ndarray:
        """Extract date feature: normalized days since 1954-01-01 (max 2025-12-31)."""
        if not date_str:
            # Return zero for missing dates
            return np.array([0.0], dtype=np.float32)
        try:
            dt = datetime.fromisoformat(date_str)
            # Calculate days since 1954-01-01
            base_date = datetime(1954, 1, 1)
            max_date = datetime(2025, 12, 31)
            days_since_base = (dt - base_date).days
            max_days = (max_date - base_date).days

            # Normalize to [0, 1] range, clamping values outside range
            time_norm = max(0.0, min(1.0, days_since_base / max_days))
            return np.array([time_norm], dtype=np.float32)
        except (ValueError, AttributeError):
            return np.array([0.0], dtype=np.float32)

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

    def _compute_semantic_edges(
        self,
        embeddings: np.ndarray,
        times: np.ndarray | None = None,
        threshold: float = 0.7,
        max_neighbors: int = 10,
        batch_size: int = 1024,
        use_temporal_constraint: bool = False,
    ) -> list[tuple[int, int]]:
        """
        Compute semantic similarity edges using FAISS.

        Args:
            embeddings: Node embeddings, shape (n, d)
            times: Timestamps for each node. If provided with use_temporal_constraint,
                   only links to nodes with earlier timestamps are created.
            threshold: Cosine similarity threshold for creating an edge
            max_neighbors: Maximum number of neighbors per node
            batch_size: Batch size for FAISS queries
            use_temporal_constraint: If True, only link to earlier nodes

        Returns:
            List of (source_idx, target_idx) edges
        """
        n, d = embeddings.shape
        print(f"  Computing semantic edges for {n} nodes using FAISS...")

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        embeddings_normalized = (embeddings / norms).astype(np.float32)

        if not embeddings_normalized.flags["C_CONTIGUOUS"]:
            embeddings_normalized = np.ascontiguousarray(embeddings_normalized)

        edges = self._compute_edges_temporal(
            embeddings_normalized, times, threshold, max_neighbors, batch_size
        )

        print(f"  Found {len(edges)} semantic similarity edges")
        return edges

    def _compute_edges_temporal(
        self,
        embeddings: np.ndarray,
        times: np.ndarray,
        threshold: float,
        max_neighbors: int,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        """
        Compute edges with temporal constraints.

        Process nodes in chronological order, building up the index incrementally.
        Each node can only find neighbors among nodes with earlier timestamps.
        """
        n, d = embeddings.shape

        time_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(n):
            time_to_indices[times[idx]].append(idx)

        unique_times = sorted(time_to_indices.keys())
        print(f"  Processing {len(unique_times)} time groups chronologically...")

        index = faiss.IndexFlatIP(d)
        faiss_to_orig: list[int] = []

        edges = []
        nodes_processed = 0

        for time_idx, t in enumerate(unique_times):
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

                    k = min(max_neighbors, index.ntotal)
                    similarities, faiss_neighbors = index.search(batch_embs, k)

                    for i, orig_idx in enumerate(batch_indices):
                        for j in range(k):
                            sim = similarities[i, j]
                            if sim >= threshold:
                                faiss_idx = faiss_neighbors[i, j]
                                neighbor_orig_idx = faiss_to_orig[faiss_idx]
                                edges.append((orig_idx, neighbor_orig_idx))

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
    Homogeneous graph builder with citation and optional semantic edges.

    Edge types:
    - 0: "cites" (forward direction: src cites tgt)
    - 1: "cited_by" (reverse direction: tgt is cited by src)
    - 2: "similar_to" (semantic similarity edges)

    Returns PyTorch Geometric Data object.
    """

    def build_graph(
        self,
        train_cutoff_year: int | None = None,
        include_only_citing: bool = True,
        include_self_loops: bool = False,
        add_reverse_edges: bool = True,
        include_semantic_edges: bool = False,
        semantic_threshold: float = 0.7,
        semantic_max_neighbors: int = 10,
    ) -> Data:
        """
        Build homogeneous citation graph.

        Args:
            train_cutoff_year: Only include paragraphs before this year
            include_only_citing: Only include paragraphs involved in citations
            include_self_loops: Whether to add self loops to all nodes
            add_reverse_edges: Whether to add reverse edges with different edge type
            include_semantic_edges: Whether to add semantic similarity edges
            semantic_threshold: Cosine similarity threshold for semantic edges
            semantic_max_neighbors: Max number of semantic neighbors per node

        Returns:
            graph_data: PyTorch Geometric Data object
        """
        # Filter paragraphs
        selected_pars = self._filter_paragraphs(include_only_citing, train_cutoff_year)

        relative_positions = self._compute_relative_positions(selected_pars)

        # Build node mappings
        node_id_to_idx: dict[str, int] = {}
        idx_to_metadata: dict[int, dict] = {}
        doc_embeddings_list = []
        query_embeddings_list = []
        date_features_list = []
        node_times = []
        node_ids = []

        for par_idx in selected_pars:
            meta = self.par_metadata[par_idx]
            node_id = meta["id"]
            current_idx = len(node_id_to_idx)

            node_id_to_idx[node_id] = current_idx
            idx_to_metadata[current_idx] = meta

            # Get base embeddings
            doc_emb = self.par_embeddings_doc[par_idx]
            query_emb = self.par_embeddings_query[par_idx]

            # Store date feature separately instead of concatenating
            date_feature = self._extract_date_features(meta.get("date"))

            doc_embeddings_list.append(doc_emb)
            query_embeddings_list.append(query_emb)
            date_features_list.append(date_feature)
            node_ids.append(encode_celex(meta["celex"], meta["paragraph_number"]))

            # Add timestamp (convert date to Unix timestamp)
            date_str = meta.get("date")
            node_times.append(self._date_to_timestamp(date_str))

        # Build citation edges with edge attributes
        # Edge type 0 = "cites" (forward direction: src cites tgt)
        # Edge type 1 = "cited_by" (reverse direction: tgt is cited by src)
        # Edge type 2 = "similar_to" (semantic similarity edges)
        edge_list = []
        edge_attr_list = []

        for src_id, tgt_id in self.citations:
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                tgt_idx = node_id_to_idx[tgt_id]

                # Forward edge: src -> tgt (src cites tgt)
                # This means information flows from cited (tgt) to citing (src)
                # Edge type 0 = "cites" direction
                edge_list.append([src_idx, tgt_idx])
                edge_attr_list.append(0)  # citing edge

                if add_reverse_edges:
                    # Reverse edge: tgt -> src (tgt is cited by src)
                    # Edge type 1 = "cited_by" direction
                    edge_list.append([tgt_idx, src_idx])
                    edge_attr_list.append(1)  # cited_by edge

        # Add semantic similarity edges (edge type 2)
        if include_semantic_edges:
            print("Computing semantic similarity edges...")
            embeddings = np.array(doc_embeddings_list)
            times_array = np.array(node_times)

            semantic_edges = self._compute_semantic_edges(
                embeddings,
                times=times_array,
                threshold=semantic_threshold,
                max_neighbors=semantic_max_neighbors,
            )

            for src_idx, tgt_idx in semantic_edges:
                edge_list.append([tgt_idx, src_idx])
                edge_attr_list.append(2)  # similar_to edge

        # Create PyTorch Geometric Data
        x_doc = torch.tensor(np.array(doc_embeddings_list), dtype=torch.float32)
        x_query = torch.tensor(np.array(query_embeddings_list), dtype=torch.float32)
        date_features = torch.tensor(np.array(date_features_list), dtype=torch.float32)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        # Add self loops if requested
        if include_self_loops:
            num_nodes = len(doc_embeddings_list)
            edge_index, edge_attr = add_self_loops(
                edge_index, edge_attr, num_nodes=num_nodes
            )

        node_times_tensor = torch.tensor(node_times, dtype=torch.long)
        node_ids_tensor = torch.stack(node_ids)

        graph_data = Data(
            x=x_doc,  # Default to document embeddings for backward compatibility
            x_query=x_query,  # Query embeddings (for citing paragraphs)
            date_feature=date_features,  # Date features stored separately
            edge_index=edge_index,
            edge_attr=edge_attr,  # Edge direction: 0=cites, 1=cited_by
            num_nodes=len(doc_embeddings_list),
            time=node_times_tensor,  # For temporal sampling
            node_id_hash=node_ids_tensor,  # Hashed node IDs
        )

        # Report embedding dimensions
        if self.has_case_metadata:
            base_dim = self.par_embeddings_doc.shape[1]
            metadata_dim = (
                self.case_embeddings_subject_matter.shape[1]
                + self.case_embeddings_keywords.shape[1]
                + self.case_embeddings_case_law_about.shape[1]
            )
            print(
                f"Node embedding dim: {base_dim} (base) + {metadata_dim} (metadata) = {x_doc.shape[1]}"
            )

        print(
            f"Built homogeneous graph: {len(doc_embeddings_list)} nodes, {edge_index.shape[1]} edges"
        )
        cites_count = (edge_attr == 0).sum().item()
        cited_by_count = (edge_attr == 1).sum().item()
        similar_count = (edge_attr == 2).sum().item()
        print(
            f"  Edge types: 0=cites ({cites_count}), 1=cited_by ({cited_by_count}), 2=similar_to ({similar_count})"
        )

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

    def build_graph(
        self,
        train_cutoff_year: int | None = None,
        include_only_citing: bool = False,
    ) -> HeteroData:
        """
        Build heterogeneous graph with case and act nodes.

        Args:
            train_cutoff_year: Only include paragraphs before this year
            include_only_citing: Only include paragraphs involved in citations

        Returns:
            graph_data: PyTorch Geometric HeteroData object
        """
        data = HeteroData()

        # Filter paragraphs
        selected_pars = self._filter_paragraphs(include_only_citing, train_cutoff_year)

        # Build paragraph nodes
        par_node_id_to_idx: dict[str, int] = {}
        par_idx_to_metadata: dict[int, dict] = {}
        par_doc_embeddings_list = []
        par_query_embeddings_list = []
        par_times = []
        par_node_ids = []
        case_to_par_indices: dict[str, list[int]] = defaultdict(list)

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

        # Build article nodes (all articles)
        art_node_id_to_idx: dict[str, int] = {}
        art_idx_to_metadata: dict[int, dict] = {}
        art_embeddings_list = []
        art_times = []
        act_to_art_indices: dict[str, list[int]] = defaultdict(list)

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

        # Build case nodes (average of paragraphs)
        case_node_id_to_idx: dict[str, int] = {}
        case_idx_to_metadata: dict[int, dict] = {}
        case_embeddings_list = []
        case_times = []

        for celex, par_indices in case_to_par_indices.items():
            current_idx = len(case_node_id_to_idx)
            case_node_id_to_idx[f"case:{celex}"] = current_idx

            # Average embeddings of all paragraphs in case (use doc embeddings)
            case_emb = np.mean(
                [par_doc_embeddings_list[i] for i in par_indices], axis=0
            )
            case_embeddings_list.append(case_emb)

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

        # Build legal act nodes (average of articles)
        act_node_id_to_idx: dict[str, int] = {}
        act_idx_to_metadata: dict[int, dict] = {}
        act_embeddings_list = []
        act_times = []

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
        x_doc = torch.tensor(np.array(par_doc_embeddings_list), dtype=torch.float32)
        x_query = torch.tensor(np.array(par_query_embeddings_list), dtype=torch.float32)
        par_node_ids_tensor = torch.stack(par_node_ids)

        data["paragraph"].x = x_doc  # Default to document embeddings
        data["paragraph"].x_query = x_query  # Query embeddings
        data["paragraph"].time = torch.tensor(par_times, dtype=torch.long)
        data["paragraph"].node_id_hash = par_node_ids_tensor  # Hashed node IDs

        data["article"].x = torch.tensor(
            np.array(art_embeddings_list), dtype=torch.float32
        )
        data["article"].time = torch.tensor(art_times, dtype=torch.long)

        data["case"].x = torch.tensor(
            np.array(case_embeddings_list), dtype=torch.float32
        )
        data["case"].time = torch.tensor(case_times, dtype=torch.long)

        data["legal_act"].x = torch.tensor(
            np.array(act_embeddings_list), dtype=torch.float32
        )
        data["legal_act"].time = torch.tensor(act_times, dtype=torch.long)

        # Build edges: citation (bidirectional)
        citation_edges = []
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

        # Build edges: article -> legal_act
        art_to_act_edges = []
        for celex, art_indices in act_to_art_indices.items():
            act_idx = act_node_id_to_idx[f"act:{celex}"]
            for art_idx in art_indices:
                art_to_act_edges.append([art_idx, act_idx])

        if art_to_act_edges:
            edge_index = torch.tensor(art_to_act_edges, dtype=torch.long).t()
            data["article", "belongs_to", "legal_act"].edge_index = edge_index
            # Add reverse edge
            data["legal_act", "contains", "article"].edge_index = edge_index.flip([0])

        print(f"Built heterogeneous graph:")
        print(f"  Paragraphs: {len(par_doc_embeddings_list)}")
        print(f"  Articles: {len(art_embeddings_list)}")
        print(f"  Cases: {len(case_embeddings_list)}")
        print(f"  Legal acts: {len(act_embeddings_list)}")
        print(f"  Edge types: {len(data.edge_types)}")

        return data

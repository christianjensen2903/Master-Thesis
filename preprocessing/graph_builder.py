"""
Graph builders for GNN training and evaluation.

Provides two graph builders:
- HomogeneousGraphBuilder: Simple citation graph with Data
- HeterogeneousGraphBuilder: Rich graph with case/act nodes using HeteroData
"""

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import re
import numpy as np
import torch
from torch_geometric.data import Data, HeteroData  # type: ignore


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

        # Create ID mappings
        self.par_id_to_idx = {m["id"]: i for i, m in enumerate(self.par_metadata)}
        self.art_id_to_idx = {m["id"]: i for i, m in enumerate(self.art_metadata)}

        print(
            f"Loaded {len(self.par_metadata)} paragraphs, {len(self.art_metadata)} articles"
        )
        print(f"Loaded {len(self.citations)} citation edges")

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
    Homogeneous graph builder with only bidirectional citing edges.

    Returns PyTorch Geometric Data object.
    """

    def build_graph(
        self,
        train_cutoff_year: int | None = None,
        include_only_citing: bool = True,
    ) -> Data:
        """
        Build homogeneous citation graph.

        Args:
            train_cutoff_year: Only include paragraphs before this year
            include_only_citing: Only include paragraphs involved in citations

        Returns:
            graph_data: PyTorch Geometric Data object
        """
        # Filter paragraphs
        selected_pars = self._filter_paragraphs(include_only_citing, train_cutoff_year)

        # Build node mappings
        node_id_to_idx: dict[str, int] = {}
        idx_to_metadata: dict[int, dict] = {}
        doc_embeddings_list = []
        query_embeddings_list = []
        node_times = []
        node_ids = []

        for par_idx in selected_pars:
            meta = self.par_metadata[par_idx]
            node_id = meta["id"]
            current_idx = len(node_id_to_idx)

            node_id_to_idx[node_id] = current_idx
            idx_to_metadata[current_idx] = meta
            doc_embeddings_list.append(self.par_embeddings_doc[par_idx])
            query_embeddings_list.append(self.par_embeddings_query[par_idx])
            node_ids.append(encode_celex(meta["celex"], meta["paragraph_number"]))

            # Add timestamp (convert date to Unix timestamp)
            date_str = meta.get("date")
            node_times.append(self._date_to_timestamp(date_str))

        # Build citation edges (bidirectional)
        edge_list = []
        for src_id, tgt_id in self.citations:
            if src_id in node_id_to_idx and tgt_id in node_id_to_idx:
                src_idx = node_id_to_idx[src_id]
                tgt_idx = node_id_to_idx[tgt_id]
                # Add both directions
                edge_list.append([src_idx, tgt_idx])
                edge_list.append([tgt_idx, src_idx])

        # Create PyTorch Geometric Data
        x_doc = torch.tensor(np.array(doc_embeddings_list), dtype=torch.float32)
        x_query = torch.tensor(np.array(query_embeddings_list), dtype=torch.float32)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        node_times_tensor = torch.tensor(node_times, dtype=torch.long)

        node_ids_tensor = torch.stack(node_ids)

        graph_data = Data(
            x=x_doc,  # Default to document embeddings for backward compatibility
            x_query=x_query,  # Query embeddings (for citing paragraphs)
            edge_index=edge_index,
            num_nodes=len(doc_embeddings_list),
            time=node_times_tensor,  # For temporal sampling
            node_id_hash=node_ids_tensor,  # Hashed node IDs
        )

        print(
            f"Built homogeneous graph: {len(doc_embeddings_list)} nodes, {edge_index.shape[1]} edges"
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

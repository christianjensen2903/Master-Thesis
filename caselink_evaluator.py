"""
CaseLink-style incremental evaluator.

Extends the base GNN evaluator with CaseLink-specific features:
- Support for semantic similarity edges
- Article node handling
- Multi-edge type graph structure
"""

import os

# Fix OpenMP conflict on macOS (FAISS and PyTorch may use different OpenMP runtimes)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from datetime import datetime

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from caselink import CaseLinkGraphBuilder
from preprocessing.graph_builder import decode_celex, encode_celex


def compute_ap(ranked_pids: np.ndarray, relevant_pids: set[tuple[str, int]]) -> float:
    """Compute average precision."""
    num_relevant = len(relevant_pids)
    if num_relevant == 0:
        return 0.0

    ranked_list = [
        tuple(pid.tolist()) if isinstance(pid, np.ndarray) else pid
        for pid in ranked_pids
    ]

    precisions = []
    num_hits = 0

    for i, pid in enumerate(ranked_list):
        if pid in relevant_pids:
            num_hits += 1
            precision = num_hits / (i + 1)
            precisions.append(precision)

    return sum(precisions) / num_relevant if precisions else 0.0


def compute_recall_at_k(
    ranked_pids: np.ndarray, relevant_pids: set[tuple[str, int]], k: int
) -> float:
    """Compute recall at k."""
    if len(relevant_pids) == 0:
        return 0.0

    top_k = ranked_pids[:k]
    top_k_list = [
        tuple(pid.tolist()) if isinstance(pid, np.ndarray) else pid for pid in top_k
    ]
    top_k_set = set(top_k_list)

    num_relevant_in_top_k = len(top_k_set & relevant_pids)
    return num_relevant_in_top_k / len(relevant_pids)


class CaseLinkEvaluator:
    """
    CaseLink-style incremental GNN evaluator.

    Key differences from base evaluator:
    - Uses CaseLinkGraphBuilder with semantic edges and article nodes
    - Handles multiple edge types
    - Only evaluates paragraph nodes (not articles)
    """

    def __init__(
        self,
        gnn_model: nn.Module,
        preprocessed_dir: str,
        par_to_par_path: str,
        train_cutoff_year: int = 2018,
        k_hops: int = 2,
        device: str | None = None,
        top_k: int = 10000,
        # CaseLink-specific
        include_semantic_edges: bool = True,
        semantic_threshold: float = 0.7,
        semantic_max_neighbors: int = 10,
        include_article_nodes: bool = True,
    ):
        self.gnn_model = gnn_model
        self.preprocessed_dir = preprocessed_dir
        self.par_to_par_path = par_to_par_path
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.top_k = top_k

        # CaseLink-specific
        self.include_semantic_edges = include_semantic_edges
        self.semantic_threshold = semantic_threshold
        self.semantic_max_neighbors = semantic_max_neighbors
        self.include_article_nodes = include_article_nodes

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gnn_model = self.gnn_model.to(self.device)
        self.gnn_model.eval()

        print(f"Using device: {self.device}")
        print(f"CaseLink evaluator settings:")
        print(f"  include_semantic_edges: {include_semantic_edges}")
        print(f"  include_article_nodes: {include_article_nodes}")

        # Build graph using CaseLinkGraphBuilder
        self.builder = CaseLinkGraphBuilder(preprocessed_dir)
        self.graph_data = self.builder.build_graph(
            include_only_citing=True,  # Only paragraphs involved in citations
            include_semantic_edges=include_semantic_edges,
            semantic_threshold=semantic_threshold,
            semantic_max_neighbors=semantic_max_neighbors,
            include_article_nodes=include_article_nodes,
        ).to(self.device)

        self.num_par_nodes = self.graph_data.num_par_nodes
        print(f"Graph has {self.num_par_nodes} paragraph nodes")

        # Create node_id_hash for paragraph nodes
        # We need to encode paragraph IDs for later decoding
        self._build_node_id_hash()

        # Compute initial embeddings up to cutoff year
        self.embeddings = self._compute_initial_embeddings(
            self.graph_data, self.train_cutoff_year
        )

    def _build_node_id_hash(self) -> None:
        """Build node ID hash from paragraph metadata for later decoding."""
        # Rebuild the paragraph selection logic to get IDs in correct order
        selected_pars = self.builder._filter_paragraphs(
            include_only_citing=True,
            train_cutoff_year=None,  # Include all for evaluation
        )

        node_id_hash_list = []
        for par_idx in selected_pars:
            meta = self.builder.par_metadata[par_idx]
            # Use the same encode_celex as the base graph builder
            encoded = encode_celex(meta["celex"], meta["paragraph_number"])
            node_id_hash_list.append(encoded)

        # Only store for paragraph nodes (first num_par_nodes)
        self.node_id_hash = torch.stack(node_id_hash_list[: self.num_par_nodes]).to(
            self.device
        )

        # Store mapping for quick lookup
        self.par_id_to_node_idx = {}
        for idx, par_idx in enumerate(selected_pars[: self.num_par_nodes]):
            meta = self.builder.par_metadata[par_idx]
            key = (meta["celex"], meta["paragraph_number"])
            self.par_id_to_node_idx[key] = idx

    def _compute_initial_embeddings(
        self, graph_data: Data, cutoff_year: int
    ) -> torch.Tensor:
        """Compute embeddings for all nodes up to cutoff year."""
        cutoff_timestamp = datetime.strptime(str(cutoff_year), "%Y").timestamp()

        # Filter edges based on time
        source_nodes = graph_data.edge_index[0]
        target_nodes = graph_data.edge_index[1]
        source_times = graph_data.time[source_nodes]
        target_times = graph_data.time[target_nodes]

        # Keep edges where both source and target are before cutoff
        # For article nodes (time=0), they're always included
        edge_mask = (source_times < cutoff_timestamp) & (
            target_times < cutoff_timestamp
        )
        edge_index_cutoff = graph_data.edge_index[:, edge_mask]

        # Filter edge_attr if present
        edge_attr_cutoff = None
        if hasattr(graph_data, "edge_attr") and graph_data.edge_attr is not None:
            edge_attr_cutoff = graph_data.edge_attr[edge_mask]

        with torch.no_grad():
            embeddings = self.gnn_model(
                graph_data.x,
                edge_index_cutoff,
                date_feature=graph_data.date_feature,
                edge_attr=edge_attr_cutoff,
                node_type=graph_data.node_type,
            )

        return embeddings

    def run(self, k_values: list[int] | None = None) -> float:
        """Run full incremental evaluation."""
        if k_values is None:
            k_values = [5, 10, 100]

        df = pd.read_csv(self.par_to_par_path)

        # Convert DATE_FROM to datetime and filter to after cutoff
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        cutoff_date = datetime.strptime(str(self.train_cutoff_year), "%Y")
        df = df[df["DATE_FROM"] >= cutoff_date]

        # Group by date for incremental evaluation
        grouped_by_date = df.groupby("DATE_FROM")

        ap_scores = []
        recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for date, group in tqdm(grouped_by_date, desc="Evaluating"):
            date_str = date.normalize().strftime("%Y-%m-%d")
            date_normalized = datetime.fromisoformat(date_str)
            time = int(date_normalized.timestamp())

            # Get paragraph nodes at this time (only paragraphs, not articles)
            node_type = self.graph_data.node_type
            par_mask = node_type == 0  # Only paragraphs
            time_mask = self.graph_data.time == time

            mask = par_mask & time_mask
            cand_mask = par_mask & (self.graph_data.time < time)

            # Get candidate indices (earlier paragraph nodes)
            cand_indices = torch.where(cand_mask)[0]
            cand_emb = self.embeddings[cand_indices]

            # Get nodes at current time
            nodes_at_time = mask.nonzero(as_tuple=True)[0]

            # Find which nodes have outgoing citations (from citation_pairs)
            citation_pairs = self.graph_data.citation_pairs
            citing_nodes = citation_pairs[0]  # Source nodes that cite

            nodes_with_out_edges = nodes_at_time[
                torch.isin(nodes_at_time, citing_nodes)
            ]

            if len(nodes_with_out_edges) == 0:
                continue

            num_nodes = nodes_with_out_edges.size(0)

            # Load subgraph for these nodes
            loader = NeighborLoader(
                data=self.graph_data,
                shuffle=False,
                input_nodes=nodes_with_out_edges,
                num_neighbors=[-1] * self.k_hops,
                time_attr="time",
                batch_size=100000,
                subgraph_type="bidirectional",
            )
            sub: Data = next(iter(loader))

            # Mask edges to prevent info leakage:
            # 1. Mask ALL citation edges (type 4) - they're only for neighbor sampling
            # 2. Mask outgoing edges from anchors
            src, tgt = sub.edge_index
            outgoing_from_anchor = src < num_nodes
            sub_edge_attr = (
                sub.edge_attr
                if hasattr(sub, "edge_attr") and sub.edge_attr is not None
                else None
            )
            is_citation_edge = (
                sub_edge_attr == 4
                if sub_edge_attr is not None
                else torch.zeros(
                    sub.edge_index.size(1),
                    dtype=torch.bool,
                    device=sub.edge_index.device,
                )
            )
            edge_mask = ~(outgoing_from_anchor | is_citation_edge)
            masked_edge_index = sub.edge_index[:, edge_mask]
            masked_edge_attr = (
                sub_edge_attr[edge_mask] if sub_edge_attr is not None else None
            )

            # Create feature matrix with query embeddings for input nodes
            x = sub.x.clone()
            if hasattr(sub, "x_query"):
                x[:num_nodes] = sub.x_query[:num_nodes]

            with torch.no_grad():
                embeddings = self.gnn_model(
                    x,
                    masked_edge_index,
                    date_feature=sub.date_feature,
                    edge_attr=masked_edge_attr,
                    node_type=sub.node_type,
                )

            query_emb = embeddings[:num_nodes]

            # Compute similarities
            sim = torch.matmul(query_emb, cand_emb.T)

            k = min(self.top_k, sim.size(1))
            _, sim_ord = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)

            # Decode query IDs
            query_ids = [
                decode_celex(node_id)
                for node_id in self.node_id_hash[nodes_with_out_edges]
            ]

            # Map ranked indices back to original graph and decode
            ranked_node_indices = cand_indices[sim_ord]
            ranked_ids = [
                [decode_celex(self.node_id_hash[idx]) for idx in row]
                for row in ranked_node_indices
            ]

            # Group by CELEX_FROM and NUMBER_FROM for evaluation
            grouped_by_celex_and_number = group.groupby(["CELEX_FROM", "NUMBER_FROM"])

            for i, (celex_from, number_from) in enumerate(query_ids):
                if (celex_from, number_from) not in grouped_by_celex_and_number.groups:
                    continue

                relevant_rows = group.loc[
                    grouped_by_celex_and_number.groups[(celex_from, number_from)]
                ]
                relevant_set = set(
                    zip(
                        relevant_rows["CELEX_TO"].astype(str),
                        relevant_rows["NUMBER_TO"].astype(int),
                    )
                )

                ranked_list = ranked_ids[i]
                ranked_array: NDArray = np.asarray(ranked_list, dtype=object)

                ap = compute_ap(ranked_array, relevant_set)
                ap_scores.append(ap)

                for k_val in k_values:
                    actual_k = min(k_val, len(ranked_array))
                    recall_at_k = compute_recall_at_k(
                        ranked_array, relevant_set, actual_k
                    )
                    recall_scores[k_val].append(recall_at_k)

            # Update embeddings for future queries
            with torch.no_grad():
                sub_n_id = sub.n_id
                expanded_loader = NeighborLoader(
                    data=self.graph_data,
                    shuffle=False,
                    input_nodes=sub_n_id,
                    num_neighbors=[-1] * self.k_hops,
                    time_attr="time",
                    batch_size=100000,
                    subgraph_type="bidirectional",
                )
                expanded_sub: Data = next(iter(expanded_loader))

                expanded_edge_attr = (
                    expanded_sub.edge_attr
                    if hasattr(expanded_sub, "edge_attr")
                    and expanded_sub.edge_attr is not None
                    else None
                )

                expanded_embeddings = self.gnn_model(
                    expanded_sub.x,
                    expanded_sub.edge_index,
                    date_feature=expanded_sub.date_feature,
                    edge_attr=expanded_edge_attr,
                    node_type=expanded_sub.node_type,
                )

                original_node_count = len(sub_n_id)
                embeddings_to_update = expanded_embeddings[:original_node_count]
                self.embeddings[sub_n_id] = embeddings_to_update

        map_score = float(np.mean(ap_scores)) if ap_scores else 0.0
        print(f"\nMAP: {map_score:.4f}")
        for k_val, recall in recall_scores.items():
            mean_recall = float(np.mean(recall)) if recall else 0.0
            print(f"Recall@{k_val}: {mean_recall:.4f}")

        return map_score


if __name__ == "__main__":
    from caselink import CaseLinkGNN

    # Initialize model
    model = CaseLinkGNN(
        input_dim=384,
        hidden_dim=384,
        output_dim=384,
        num_layers=1,
        num_edge_types=5,
    )

    # Load trained weights if available
    checkpoint_path = "output/caselink/best_model.pt"
    if os.path.exists(checkpoint_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(
            torch.load(checkpoint_path, map_location=device, weights_only=True)
        )
        print(f"Loaded model from {checkpoint_path}")

    # Run evaluation
    evaluator = CaseLinkEvaluator(
        gnn_model=model,
        preprocessed_dir="data/preprocessed",
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,
        k_hops=1,
        top_k=1000,
        include_semantic_edges=True,
        include_article_nodes=True,
    )

    map_score = evaluator.run(k_values=[5, 10, 100])

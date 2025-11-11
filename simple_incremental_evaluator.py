"""
Simple incremental GNN evaluator.

Workflow:
1. Build graph with all training data and embed it
2. Load test queries and qrels
3. For each test query in chronological order:
   - Embed query with modified graph (only incoming edges, no edges to citations)
   - Rank all nodes before it
   - Compute metrics (MAP, recall)
   - Add query to graph and re-embed its k-hop neighborhood
"""

import pickle
import csv
from collections import defaultdict
from evaluator import EvaluatorMode
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from tqdm import tqdm  # type: ignore

from preprocessing.graph_builder import HomogeneousGraphBuilder


def compute_ap(ranked_pids: np.ndarray, relevant_pids: set[str]) -> float:
    """Compute average precision."""
    num_relevant = len(relevant_pids)
    if num_relevant == 0:
        return 0.0

    precisions = []
    num_hits = 0

    for i, pid in enumerate(ranked_pids):
        if pid in relevant_pids:
            num_hits += 1
            precision = num_hits / (i + 1)
            precisions.append(precision)

    return sum(precisions) / num_relevant if precisions else 0.0


def compute_recall_at_k(
    ranked_pids: np.ndarray, relevant_pids: set[str], k: int
) -> float:
    """Compute recall at k."""
    if len(relevant_pids) == 0:
        return 0.0

    top_k = ranked_pids[:k]
    num_relevant_in_top_k = len(set(top_k) & relevant_pids)
    return num_relevant_in_top_k / len(relevant_pids)


class SimpleIncrementalEvaluator:
    """Simple incremental GNN evaluator."""

    def __init__(
        self,
        gnn_model: nn.Module,
        preprocessed_dir: str,
        queries_path: str,
        qrel_path: str,
        train_cutoff_year: int = 2018,
        k_hops: int = 2,
        device: str | None = None,
        mode: EvaluatorMode = "citation_pairs",
    ):
        self.gnn_model = gnn_model
        self.preprocessed_dir = preprocessed_dir
        self.queries_path = queries_path
        self.qrel_path = qrel_path
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.mode = mode

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gnn_model = self.gnn_model.to(self.device)
        self.gnn_model.eval()

        print(f"Using device: {self.device}")

        # Load preprocessed data
        self.builder = HomogeneousGraphBuilder(preprocessed_dir)

        self.graph_data = self.builder.build_graph(
            include_only_citing=(self.mode == "citation_pairs")
        )

        self.graph_data = self.graph_data.sort_by_time()

        self.embeddings = self._compute_initial_embeddings(
            self.graph_data, self.train_cutoff_year
        )

    def load_queries_and_qrels(
        self,
    ) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
        """
        Load test queries and qrels.

        Returns:
            queries: List of (query_id, query_text) for test queries
            qrels: Dict mapping query_id -> list of relevant doc ids
        """
        print("\nLoading queries...")
        queries = []

        with open(self.queries_path) as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # Skip header

            for celex, par_num, query_text in reader:
                query_id = f"par:{celex}:{par_num}"
                queries.append((query_id, query_text))

        print("\nLoading qrels...")
        qrels: dict[str, list[str]] = defaultdict(list)

        with open(self.qrel_path) as f:
            for line in f:
                parts = line.strip().split()
                query_id_raw = parts[0]
                doc_id_raw = parts[2]

                # Parse celex_paragraph format
                celex_q, par_num_q = query_id_raw.rsplit("_", 1)
                celex_d, par_num_d = doc_id_raw.rsplit("_", 1)

                query_id = f"par:{celex_q}:{par_num_q}"
                doc_id = f"par:{celex_d}:{par_num_d}"

                qrels[query_id].append(doc_id)

        print(f"Loaded {len(queries)} test queries with qrels")

        return queries, dict(qrels)

    def _compute_initial_embeddings(
        self, graph_data: Data, cutoff_year: int
    ) -> torch.Tensor:

        # convert cutoff year to timestamp
        cutoff_timestamp = datetime.strptime(str(cutoff_year), "%Y").timestamp()
        source_nodes = graph_data.edge_index[0]
        target_nodes = graph_data.edge_index[1]
        source_times = graph_data.time[source_nodes]
        target_times = graph_data.time[target_nodes]

        edge_mask = (source_times < cutoff_timestamp) & (
            target_times < cutoff_timestamp
        )
        edge_index_cutoff = graph_data.edge_index[:, edge_mask]

        with torch.no_grad():
            return self.gnn_model(graph_data.x, edge_index_cutoff)

    def run(self, k_values: list[int] = [5, 10, 100]) -> float:
        """Run full incremental evaluation."""
        # Load data
        queries, qrels = self.load_queries_and_qrels()

        cutoff_timestamp = datetime.strptime(
            str(self.train_cutoff_year), "%Y"
        ).timestamp()
        node_mask = self.graph_data.time >= cutoff_timestamp

        unique_times = self.graph_data.time[node_mask].unique().sort()[0]

        ap_scores = []
        recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for time in tqdm(unique_times, desc="Evaluating"):
            mask = self.graph_data.time == time
            cand_indices = self.graph_data.time < time
            cand_emb = self.embeddings[cand_indices]

            num_nodes = mask.sum()

            loader = NeighborLoader(
                data=self.graph_data,
                shuffle=False,
                input_nodes=mask.nonzero(as_tuple=True)[0],
                num_neighbors=[-1] * self.k_hops,
                time_attr="time",
            )
            sub = next(iter(loader))

            src, dst = sub.edge_index
            edge_mask = (dst < num_nodes) | (src >= num_nodes)
            masked_edge_index = sub.edge_index[:, edge_mask]

            with torch.no_grad():
                embeddings = self.gnn_model(sub.x, masked_edge_index)

            query_global_idx = sub.node_indices[:num_nodes]
            cand_global_idx = torch.nonzero(cand_indices, as_tuple=False).squeeze(1)

            cand_emb = self.embeddings[cand_global_idx]
            query_emb = embeddings[:num_nodes]
            sim = torch.matmul(query_emb, cand_emb.T)

            _, sim_ord = torch.topk(sim, k=10, dim=1, largest=True, sorted=True)

            ranked_global_idx = cand_global_idx[sim_ord]

            def to_id(gidx: int):
                return self.builder.par_metadata[gidx]["id"]

            query_ids = [to_id(int(g)) for g in query_global_idx]
            ranked_ids = [[to_id(int(g)) for g in row] for row in ranked_global_idx]

            for qi, qid in enumerate(query_ids):
                relevant_set = set(qrels.get(qid, []))  # robust if missing
                ranked_list = ranked_ids[qi]

                ranked_array: NDArray = np.asarray(ranked_list)
                ap = compute_ap(ranked_array, relevant_set)
                for k in k_values:
                    k = min(k, len(ranked_array))
                    recall_at_k = compute_recall_at_k(ranked_array, relevant_set, k)
                    recall_scores[k].append(recall_at_k)
                ap_scores.append(ap)

            with torch.no_grad():
                embeddings = self.gnn_model(sub.x, sub.edge_index)
                # Update stored embeddings for the nodes present in this subgraph
                self.embeddings[sub.n_id] = embeddings

        map = float(np.mean(ap_scores))
        print(f"MAP: {map}")
        for k, recall in recall_scores.items():
            print(f"Recall@{k}: {float(np.mean(recall))}")
        return map


if __name__ == "__main__":
    from example_gnn_usage import CitationGNN
    from sentence_transformers import SentenceTransformer

    # Load model
    encoding_model = "checkpoints/simcse_citation_model"
    text_encoder = SentenceTransformer(encoding_model)
    in_channels = text_encoder.get_sentence_embedding_dimension()

    model = CitationGNN(
        in_channels, hidden_dim=512, output_dim=in_channels, num_layers=2
    )
    model.load_state_dict(torch.load("checkpoints/gnn/best_model.pt"))

    # Run evaluation
    evaluator = SimpleIncrementalEvaluator(
        gnn_model=model,
        preprocessed_dir="data/preprocessed",
        queries_path="data/evaluation/queries_cleaned_masked.tsv",
        qrel_path="data/evaluation/qrel.txt",
        train_cutoff_year=2018,
        k_hops=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = evaluator.run()

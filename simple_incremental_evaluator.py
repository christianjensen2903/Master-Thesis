import pandas as pd
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

from preprocessing.graph_builder import HomogeneousGraphBuilder, decode_celex


def compute_ap(ranked_pids: np.ndarray, relevant_pids: set[tuple[str, int]]) -> float:
    """Compute average precision."""
    num_relevant = len(relevant_pids)
    if num_relevant == 0:
        return 0.0

    # Convert numpy array to list of tuples to ensure hashability
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
    # Convert numpy array elements to tuples for set operations
    top_k_list = [
        tuple(pid.tolist()) if isinstance(pid, np.ndarray) else pid for pid in top_k
    ]
    top_k_set = set(top_k_list)

    num_relevant_in_top_k = len(top_k_set & relevant_pids)
    return num_relevant_in_top_k / len(relevant_pids)


class SimpleIncrementalEvaluator:
    """Simple incremental GNN evaluator."""

    def __init__(
        self,
        gnn_model: nn.Module,
        preprocessed_dir: str,
        par_to_par_path: str,
        train_cutoff_year: int = 2018,
        k_hops: int = 2,
        device: str | None = None,
        mode: EvaluatorMode = "citation_pairs",
    ):
        self.gnn_model = gnn_model
        self.preprocessed_dir = preprocessed_dir
        self.par_to_par_path = par_to_par_path
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

        self.embeddings = self._compute_initial_embeddings(
            self.graph_data, self.train_cutoff_year
        )

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
        df = pd.read_csv(self.par_to_par_path)

        # Convert DATE_FROM to datetime and only keep those after the cutoff year
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        cutoff_date = datetime.strptime(str(self.train_cutoff_year), "%Y")
        df = df[df["DATE_FROM"] >= cutoff_date]

        # Group by DATE_FROM
        grouped_by_date = df.groupby("DATE_FROM")

        ap_scores = []
        recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for date, group in tqdm(grouped_by_date, desc="Evaluating"):

            date_str = date.normalize().strftime("%Y-%m-%d")
            date_normalized = datetime.fromisoformat(date_str)
            time = int(date_normalized.timestamp())

            # Try exact match first
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
                batch_size=10000,
            )
            sub: Data = next(iter(loader))

            src, dst = sub.edge_index
            edge_mask = (dst < num_nodes) | (src >= num_nodes)
            masked_edge_index = sub.edge_index[:, edge_mask]

            with torch.no_grad():
                embeddings = self.gnn_model(sub.x, masked_edge_index)

            query_emb = embeddings[:num_nodes]
            sim = torch.matmul(query_emb, cand_emb.T)

            _, sim_ord = torch.topk(sim, k=1000, dim=1, largest=True, sorted=True)

            query_ids = [
                decode_celex(node_id) for node_id in sub.node_id_hash[:num_nodes]
            ]

            ranked_ids = [
                [decode_celex(node_id) for node_id in row]
                for row in self.graph_data.node_id_hash[sim_ord]
            ]

            # Group by CELEX_FROM and NUMBER_FROM
            grouped_by_celex_and_number = group.groupby(["CELEX_FROM", "NUMBER_FROM"])

            for celex_from, number_from in query_ids:
                # Only evaluate queries that actually cite (appear in citation data as sources)
                if (celex_from, number_from) not in grouped_by_celex_and_number.groups:
                    # Skip nodes that are only cited but don't cite anything themselves
                    continue

                # Get the relevant rows for this query
                relevant_rows = group.loc[
                    grouped_by_celex_and_number.groups[(celex_from, number_from)]
                ]
                # Extract (CELEX_TO, NUMBER_TO) tuples as the relevant set
                relevant_set = set(
                    zip(
                        relevant_rows["CELEX_TO"].astype(str),
                        relevant_rows["NUMBER_TO"].astype(int),
                    )
                )

                ranked_list = ranked_ids[query_ids.index((celex_from, number_from))]

                ranked_array: NDArray = np.asarray(ranked_list, dtype=object)
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
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,
        k_hops=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = evaluator.run()

from typing import Hashable
from pandas.core.frame import DataFrame
from pandas.core.series import Series
import pandas as pd
from evaluator import EvaluatorMode
from datetime import datetime

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData  # type: ignore
from torch_geometric.utils import k_hop_subgraph  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from tqdm import tqdm  # type: ignore
from torch_geometric.transforms import ToUndirected

from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
    decode_celex,
)


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
        top_k: int = 10000,
        graph_type: str = "homogeneous",
    ):
        self.gnn_model = gnn_model
        self.preprocessed_dir = preprocessed_dir
        self.par_to_par_path = par_to_par_path
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.mode = mode
        self.top_k = top_k
        self.graph_type = graph_type
        self.is_hetero = graph_type == "heterogeneous"

        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gnn_model = self.gnn_model.to(self.device)
        self.gnn_model.eval()

        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")

        graph_data: Data | HeteroData
        if self.is_hetero:
            graph_data = HeterogeneousGraphBuilder(preprocessed_dir).build_graph(
                include_only_citing=False
            )
            graph_data = ToUndirected()(graph_data)
        else:
            graph_data = HomogeneousGraphBuilder(preprocessed_dir).build_graph(
                include_only_citing=(self.mode == "citation_pairs"),
                add_reverse_edges=True,
            )
        self.graph_data = graph_data

        self.embeddings = self._compute_initial_embeddings(
            self.graph_data, self.train_cutoff_year
        )

        if self.is_hetero and self.mode == "citation_pairs":
            if ("paragraph", "cites", "paragraph") in self.graph_data.edge_types:
                cite_edges = self.graph_data[
                    "paragraph", "cites", "paragraph"
                ].edge_index
                citing_paragraphs = torch.unique(cite_edges[0])
                cited_paragraphs = torch.unique(cite_edges[1])
                involved_in_citations = torch.cat([citing_paragraphs, cited_paragraphs])
                self.citation_involved_mask = torch.unique(involved_in_citations)
            else:
                self.citation_involved_mask = torch.tensor([], dtype=torch.long)
        else:
            self.citation_involved_mask = None

    def _compute_initial_embeddings(
        self, graph_data: Data | HeteroData, cutoff_year: int
    ) -> torch.Tensor:
        cutoff_timestamp = datetime.strptime(str(cutoff_year), "%Y").timestamp()

        if self.is_hetero:
            par_time = graph_data["paragraph"].time
            case_time = (
                graph_data["case"].time if "case" in graph_data.node_types else None
            )
            art_time = (
                graph_data["article"].time
                if "article" in graph_data.node_types
                else None
            )

            modified_data = graph_data.clone()

            for edge_type in graph_data.edge_types:
                src_node_type, _, dst_node_type = edge_type
                edge_index = graph_data[edge_type].edge_index

                if src_node_type == "paragraph":
                    source_times = par_time[edge_index[0]]
                elif src_node_type == "case" and case_time is not None:
                    source_times = case_time[edge_index[0]]
                elif src_node_type == "article" and art_time is not None:
                    source_times = art_time[edge_index[0]]
                else:
                    continue

                if dst_node_type == "paragraph":
                    target_times = par_time[edge_index[1]]
                elif dst_node_type == "case" and case_time is not None:
                    target_times = case_time[edge_index[1]]
                elif dst_node_type == "article" and art_time is not None:
                    target_times = art_time[edge_index[1]]
                else:
                    continue

                edge_mask = (source_times < cutoff_timestamp) & (
                    target_times < cutoff_timestamp
                )
                modified_data[edge_type].edge_index = edge_index[:, edge_mask]

            with torch.no_grad():
                out = self.gnn_model(modified_data)
                return out["paragraph"]
        else:
            source_nodes = graph_data.edge_index[0]
            target_nodes = graph_data.edge_index[1]
            source_times = graph_data.time[source_nodes]
            target_times = graph_data.time[target_nodes]

            edge_mask = (source_times < cutoff_timestamp) & (
                target_times < cutoff_timestamp
            )
            edge_index_cutoff = graph_data.edge_index[:, edge_mask]

            edge_attr_cutoff = None
            if hasattr(graph_data, "edge_attr") and graph_data.edge_attr is not None:
                edge_attr_cutoff = graph_data.edge_attr[edge_mask]

            with torch.no_grad():
                return self.gnn_model.encode_candidates(
                    graph_data.x,
                    edge_index_cutoff,
                    date_feature=graph_data.date_feature,
                    edge_attr=edge_attr_cutoff,
                )

    def run(self, k_values: list[int] = [5, 10, 100]) -> float:
        """Run full incremental evaluation."""
        df = pd.read_csv(self.par_to_par_path)

        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        cutoff_date = datetime.strptime(str(self.train_cutoff_year), "%Y")
        df = df[df["DATE_FROM"] >= cutoff_date]

        grouped_by_date = df.groupby("DATE_FROM")

        ap_scores = []
        recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for date, group in tqdm(grouped_by_date, desc="Evaluating"):

            date_str = date.normalize().strftime("%Y-%m-%d")
            date_normalized = datetime.fromisoformat(date_str)
            time = int(date_normalized.timestamp())

            if self.is_hetero:
                par_time = self.graph_data["paragraph"].time
                mask = par_time == time
                cand_mask = par_time < time
                node_id_hash = self.graph_data["paragraph"].node_id_hash
                source_nodes = self.graph_data[
                    "paragraph", "cites", "paragraph"
                ].edge_index[0]
            else:
                mask = self.graph_data.time == time
                cand_mask = self.graph_data.time < time
                node_id_hash = self.graph_data.node_id_hash
                if (
                    hasattr(self.graph_data, "edge_attr")
                    and self.graph_data.edge_attr is not None
                ):
                    cites_mask = self.graph_data.edge_attr == 0
                    source_nodes = self.graph_data.edge_index[0, cites_mask]
                else:
                    source_nodes = self.graph_data.edge_index[0]

            cand_indices = torch.where(cand_mask)[0]

            if self.citation_involved_mask is not None:
                cand_indices = cand_indices[
                    torch.isin(cand_indices, self.citation_involved_mask)
                ]

            cand_emb = self.embeddings[cand_indices]

            nodes_at_time = mask.nonzero(as_tuple=True)[0]

            nodes_with_out_edges = nodes_at_time[
                torch.isin(nodes_at_time, source_nodes)
            ]

            num_nodes = nodes_with_out_edges.size(0)

            if self.is_hetero:
                input_nodes = ("paragraph", nodes_with_out_edges)
            else:
                input_nodes = nodes_with_out_edges

            loader = NeighborLoader(
                data=self.graph_data,
                shuffle=False,
                input_nodes=input_nodes,
                num_neighbors=[-1] * self.k_hops,
                time_attr="time",
                batch_size=100000,
                subgraph_type="bidirectional",
            )
            sub: Data | HeteroData = next(iter(loader))

            if self.is_hetero:
                cite_edge_index = sub["paragraph", "cites", "paragraph"].edge_index
                src, tgt = cite_edge_index
                edge_mask = ~((src < num_nodes) & (tgt >= num_nodes))
                masked_cite_edges = cite_edge_index[:, edge_mask]

                modified_sub = sub.clone()
                modified_sub["paragraph", "cites", "paragraph"].edge_index = (
                    masked_cite_edges
                )

                x = sub["paragraph"].x.clone()
                x[:num_nodes] = sub["paragraph"].x_query[:num_nodes]
                modified_sub["paragraph"].x = x

                with torch.no_grad():
                    out = self.gnn_model(modified_sub)
                    embeddings = out["paragraph"]

                query_emb = embeddings[:num_nodes]
                sub_node_id_hash = sub["paragraph"].node_id_hash
                sub_n_id = sub["paragraph"].n_id

            else:
                # Query embedding - use encode_query (no graph structure)
                x_query = (
                    sub.x_query[:num_nodes]
                    if hasattr(sub, "x_query")
                    else sub.x[:num_nodes]
                )
                with torch.no_grad():
                    query_emb = self.gnn_model.encode_query(x_query)

                sub_node_id_hash = sub.node_id_hash
                sub_n_id = sub.n_id

            sim = torch.matmul(query_emb, cand_emb.T)

            k = min(self.top_k, sim.size(1))
            _, sim_ord = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)

            query_ids = [
                decode_celex(node_id) for node_id in sub_node_id_hash[:num_nodes]
            ]

            ranked_node_indices = cand_indices[sim_ord]
            ranked_ids = [
                [decode_celex(node_id) for node_id in row]
                for row in node_id_hash[ranked_node_indices]
            ]

            grouped_by_celex_and_number = group.groupby(["CELEX_FROM", "NUMBER_FROM"])

            for celex_from, number_from in query_ids:
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

                ranked_list = ranked_ids[query_ids.index((celex_from, number_from))]

                ranked_array: NDArray = np.asarray(ranked_list, dtype=object)
                ap = compute_ap(ranked_array, relevant_set)
                for k in k_values:
                    k = min(k, len(ranked_array))
                    recall_at_k = compute_recall_at_k(ranked_array, relevant_set, k)
                    recall_scores[k].append(recall_at_k)
                ap_scores.append(ap)

            # Update embeddings with candidate embeddings (for future queries to cite them)
            with torch.no_grad():
                if self.is_hetero:
                    expanded_input_nodes = ("paragraph", sub_n_id)
                    expanded_loader = NeighborLoader(
                        data=self.graph_data,
                        shuffle=False,
                        input_nodes=expanded_input_nodes,
                        num_neighbors=[-1] * self.k_hops,
                        time_attr="time",
                        batch_size=100000,
                        subgraph_type="bidirectional",
                    )
                    expanded_sub: HeteroData = next(iter(expanded_loader))

                    out = self.gnn_model(expanded_sub)
                    expanded_embeddings = out["paragraph"]

                    original_node_count = len(sub_n_id)
                    embeddings_to_update = expanded_embeddings[:original_node_count]
                    self.embeddings[sub_n_id] = embeddings_to_update
                else:
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

                    expanded_edge_attr = None
                    if (
                        hasattr(expanded_sub, "edge_attr")
                        and expanded_sub.edge_attr is not None
                    ):
                        expanded_edge_attr = expanded_sub.edge_attr

                    # Use encode_candidates for document embeddings
                    expanded_embeddings = self.gnn_model.encode_candidates(
                        expanded_sub.x,
                        expanded_sub.edge_index,
                        date_feature=expanded_sub.date_feature,
                        edge_attr=expanded_edge_attr,
                    )

                    original_node_count = len(sub_n_id)
                    embeddings_to_update = expanded_embeddings[:original_node_count]
                    self.embeddings[sub_n_id] = embeddings_to_update

        map = float(np.mean(ap_scores))
        print(f"MAP: {map}")
        for k, recall in recall_scores.items():
            print(f"Recall@{k}: {float(np.mean(recall))}")
        return map


if __name__ == "__main__":
    from models import CitationGNN

    # Load model
    in_channels = 384
    hidden_dim = 384
    out_channels = 384

    model = CitationGNN(in_channels, output_dim=out_channels, num_layers=2)
    model.load_state_dict(torch.load("checkpoints/homo_gnn/best_model.pt"))

    # Run evaluation
    evaluator = SimpleIncrementalEvaluator(
        gnn_model=model,
        preprocessed_dir="data/preprocessed",
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,
        k_hops=2,
        device="cuda" if torch.cuda.is_available() else "cpu",
        top_k=1000,
        graph_type="homogeneous",
    )

    metrics = evaluator.run()

from datetime import datetime
import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
from torch_geometric.data import Data, HeteroData  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.transforms import ToUndirected  # type: ignore
from tqdm import tqdm  # type: ignore

from evaluator import EvaluatorMode
from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
    decode_celex,
)


def compute_ap(ranked_pids: np.ndarray, relevant_pids: set[tuple[str, int]]) -> float:
    """Compute average precision."""
    if not relevant_pids:
        return 0.0

    ranked_list = [
        tuple(pid.tolist()) if isinstance(pid, np.ndarray) else pid
        for pid in ranked_pids
    ]

    precisions = []
    hits = 0
    for i, pid in enumerate(ranked_list):
        if pid in relevant_pids:
            hits += 1
            precisions.append(hits / (i + 1))

    return sum(precisions) / len(relevant_pids) if precisions else 0.0


def compute_recall_at_k(
    ranked_pids: np.ndarray, relevant_pids: set[tuple[str, int]], k: int
) -> float:
    """Compute recall at k."""
    if not relevant_pids:
        return 0.0

    top_k = [
        tuple(pid.tolist()) if isinstance(pid, np.ndarray) else pid
        for pid in ranked_pids[:k]
    ]
    return len(set(top_k) & relevant_pids) / len(relevant_pids)


class SimpleIncrementalEvaluator:
    """Incremental GNN evaluator with support for heterogeneous and homogeneous graphs."""

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
        include_semantic_edges: bool = False,
        semantic_threshold: float = 0.7,
        semantic_max_neighbors: int = 10,
    ):
        self.preprocessed_dir = preprocessed_dir
        self.par_to_par_path = par_to_par_path
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.mode = mode
        self.top_k = top_k
        self.graph_type = graph_type
        self.is_hetero = graph_type == "heterogeneous"
        self.include_semantic_edges = include_semantic_edges
        self.semantic_threshold = semantic_threshold
        self.semantic_max_neighbors = semantic_max_neighbors

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gnn_model = gnn_model.to(self.device)
        self.gnn_model.eval()

        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")
        if include_semantic_edges and not self.is_hetero:
            print(
                f"  Semantic edges: threshold={semantic_threshold}, max_neighbors={semantic_max_neighbors}"
            )

        self.graph_data = self._build_graph()
        self.embeddings = self._compute_initial_embeddings()
        self.citation_mask = self._build_citation_mask()

    def _build_graph(self) -> Data | HeteroData:
        """Build the evaluation graph."""
        if self.is_hetero:
            graph = HeterogeneousGraphBuilder(self.preprocessed_dir).build_graph(
                include_only_citing=False
            )
            return ToUndirected()(graph)
        else:
            return HomogeneousGraphBuilder(self.preprocessed_dir).build_graph(
                include_only_citing=(self.mode == "citation_pairs"),
                add_reverse_edges=True,
                include_semantic_edges=self.include_semantic_edges,
                semantic_threshold=self.semantic_threshold,
                semantic_max_neighbors=self.semantic_max_neighbors,
            )

    def _build_citation_mask(self) -> torch.Tensor | None:
        """Build mask of paragraphs involved in citations (for hetero graphs in citation_pairs mode)."""
        if not (self.is_hetero and self.mode == "citation_pairs"):
            return None

        if ("paragraph", "cites", "paragraph") not in self.graph_data.edge_types:
            return torch.tensor([], dtype=torch.long)

        cite_edges = self.graph_data["paragraph", "cites", "paragraph"].edge_index
        return torch.unique(torch.cat([cite_edges[0], cite_edges[1]]))

    def _compute_initial_embeddings(self) -> torch.Tensor:
        """Compute initial embeddings for all nodes before cutoff year."""
        cutoff_ts = datetime.strptime(str(self.train_cutoff_year), "%Y").timestamp()

        if self.is_hetero:
            return self._compute_hetero_embeddings(cutoff_ts)
        return self._compute_homo_embeddings(cutoff_ts)

    def _compute_hetero_embeddings(self, cutoff_ts: float) -> torch.Tensor:
        """Compute embeddings for heterogeneous graph."""
        modified = self.graph_data.clone()
        node_times = {
            "paragraph": self.graph_data["paragraph"].time,
            "case": (
                getattr(self.graph_data.get("case"), "time", None)
                if "case" in self.graph_data.node_types
                else None
            ),
            "article": (
                getattr(self.graph_data.get("article"), "time", None)
                if "article" in self.graph_data.node_types
                else None
            ),
        }

        for edge_type in self.graph_data.edge_types:
            src_type, _, dst_type = edge_type
            edge_index = self.graph_data[edge_type].edge_index

            src_times = node_times.get(src_type)
            dst_times = node_times.get(dst_type)

            if src_times is None or dst_times is None:
                continue

            mask = (src_times[edge_index[0]] < cutoff_ts) & (
                dst_times[edge_index[1]] < cutoff_ts
            )
            modified[edge_type].edge_index = edge_index[:, mask]

        with torch.no_grad():
            return self.gnn_model(modified)["paragraph"]

    def _compute_homo_embeddings(self, cutoff_ts: float) -> torch.Tensor:
        """Compute embeddings for homogeneous graph (document embeddings for corpus)."""
        edge_index = self.graph_data.edge_index
        times = self.graph_data.time

        mask = (times[edge_index[0]] < cutoff_ts) & (times[edge_index[1]] < cutoff_ts)
        filtered_edges = edge_index[:, mask]
        filtered_attr = (
            self.graph_data.edge_attr[mask]
            if hasattr(self.graph_data, "edge_attr")
            and self.graph_data.edge_attr is not None
            else None
        )
        language = getattr(self.graph_data, "language", None)

        with torch.no_grad():
            # Use document encoder for corpus embeddings if dual encoder
            if hasattr(self.gnn_model, "encode_document"):
                return self.gnn_model.encode_document(
                    self.graph_data.x,
                    filtered_edges,
                    date_feature=getattr(self.graph_data, "date_feature", None),
                    edge_attr=filtered_attr,
                    language=language,
                )
            return self.gnn_model(
                self.graph_data.x,
                filtered_edges,
                date_feature=self.graph_data.date_feature,
                edge_attr=filtered_attr,
                language=language,
            )

    def _get_graph_attrs(self) -> tuple:
        """Get graph attributes based on graph type."""
        if self.is_hetero:
            return (
                self.graph_data["paragraph"].time,
                self.graph_data["paragraph"].node_id_hash,
                self.graph_data["paragraph", "cites", "paragraph"].edge_index[0],
            )
        else:
            edge_attr = getattr(self.graph_data, "edge_attr", None)
            if edge_attr is not None:
                source_nodes = self.graph_data.edge_index[0, edge_attr == 0]
            else:
                source_nodes = self.graph_data.edge_index[0]
            return self.graph_data.time, self.graph_data.node_id_hash, source_nodes

    def _create_loader(self, input_nodes) -> NeighborLoader:
        """Create a NeighborLoader for the given input nodes."""
        return NeighborLoader(
            data=self.graph_data,
            shuffle=False,
            input_nodes=input_nodes,
            num_neighbors=[-1] * self.k_hops,
            time_attr="time",
            batch_size=100000,
            subgraph_type="bidirectional",
        )

    def _process_subgraph(self, sub: Data | HeteroData, num_nodes: int) -> torch.Tensor:
        """Process subgraph and return query embeddings."""
        if self.is_hetero:
            return self._process_hetero_subgraph(sub, num_nodes)
        return self._process_homo_subgraph(sub, num_nodes)

    def _process_hetero_subgraph(self, sub: HeteroData, num_nodes: int) -> torch.Tensor:
        """Process heterogeneous subgraph."""
        cite_edges = sub["paragraph", "cites", "paragraph"].edge_index
        src, tgt = cite_edges

        # Mask leaking edges
        mask = ~((src < num_nodes) & (tgt >= num_nodes))
        modified = sub.clone()
        modified["paragraph", "cites", "paragraph"].edge_index = cite_edges[:, mask]

        # Use query features for anchor nodes
        x = sub["paragraph"].x.clone()
        x[:num_nodes] = sub["paragraph"].x_query[:num_nodes]
        modified["paragraph"].x = x

        with torch.no_grad():
            return self.gnn_model(modified)["paragraph"][:num_nodes]

    def _process_homo_subgraph(self, sub: Data, num_nodes: int) -> torch.Tensor:
        """Process homogeneous subgraph."""
        # Use query features for anchor nodes
        x = sub.x.clone()
        if hasattr(sub, "x_query"):
            x[:num_nodes] = sub.x_query[:num_nodes]

        with torch.no_grad():
            # Use query encoder for queries if dual encoder
            if hasattr(self.gnn_model, "encode_query"):
                date_feature = getattr(sub, "date_feature", None)
                language = getattr(sub, "language", None)
                return self.gnn_model.encode_query(
                    x[:num_nodes],
                    date_feature=(
                        date_feature[:num_nodes] if date_feature is not None else None
                    ),
                    language=language[:num_nodes] if language is not None else None,
                )

            # Fall back to full model with edge masking
            src, tgt = sub.edge_index
            edge_attr = getattr(sub, "edge_attr", None)
            language = getattr(sub, "language", None)

            outgoing = src < num_nodes
            incoming = tgt < num_nodes

            if edge_attr is not None:
                is_citation = (edge_attr == 0) | (edge_attr == 1)
                leakage_mask = outgoing | (incoming & is_citation)
            else:
                leakage_mask = outgoing | incoming

            masked_edges = sub.edge_index[:, ~leakage_mask]
            masked_attr = edge_attr[~leakage_mask] if edge_attr is not None else None

            embeddings = self.gnn_model(
                x,
                masked_edges,
                date_feature=sub.date_feature,
                edge_attr=masked_attr,
                language=language,
            )
            return embeddings[:num_nodes]

    def _update_embeddings(self, sub: Data | HeteroData, sub_n_id: torch.Tensor):
        """Update embeddings for nodes in the subgraph (document embeddings)."""
        loader = self._create_loader(
            ("paragraph", sub_n_id) if self.is_hetero else sub_n_id
        )
        expanded_sub = next(iter(loader))

        with torch.no_grad():
            if self.is_hetero:
                embeddings = self.gnn_model(expanded_sub)["paragraph"]
            else:
                edge_attr = getattr(expanded_sub, "edge_attr", None)
                date_feature = getattr(expanded_sub, "date_feature", None)
                language = getattr(expanded_sub, "language", None)
                # Use document encoder if dual encoder
                if hasattr(self.gnn_model, "encode_document"):
                    embeddings = self.gnn_model.encode_document(
                        expanded_sub.x,
                        expanded_sub.edge_index,
                        date_feature=date_feature,
                        edge_attr=edge_attr,
                        language=language,
                    )
                else:
                    embeddings = self.gnn_model(
                        expanded_sub.x,
                        expanded_sub.edge_index,
                        date_feature=date_feature,
                        edge_attr=edge_attr,
                        language=language,
                    )

        self.embeddings[sub_n_id] = embeddings[: len(sub_n_id)]

    def run(self, k_values: list[int] = [5, 10, 100]) -> float:
        """Run full incremental evaluation."""
        df = pd.read_csv(self.par_to_par_path)
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        cutoff_date = datetime.strptime(str(self.train_cutoff_year), "%Y")
        df = df[df["DATE_FROM"] >= cutoff_date]

        times, node_id_hash, source_nodes = self._get_graph_attrs()

        ap_scores = []
        recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for date, group in tqdm(df.groupby("DATE_FROM"), desc="Evaluating"):
            date_str = date.normalize().strftime("%Y-%m-%d")
            timestamp = int(datetime.fromisoformat(date_str).timestamp())

            # Get candidate and query nodes
            cand_mask = times < timestamp
            cand_indices = torch.where(cand_mask)[0]

            if self.citation_mask is not None:
                cand_indices = cand_indices[
                    torch.isin(cand_indices, self.citation_mask)
                ]

            cand_emb = self.embeddings[cand_indices]

            # Get nodes at current time with outgoing edges
            nodes_at_time = (times == timestamp).nonzero(as_tuple=True)[0]
            nodes_with_edges = nodes_at_time[torch.isin(nodes_at_time, source_nodes)]
            num_nodes = nodes_with_edges.size(0)

            if num_nodes == 0:
                continue

            # Create subgraph and compute query embeddings
            input_nodes = (
                ("paragraph", nodes_with_edges) if self.is_hetero else nodes_with_edges
            )
            sub = next(iter(self._create_loader(input_nodes)))
            query_emb = self._process_subgraph(sub, num_nodes)

            # Compute similarities (language is already in embeddings via concatenation)
            sim = torch.matmul(query_emb, cand_emb.T)

            k = min(self.top_k, sim.size(1))
            _, sim_ord = torch.topk(sim, k=k, dim=1, largest=True, sorted=True)

            # Get node IDs
            sub_node_id_hash = (
                sub["paragraph"].node_id_hash if self.is_hetero else sub.node_id_hash
            )
            sub_n_id = sub["paragraph"].n_id if self.is_hetero else sub.n_id

            query_ids = [decode_celex(nid) for nid in sub_node_id_hash[:num_nodes]]
            ranked_ids = [
                [decode_celex(nid) for nid in row]
                for row in node_id_hash[cand_indices[sim_ord]]
            ]

            # Compute metrics
            grouped = group.groupby(["CELEX_FROM", "NUMBER_FROM"])
            for celex_from, number_from in query_ids:
                if (celex_from, number_from) not in grouped.groups:
                    continue

                relevant_rows = group.loc[grouped.groups[(celex_from, number_from)]]
                relevant_set = set(
                    zip(
                        relevant_rows["CELEX_TO"].astype(str),
                        relevant_rows["NUMBER_TO"].astype(int),
                    )
                )

                idx = query_ids.index((celex_from, number_from))
                ranked_array: NDArray = np.asarray(ranked_ids[idx], dtype=object)

                ap_scores.append(compute_ap(ranked_array, relevant_set))
                for k in k_values:
                    recall_scores[k].append(
                        compute_recall_at_k(
                            ranked_array, relevant_set, min(k, len(ranked_array))
                        )
                    )

            # Update embeddings for future queries
            self._update_embeddings(sub, sub_n_id)

        map_score = float(np.mean(ap_scores))
        print(f"MAP: {map_score}")
        for k, recalls in recall_scores.items():
            print(f"Recall@{k}: {float(np.mean(recalls))}")

        return map_score


if __name__ == "__main__":
    from models import DualEncoderGNN

    model = DualEncoderGNN(input_dim=384, output_dim=384, num_layers=1)
    model.load_state_dict(torch.load("checkpoints/homo_gnn/best_model.pt"))

    evaluator = SimpleIncrementalEvaluator(
        gnn_model=model,
        preprocessed_dir="data/preprocessed",
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,
        k_hops=1,
        device="cuda" if torch.cuda.is_available() else "cpu",
        top_k=1000,
        graph_type="homogeneous",
    )

    evaluator.run()

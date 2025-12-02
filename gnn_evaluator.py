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
    SemanticGraphBuilder,
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


class GNNEvaluator:
    """Incremental GNN evaluator with support for different graph types."""

    def __init__(
        self,
        gnn_model: nn.Module,
        graph_builder: "HomogeneousGraphBuilder | HeterogeneousGraphBuilder | SemanticGraphBuilder",
        par_to_par_path: str,
        train_cutoff_year: int = 2018,
        k_hops: int = 2,
        device: str | None = None,
        mode: EvaluatorMode = "citation_pairs",
        top_k: int = 10000,
    ):
        """
        Initialize the incremental evaluator.

        Args:
            gnn_model: The GNN model to evaluate
            graph_builder: Pre-configured graph builder instance
            par_to_par_path: Path to paragraph-to-paragraph citation CSV
            train_cutoff_year: Evaluate on data after this year
            k_hops: Number of hops for neighbor sampling
            device: Device to use (cuda/cpu)
            mode: Evaluation mode ("citation_pairs" or other)
            top_k: Number of top candidates to consider
        """
        self.graph_builder = graph_builder
        self.par_to_par_path = par_to_par_path
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.mode = mode
        self.top_k = top_k
        self.is_hetero = isinstance(graph_builder, HeterogeneousGraphBuilder)

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gnn_model = gnn_model.to(self.device)
        self.gnn_model.eval()

        print(f"Using device: {self.device}")
        print(f"Graph builder: {type(graph_builder).__name__}")
        print(f"Train cutoff year: {self.train_cutoff_year}")

        self.graph_data = self._build_graph()
        self.embeddings = self._compute_initial_embeddings()
        self.citation_mask = self._build_citation_mask()

    def _build_graph(self) -> Data | HeteroData:
        """Build the evaluation graph using the provided graph builder.

        Note: Graph stays on CPU for NeighborLoader compatibility.
        Batches are moved to GPU during processing.
        """
        # For evaluation, we include all data (no cutoff) since we evaluate incrementally
        graph = self.graph_builder.build_graph(train_cutoff_year=None)
        if self.is_hetero:
            return ToUndirected()(graph)
        return graph

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
            # Move to device for model inference
            modified = modified.to(self.device)
            return self.gnn_model(modified)["paragraph"].cpu()

    def _compute_homo_embeddings(self, cutoff_ts: float) -> torch.Tensor:
        """Compute embeddings for homogeneous graph (document embeddings for corpus).

        Filters edges by time to ensure only past edges are used for initial embeddings.
        """
        edge_index = self.graph_data.edge_index
        times = self.graph_data.time
        edge_attr = getattr(self.graph_data, "edge_attr", None)

        # Filter edges by time (only use edges where both nodes are before cutoff)
        mask = (times[edge_index[0]] < cutoff_ts) & (times[edge_index[1]] < cutoff_ts)
        filtered_edges = edge_index[:, mask].to(self.device)
        filtered_attr = (
            edge_attr[mask].to(self.device) if edge_attr is not None else None
        )
        language = getattr(self.graph_data, "language", None)
        node_type = getattr(self.graph_data, "node_type", None)

        # Move features to device
        x = self.graph_data.x.to(self.device)
        date_feature = getattr(self.graph_data, "date_feature", None)
        if date_feature is not None:
            date_feature = date_feature.to(self.device)
        if language is not None:
            language = language.to(self.device)
        if node_type is not None:
            node_type = node_type.to(self.device)
        subject_matter = getattr(self.graph_data, "subject_matter", None)
        if subject_matter is not None:
            subject_matter = subject_matter.to(self.device)
        keywords = getattr(self.graph_data, "keywords", None)
        if keywords is not None:
            keywords = keywords.to(self.device)
        case_law_about = getattr(self.graph_data, "case_law_about", None)
        if case_law_about is not None:
            case_law_about = case_law_about.to(self.device)

        with torch.no_grad():
            # Use document encoder for corpus embeddings if dual encoder
            if hasattr(self.gnn_model, "encode_document"):
                emb = self.gnn_model.encode_document(
                    x,
                    filtered_edges,
                    date_feature=date_feature,
                    edge_attr=filtered_attr,
                    language=language,
                    subject_matter=subject_matter,
                    keywords=keywords,
                    case_law_about=case_law_about,
                )
            else:
                emb = self.gnn_model(
                    x,
                    filtered_edges,
                    date_feature=date_feature,
                    edge_attr=filtered_attr,
                    language=language,
                    node_type=node_type,
                )
            return emb.cpu()

    def _get_graph_attrs(self) -> tuple:
        """Get graph attributes based on graph type."""
        if self.is_hetero:
            return (
                self.graph_data["paragraph"].time,
                self.graph_data["paragraph"].node_id_hash,
                self.graph_data["paragraph", "cites", "paragraph"].edge_index[0],
            )
        else:
            # Use citation_pairs if available (CaseLink-style), otherwise edge_index
            if (
                hasattr(self.graph_data, "citation_pairs")
                and self.graph_data.citation_pairs.size(1) > 0
            ):
                source_nodes = self.graph_data.citation_pairs[0]
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
        # Move subgraph to device
        sub = sub.to(self.device)

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
            return self.gnn_model(modified)["paragraph"][:num_nodes].cpu()

    def _process_homo_subgraph(self, sub: Data, num_nodes: int) -> torch.Tensor:
        """Process homogeneous subgraph using the graph builder's masking strategy."""
        # Move subgraph to device
        sub = sub.to(self.device)

        # Use query features for anchor nodes
        x = sub.x.clone()
        if hasattr(sub, "x_query"):
            x[:num_nodes] = sub.x_query[:num_nodes]

        with torch.no_grad():
            # Use query encoder for queries if dual encoder
            if hasattr(self.gnn_model, "encode_query"):
                date_feature = getattr(sub, "date_feature", None)
                language = getattr(sub, "language", None)
                subject_matter = getattr(sub, "subject_matter", None)
                keywords = getattr(sub, "keywords", None)
                case_law_about = getattr(sub, "case_law_about", None)
                emb = self.gnn_model.encode_query(
                    x[:num_nodes],
                    date_feature=(
                        date_feature[:num_nodes] if date_feature is not None else None
                    ),
                    language=language[:num_nodes] if language is not None else None,
                    subject_matter=(
                        subject_matter[:num_nodes]
                        if subject_matter is not None
                        else None
                    ),
                    keywords=keywords[:num_nodes] if keywords is not None else None,
                    case_law_about=(
                        case_law_about[:num_nodes]
                        if case_law_about is not None
                        else None
                    ),
                )
                return emb.cpu()

            # Fall back to full model with edge masking via graph builder
            edge_attr = getattr(sub, "edge_attr", None)
            language = getattr(sub, "language", None)
            node_type = getattr(sub, "node_type", None)

            # Use the graph builder's masking logic
            masked_edges, masked_attr = self.graph_builder.mask_edges_for_training(
                sub.edge_index, edge_attr, num_nodes
            )

            embeddings = self.gnn_model(
                x,
                masked_edges,
                date_feature=getattr(sub, "date_feature", None),
                edge_attr=masked_attr,
                language=language,
                node_type=node_type,
            )
            return embeddings[:num_nodes].cpu()

    def _update_embeddings(self, sub: Data | HeteroData, sub_n_id: torch.Tensor):
        """Update embeddings for nodes in the subgraph (document embeddings)."""
        loader = self._create_loader(
            ("paragraph", sub_n_id) if self.is_hetero else sub_n_id
        )
        expanded_sub = next(iter(loader))
        # Move subgraph to device
        expanded_sub = expanded_sub.to(self.device)

        with torch.no_grad():
            if self.is_hetero:
                embeddings = self.gnn_model(expanded_sub)["paragraph"]
            else:
                edge_attr = getattr(expanded_sub, "edge_attr", None)
                date_feature = getattr(expanded_sub, "date_feature", None)
                language = getattr(expanded_sub, "language", None)
                node_type = getattr(expanded_sub, "node_type", None)
                subject_matter = getattr(expanded_sub, "subject_matter", None)
                keywords = getattr(expanded_sub, "keywords", None)
                case_law_about = getattr(expanded_sub, "case_law_about", None)
                # Use document encoder if dual encoder
                if hasattr(self.gnn_model, "encode_document"):
                    embeddings = self.gnn_model.encode_document(
                        expanded_sub.x,
                        expanded_sub.edge_index,
                        date_feature=date_feature,
                        edge_attr=edge_attr,
                        language=language,
                        subject_matter=subject_matter,
                        keywords=keywords,
                        case_law_about=case_law_about,
                    )
                else:
                    embeddings = self.gnn_model(
                        expanded_sub.x,
                        expanded_sub.edge_index,
                        date_feature=date_feature,
                        edge_attr=edge_attr,
                        language=language,
                        node_type=node_type,
                    )

        self.embeddings[sub_n_id] = embeddings[: len(sub_n_id)].cpu()

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

    layers = 1

    model = DualEncoderGNN(
        input_dim=384,
        output_dim=384,
        num_layers=layers,
        fusion_mode="scalar",
    )
    model.load_state_dict(torch.load("checkpoints/homo_gnn/best_model.pt"))

    # Option 1: Citation-based graph (HomogeneousGraphBuilder)
    graph_builder = HomogeneousGraphBuilder(
        preprocessed_dir="data/preprocessed",
        include_only_citing=True,
    )

    # Option 2: Semantic similarity graph (SemanticGraphBuilder / CaseLink-style)
    # graph_builder = SemanticGraphBuilder(
    #     preprocessed_dir="data/preprocessed",
    #     judgments_path="data/judgments_cleaned.json",
    #     semantic_threshold=0.3,
    #     include_article_nodes=True,
    # )

    evaluator = GNNEvaluator(
        gnn_model=model,
        graph_builder=graph_builder,
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,  # Evaluate on data after this year
        k_hops=layers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        top_k=1000,
    )

    evaluator.run()

import json
import os
import csv
from datetime import datetime as dt
from collections import defaultdict
from typing import Literal, Any

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import k_hop_subgraph, to_undirected  # type: ignore
from tqdm import tqdm  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from evaluator import compute_ap_fast, compute_recall_at_k_fast
from preprocessing.graph_builder import HomogeneousGraphBuilder

# Type alias for evaluator modes
EvaluatorMode = Literal["citation_pairs", "all_paragraphs"]


class IncrementalGNNEvaluator:
    """
    Incremental GNN evaluation that prevents data leakage by:
    1. Starting with all training paragraphs embedded
    2. Processing test paragraphs chronologically
    3. For each test paragraph:
       - Query against all paragraphs that came before it
       - Evaluate retrieval performance
       - Add it to the graph
       - Re-embed its k-egonet to keep embeddings current
    """

    def __init__(
        self,
        gnn_model: nn.Module,
        preprocessed_dir: str,
        queries_path: str,
        qrel_path: str,
        text_encoder_name: str,
        mode: EvaluatorMode = "citation_pairs",
        train_cutoff_year: int = 2018,
        k_hops: int = 2,
        top_k: int | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ):
        self.gnn_model = gnn_model
        self.preprocessed_dir = preprocessed_dir
        self.queries_path = queries_path
        self.qrel_path = qrel_path
        self.text_encoder_name = text_encoder_name
        self.mode: EvaluatorMode = mode
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.top_k = top_k
        self.normalize_embeddings = normalize_embeddings

        # Validate mode
        if mode not in ["citation_pairs", "all_paragraphs"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'citation_pairs' or 'all_paragraphs'"
            )

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gnn_model = self.gnn_model.to(self.device)
        self.gnn_model.eval()

        # Initialize text encoder for encoding queries at evaluation time
        print(f"Loading text encoder for queries: {text_encoder_name}")
        self.text_encoder = SentenceTransformer(text_encoder_name)

        # Data structures (populated by load_and_prepare)
        self.pid_to_text: NDArray[np.object_] | None = None
        self.celex_number_to_pid: dict[tuple[str, int], int] | None = None
        self.paragraph_dates: NDArray | None = None
        self.paragraph_celex: NDArray[np.object_] | None = None
        self.paragraph_number: NDArray[np.object_] | None = None
        self.paragraph_set: NDArray[np.object_] | None = None

        self.query_pids: list[int] | None = None
        self.query_texts: NDArray[np.object_] | None = None
        self.qrel: dict[int, list[int]] | None = None

        # Graph state (evolves during evaluation)
        self.graph_data: Data | None = None
        self.current_embeddings: torch.Tensor | None = None
        self.text_embeddings: torch.Tensor | None = None  # Raw text embeddings
        self.nodes_in_graph: set[int] = set()
        self.pid_to_node_idx: dict[int, int] = (
            {}
        )  # Maps evaluator PID to graph node index
        self.node_idx_to_pid: dict[int, int] = (
            {}
        )  # Maps graph node index to evaluator PID

        # Evaluation metrics
        self.query_results: list[dict] = []

    def load_and_prepare(self) -> None:
        """Load all data and prepare for evaluation."""
        self._load_from_graph_builder()

    def _load_from_graph_builder(self) -> None:
        """Load data using HomogeneousGraphBuilder from preprocessed data."""
        if not self.preprocessed_dir:
            raise ValueError("preprocessed_dir must be set to use graph builder")

        print(f"\nLoading data from preprocessed directory: {self.preprocessed_dir}")
        builder = HomogeneousGraphBuilder(self.preprocessed_dir)

        # Build mappings from metadata
        self.pid_to_text = np.array(
            [meta.get("text", "") for meta in builder.par_metadata], dtype=object
        )
        self.celex_number_to_pid = {
            (meta["celex"], meta["paragraph_number"]): i
            for i, meta in enumerate(builder.par_metadata)
        }

        def extract_year_from_date(meta: dict) -> int | None:
            """Extract year from date field in metadata."""
            date_str = meta.get("date")
            if not date_str or (isinstance(date_str, str) and not date_str.strip()):
                return None
            try:
                # Date might be ISO format string (YYYY-MM-DD) or datetime object
                if isinstance(date_str, str):
                    date_obj = dt.strptime(date_str, "%Y-%m-%d")
                    return date_obj.year
                elif hasattr(date_str, "year"):
                    return date_str.year
                return None
            except (ValueError, AttributeError):
                return None

        self.paragraph_dates = np.array(
            [
                (
                    np.datetime64(meta["date"])
                    if meta.get("date")
                    else np.datetime64("NaT")
                )
                for meta in builder.par_metadata
            ],
            dtype="datetime64[ns]",
        )
        self.paragraph_celex = np.array(
            [meta["celex"] for meta in builder.par_metadata], dtype=object
        )
        self.paragraph_number = np.array(
            [meta["paragraph_number"] for meta in builder.par_metadata], dtype=object
        )
        self.paragraph_set = np.array(
            [
                (
                    "train"
                    if (year := extract_year_from_date(meta)) is not None
                    and year < self.train_cutoff_year
                    else "test"
                )
                for meta in builder.par_metadata
            ],
            dtype=object,
        )

        # Load queries and qrel (still need these from files)
        self._load_queries_and_qrel()

        # Filter paragraphs for citation_pairs mode
        if self.mode == "citation_pairs":
            self._filter_to_citation_paragraphs()

    def _load_queries_and_qrel(self) -> None:
        """Load queries and qrel files."""
        assert self.celex_number_to_pid is not None

        print("Loading queries...")
        query_data = []
        with open(self.queries_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # Skip header
            for celex, par_num, query_text in reader:
                key = (celex, int(par_num))
                if key in self.celex_number_to_pid:
                    query_data.append((self.celex_number_to_pid[key], query_text))

        self.query_pids = [pid for pid, _ in query_data]
        self.query_texts = np.array([text for _, text in query_data], dtype=object)

        print("Loading qrel...")
        self.qrel = defaultdict(list)
        with open(self.qrel_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                query_id = parts[0]
                doc_id = parts[2]

                # Parse celex_paragraph_number format
                celex_q, par_num_q = query_id.rsplit("_", 1)
                celex_d, par_num_d = doc_id.rsplit("_", 1)

                query_key = (celex_q, int(par_num_q))
                doc_key = (celex_d, int(par_num_d))

                if (
                    query_key in self.celex_number_to_pid
                    and doc_key in self.celex_number_to_pid
                ):
                    query_pid = self.celex_number_to_pid[query_key]
                    doc_pid = self.celex_number_to_pid[doc_key]
                    self.qrel[query_pid].append(doc_pid)

        print(
            f"Loaded {len(self.query_pids)} queries with {sum(len(v) for v in self.qrel.values())} qrel entries"
        )

    def _filter_to_citation_paragraphs(self) -> None:
        """Filter paragraphs to only those involved in citations for citation_pairs mode."""
        assert self.qrel is not None
        assert self.pid_to_text is not None
        assert self.celex_number_to_pid is not None
        assert self.paragraph_dates is not None
        assert self.paragraph_celex is not None
        assert self.paragraph_number is not None
        assert self.paragraph_set is not None
        assert self.query_pids is not None
        assert self.query_texts is not None

        print("Filtering to citation-involved paragraphs...")

        # Collect all paragraphs involved in any citation
        citation_involved_pids: set[int] = set()
        for query_pid, cited_pids in self.qrel.items():
            citation_involved_pids.add(query_pid)
            citation_involved_pids.update(cited_pids)

        # Create sorted list of citation-involved pids
        old_pids = sorted(citation_involved_pids)

        # Create mapping from old pid to new pid
        old_to_new_pid = {old_pid: new_pid for new_pid, old_pid in enumerate(old_pids)}

        # Filter all arrays to only citation-involved paragraphs
        self.pid_to_text = self.pid_to_text[old_pids]
        self.paragraph_dates = self.paragraph_dates[old_pids]
        self.paragraph_celex = self.paragraph_celex[old_pids]
        self.paragraph_number = self.paragraph_number[old_pids]
        self.paragraph_set = self.paragraph_set[old_pids]

        # Update celex_number_to_pid mapping
        new_celex_number_to_pid = {}
        for (celex, number), old_pid in self.celex_number_to_pid.items():
            if old_pid in old_to_new_pid:
                new_celex_number_to_pid[(celex, number)] = old_to_new_pid[old_pid]
        self.celex_number_to_pid = new_celex_number_to_pid

        # Update query_pids and query_texts
        new_query_pids = []
        new_query_texts = []
        old_query_texts = self.query_texts
        for idx, old_pid in enumerate(self.query_pids):
            if old_pid in old_to_new_pid:
                new_query_pids.append(old_to_new_pid[old_pid])
                new_query_texts.append(old_query_texts[idx])
        self.query_pids = new_query_pids
        self.query_texts = np.array(new_query_texts, dtype=object)

        # Update qrel
        new_qrel: dict[int, list[int]] = {}
        for old_query_pid, old_cited_pids in self.qrel.items():
            if old_query_pid in old_to_new_pid:
                new_query_pid = old_to_new_pid[old_query_pid]
                new_cited_pids = [
                    old_to_new_pid[old_cited_pid]
                    for old_cited_pid in old_cited_pids
                    if old_cited_pid in old_to_new_pid
                ]
                if new_cited_pids:
                    new_qrel[new_query_pid] = new_cited_pids
        self.qrel = new_qrel

        print(
            f"Filtered from {len(citation_involved_pids)} to {len(old_pids)} citation-involved paragraphs"
        )

    def initialize_graph_with_training_data(self) -> None:
        """
        Initialize graph with ALL nodes (train + test) but only training edges.
        This allows us to use PIDs directly as graph node indices.
        """
        if not self.preprocessed_dir:
            raise ValueError("preprocessed_dir must be set to use graph builder")

        print("\nInitializing graph with all nodes and training edges...")
        builder = HomogeneousGraphBuilder(self.preprocessed_dir)

        # Build graph with training edges to get the structure
        training_graph = builder.build_graph(
            train_cutoff_year=self.train_cutoff_year,
            include_only_citing=(self.mode == "citation_pairs"),
        )

        # Get text embeddings for ALL citation-involved paragraphs from builder
        # We need embeddings for all paragraphs that evaluator has after filtering
        assert self.celex_number_to_pid is not None
        assert self.paragraph_set is not None
        assert self.pid_to_text is not None

        num_pids = len(self.pid_to_text)
        embedding_dim = training_graph.x.shape[1]

        # Load text embeddings for all paragraphs from builder
        # Create embeddings array indexed by evaluator PIDs
        all_embeddings = np.zeros((num_pids, embedding_dim), dtype=np.float32)

        # Map builder's paragraph indices to evaluator PIDs and copy embeddings
        for builder_idx, meta in enumerate(builder.par_metadata):
            celex = meta["celex"]
            par_num = meta["paragraph_number"]
            key = (celex, par_num)
            if key in self.celex_number_to_pid:
                eval_pid = self.celex_number_to_pid[key]
                all_embeddings[eval_pid] = builder.par_embeddings[builder_idx]

        self.text_embeddings = torch.tensor(all_embeddings, dtype=torch.float32)

        # Create graph with all nodes but only training edges
        # Get training edge indices and map them to evaluator PIDs
        train_edges = []
        training_pids = set(
            int(pid) for pid in np.where(self.paragraph_set == "train")[0]
        )

        # Build edges from qrel for training paragraphs only
        assert self.qrel is not None
        for src_pid in training_pids:
            if src_pid in self.qrel:
                for tgt_pid in self.qrel[src_pid]:
                    if tgt_pid in training_pids:
                        # Bidirectional edges
                        train_edges.append([src_pid, tgt_pid])
                        train_edges.append([tgt_pid, src_pid])

        # Create graph data
        if train_edges:
            edge_index = torch.tensor(train_edges, dtype=torch.long).t().contiguous()
            # Remove duplicate edges
            edge_index = torch.unique(edge_index, dim=1)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        self.graph_data = Data(
            x=self.text_embeddings,
            edge_index=edge_index,
            num_nodes=num_pids,
        )

        # Track which PIDs have edges (training paragraphs)
        self.nodes_in_graph = training_pids

        # PIDs == graph node indices, so mappings are identity
        self.pid_to_node_idx = {pid: pid for pid in range(num_pids)}
        self.node_idx_to_pid = {pid: pid for pid in range(num_pids)}

        print(
            f"Initialized graph: {num_pids} nodes total, "
            f"{len(training_pids)} training nodes, "
            f"{edge_index.shape[1]} edges"
        )

        # Compute initial GNN embeddings
        self._compute_gnn_embeddings()

    def _compute_gnn_embeddings(self) -> None:
        """Compute GNN embeddings for all nodes currently in the graph."""
        assert self.graph_data is not None

        self.gnn_model.eval()
        with torch.no_grad():
            x = self.graph_data.x.to(self.device)
            edge_index = self.graph_data.edge_index.to(self.device)

            # Validate tensor shape
            if x.dim() != 2:
                raise ValueError(
                    f"Expected 2D tensor for x, got shape {x.shape}. "
                    f"graph_data.x shape: {self.graph_data.x.shape}"
                )

            embeddings = self.gnn_model(x, edge_index)

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            # Since PIDs == graph node indices, embeddings are already PID-indexed
            self.current_embeddings = embeddings.cpu()

    def _update_k_egonet(self, node_id: int) -> None:
        """
        Re-compute embeddings for k-hop neighborhood of a node.
        This ensures that when we add a new node, its neighbors get updated embeddings.
        """
        assert self.graph_data is not None
        assert self.current_embeddings is not None

        # Find k-hop subgraph around the node
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx=node_id,
            num_hops=self.k_hops,
            edge_index=self.graph_data.edge_index,
            num_nodes=self.graph_data.num_nodes,
            relabel_nodes=True,  # Ensure edges are remapped to local indices
        )

        if len(subset) == 0:
            return

        # Get subgraph features
        sub_x = self.graph_data.x[subset]

        # Handle single node case - subset might select single node
        if sub_x.dim() == 1:
            sub_x = sub_x.unsqueeze(0)  # [feature_dim] -> [1, feature_dim]

        # Validate edge indices are within bounds
        if sub_edge_index.numel() > 0:
            max_edge_idx = sub_edge_index.max().item()
            if max_edge_idx >= len(subset):
                # Edge indices not properly remapped, skip update for safety
                print(
                    f"Warning: Edge index out of bounds for node {node_id}, skipping k-hop update"
                )
                return

        # Compute embeddings for subgraph
        self.gnn_model.eval()
        with torch.no_grad():
            sub_x = sub_x.to(self.device)
            sub_edge_index = sub_edge_index.to(self.device)
            sub_embeddings = self.gnn_model(sub_x, sub_edge_index)

            if self.normalize_embeddings:
                sub_embeddings = F.normalize(sub_embeddings, p=2, dim=1)

            # Update embeddings for nodes in subgraph
            self.current_embeddings[subset] = sub_embeddings.cpu()

    def _add_node_to_graph(self, node_id: int) -> None:
        """
        Add a node to the graph and update its k-hop neighborhood embeddings.
        """
        assert self.graph_data is not None
        assert self.qrel is not None

        if node_id in self.nodes_in_graph:
            return  # Already in graph

        # Add edges for this node (citations from/to nodes already in graph)
        new_edges = []

        # Add citations FROM this node to older nodes (node_id cites older nodes)
        if node_id in self.qrel:
            for tgt_pid in self.qrel[node_id]:
                if tgt_pid in self.nodes_in_graph:
                    # Citation edge: tgt -> node_id (tgt is cited by node_id)
                    new_edges.append([tgt_pid, node_id])

        # Add citations TO this node from older nodes (older nodes cite node_id)
        for src_pid in self.nodes_in_graph:
            if src_pid in self.qrel and node_id in self.qrel[src_pid]:
                # Citation edge: node_id -> src_pid (node_id is cited by src_pid)
                new_edges.append([node_id, src_pid])

        if new_edges:
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).t().contiguous()
            # Make undirected
            new_edge_index = to_undirected(
                new_edge_index, num_nodes=self.graph_data.num_nodes
            )
            # Append to existing edges
            self.graph_data.edge_index = torch.cat(
                [self.graph_data.edge_index, new_edge_index], dim=1
            )

        # Mark node as in graph
        self.nodes_in_graph.add(node_id)

        # Update k-hop neighborhood embeddings
        self._update_k_egonet(node_id)

    def _retrieve_and_evaluate(
        self, query_pid: int, candidate_pids: NDArray, k_values: list[int]
    ) -> dict | None:
        """
        Retrieve candidates for a query and compute metrics.
        Encodes query using its GNN embedding with masked citation edges (similar to trainer).
        """
        assert self.current_embeddings is not None
        assert self.graph_data is not None
        assert self.qrel is not None

        # Get query embedding from graph with masked citation edges
        # Mask edges FROM query_pid to its citations to prevent information leakage
        with torch.no_grad():
            edge_index = self.graph_data.edge_index

            # Mask edges where source is query_pid and destination is a cited paragraph
            if query_pid in self.qrel:
                cited_pids = set(self.qrel[query_pid])
                src, dst = edge_index
                # Keep edges where: source is NOT query OR destination is NOT cited
                edge_mask = (src != query_pid) | ~torch.tensor(
                    [dst_node.item() in cited_pids for dst_node in dst],
                    dtype=torch.bool,
                    device=edge_index.device,
                )
                masked_edge_index = edge_index[:, edge_mask]
            else:
                # No citations to mask
                masked_edge_index = edge_index

            # Get GNN embedding for query with masked edges
            x = self.graph_data.x.to(self.device)
            masked_edge_index = masked_edge_index.to(self.device)

            embeddings = self.gnn_model(x, masked_edge_index)

            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)

            query_emb = embeddings[query_pid].cpu().numpy()

        # Get candidate embeddings
        candidate_embs = self.current_embeddings[candidate_pids].numpy()

        # Compute similarities
        similarities = candidate_embs @ query_emb

        # Rank candidates
        if self.top_k is not None and self.top_k < len(similarities):
            top_k_indices = np.argpartition(-similarities, self.top_k)[: self.top_k]
            sorted_top_k = top_k_indices[np.argsort(-similarities[top_k_indices])]
            ranked_pids = candidate_pids[sorted_top_k]
        else:
            ranked_order = np.argsort(-similarities)
            ranked_pids = candidate_pids[ranked_order]

        # Get ground truth relevant documents from qrel
        relevant_pids = np.array(self.qrel.get(query_pid, []), dtype=np.int64)

        # Filter to only candidates that are actually relevant
        relevant_mask = np.isin(relevant_pids, candidate_pids)
        relevant_pids = relevant_pids[relevant_mask]

        if len(relevant_pids) == 0:
            return None  # No relevant docs in candidate pool

        # Compute metrics
        max_rank = (
            len(ranked_pids)
            if self.top_k is None
            else min(len(ranked_pids), self.top_k)
        )
        ap = compute_ap_fast(ranked_pids, relevant_pids, max_rank)

        recall_scores = {}
        for k in k_values:
            recall_scores[k] = compute_recall_at_k_fast(ranked_pids, relevant_pids, k)

        return {
            "query_pid": query_pid,
            "num_candidates": len(candidate_pids),
            "num_relevant": len(relevant_pids),
            "ap": ap,
            "recall": recall_scores,
        }

    def evaluate_incremental(
        self, k_values: list[int] = [5, 10, 100]
    ) -> dict[str, float]:
        """
        Run incremental evaluation on test paragraphs in chronological order.
        Only evaluates on paragraphs that are in query_pids (have queries).
        All paragraphs (including query ones) are added to the graph.
        """
        assert self.paragraph_set is not None
        assert self.paragraph_dates is not None
        assert self.query_pids is not None
        assert self.qrel is not None

        # Get test paragraphs sorted by date (already sorted in load_and_prepare)
        test_mask = self.paragraph_set == "test"
        test_pids = np.where(test_mask)[0]

        # Get test query pids (subset of test paragraphs that have queries)
        test_query_pids_set = set(
            pid for pid in self.query_pids if self.paragraph_set[pid] == "test"
        )

        print(
            f"\nEvaluating on {len(test_pids)} test paragraphs "
            f"({len(test_query_pids_set)} with queries) incrementally..."
        )

        # Track metrics
        all_ap_scores: list[float] = []
        all_recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for test_pid in tqdm(test_query_pids_set, desc="Incremental evaluation"):
            test_date = self.paragraph_dates[test_pid]

            # Candidate pool: all paragraphs strictly before this one (already in graph)
            candidate_pids = np.array(
                [
                    pid
                    for pid in self.nodes_in_graph
                    if self.paragraph_dates[pid] < test_date
                ],
                dtype=np.int64,
            )

            # Retrieve and evaluate
            result = self._retrieve_and_evaluate(
                int(test_pid), candidate_pids, k_values
            )

            if result is not None:
                all_ap_scores.append(result["ap"])
                for k in k_values:
                    all_recall_scores[k].append(result["recall"][k])
                self.query_results.append(result)

            # Add node to graph for future queries (all test paragraphs are added)
            self._add_node_to_graph(int(test_pid))

        # Compute final metrics
        metrics = {}
        if all_ap_scores:
            map_score = float(np.mean(all_ap_scores))
            metrics["map"] = map_score
            if self.top_k:
                metrics[f"map@{self.top_k}"] = map_score
        else:
            metrics["map"] = 0.0

        for k in k_values:
            if all_recall_scores[k]:
                metrics[f"recall@{k}"] = float(np.mean(all_recall_scores[k]))
            else:
                metrics[f"recall@{k}"] = 0.0

        return metrics

    def run(self) -> dict[str, float]:
        """Run full incremental evaluation pipeline."""
        print(f"Mode: {self.mode}")
        print("Loading and preparing data...")
        self.load_and_prepare()

        assert self.pid_to_text is not None
        assert self.paragraph_set is not None
        assert self.query_pids is not None

        print(f"Unique paragraphs: {len(self.pid_to_text)}")
        print(f"Train paragraphs: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test paragraphs: {np.sum(self.paragraph_set == 'test')}")
        print(f"Total queries: {len(self.query_pids)}")

        self.initialize_graph_with_training_data()
        metrics = self.evaluate_incremental()

        print("\n" + "=" * 80)
        print("Incremental Evaluation Results")
        print("=" * 80)

        metric_name = f"MAP@{self.top_k}" if self.top_k else "MAP"
        map_score = metrics.get(f"map@{self.top_k}" if self.top_k else "map", 0.0)
        print(f"\n{metric_name}: {map_score:.3f}")

        for k in [5, 10, 100]:
            if f"recall@{k}" in metrics:
                print(f"Recall@{k}: {metrics[f'recall@{k}']:.3f}")

        return metrics


if __name__ == "__main__":
    from example_gnn_usage import CitationGNN
    import torch
    from sentence_transformers import SentenceTransformer

    # Load trained GNN model
    encoding_model = "checkpoints/simcse_citation_model"
    text_encoder = SentenceTransformer(encoding_model)
    in_channels = text_encoder.get_sentence_embedding_dimension()

    model = CitationGNN(
        in_channels, hidden_dim=512, output_dim=in_channels, num_layers=2
    )
    model.load_state_dict(torch.load("checkpoints/gnn/best_model.pt"))

    # Run incremental evaluation
    evaluator = IncrementalGNNEvaluator(
        gnn_model=model,
        preprocessed_dir="data/preprocessed",
        queries_path="data/evaluation/queries_cleaned_masked.tsv",
        qrel_path="data/evaluation/qrel.txt",
        text_encoder_name=encoding_model,
        mode="citation_pairs",
        train_cutoff_year=2018,
        k_hops=2,
        top_k=10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = evaluator.run()

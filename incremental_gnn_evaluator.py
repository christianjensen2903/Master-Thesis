import json
import os
import re
from datetime import datetime as dt
from collections import defaultdict
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from torch_geometric.utils import k_hop_subgraph, to_undirected  # type: ignore
from tqdm import tqdm  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore

from evaluator import compute_ap_fast, compute_recall_at_k_fast

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
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        mode: EvaluatorMode = "all_paragraphs",
        csv_path: str = "data/par-to-par-cleaned.csv",
        metadata_path: str = "data/par-to-par.json",
        judgments_path: str = "data/judgments_cleaned.json",
        train_cutoff_year: int = 2018,
        k_hops: int = 2,
        top_k: int | None = None,
        device: str | None = None,
        normalize_embeddings: bool = True,
        embeddings_cache_dir: str | None = None,
    ):
        self.gnn_model = gnn_model
        self.text_encoder_name = text_encoder_name
        self.mode: EvaluatorMode = mode
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.judgments_path = judgments_path
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.top_k = top_k
        self.normalize_embeddings = normalize_embeddings
        self.embeddings_cache_dir = embeddings_cache_dir

        # Validate mode
        if mode not in ["citation_pairs", "all_paragraphs"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'citation_pairs' or 'all_paragraphs'"
            )

        # Create cache directory if specified
        if self.embeddings_cache_dir:
            os.makedirs(self.embeddings_cache_dir, exist_ok=True)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gnn_model = self.gnn_model.to(self.device)
        self.gnn_model.eval()

        # Initialize text encoder
        print(f"Loading text encoder: {text_encoder_name}")
        self.text_encoder = SentenceTransformer(text_encoder_name)

        # Data structures
        self.paragraphs: list[dict] = []
        self.pid_to_text: NDArray[np.object_] | None = None
        self.celex_number_to_pid: dict[tuple[str, int], int] = {}
        self.paragraph_dates: NDArray | None = None
        self.paragraph_set: NDArray[np.object_] | None = None

        # Citation graph (ground truth)
        self.cited_by_pid: dict[int, list[int]] = {}

        # Graph state (evolves during evaluation)
        self.graph_data: Data | None = None
        self.current_embeddings: torch.Tensor | None = None
        self.text_embeddings: torch.Tensor | None = None  # Raw text embeddings
        self.nodes_in_graph: set[int] = set()

        # Evaluation metrics
        self.query_results: list[dict] = []

    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name to be a valid filename by removing path separators."""
        basename = os.path.basename(model_name)
        return basename.replace("/", "_").replace("\\", "_")

    def _get_embeddings_cache_key(self) -> str:
        """Generate cache key based on mode and model."""
        model_name = self._sanitize_model_name(self.text_encoder_name)
        return f"{model_name}_{self.mode}"

    def _load_cached_embeddings(self) -> np.ndarray | None:
        """Load embeddings from cache if they exist."""
        if not self.embeddings_cache_dir:
            return None

        cache_key = self._get_embeddings_cache_key()

        cache_path = os.path.join(self.embeddings_cache_dir, f"{cache_key}.npy")

        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            try:
                embeddings = np.load(cache_path)
                return embeddings
            except Exception as e:
                print(f"Failed to load cached embeddings: {e}")
                return None
        return None

    def _save_cached_embeddings(self, embeddings: np.ndarray) -> None:
        """Save embeddings to cache."""
        if not self.embeddings_cache_dir:
            return

        cache_key = self._get_embeddings_cache_key()
        cache_path = os.path.join(self.embeddings_cache_dir, f"{cache_key}.npy")

        print(f"Saving embeddings to cache: {cache_path}")
        try:
            np.save(cache_path, embeddings)
        except Exception as e:
            print(f"Failed to save embeddings to cache: {e}")

    def load_and_prepare(self) -> None:
        """Load paragraphs and citation data based on mode."""
        if self.mode == "citation_pairs":
            self._load_citation_pairs_mode()
        else:
            self._load_all_paragraphs_mode()

    def _load_citation_pairs_mode(self) -> None:
        """Load data in citation pairs mode (using par-to-par CSV only)"""
        from data_loader import (
            load_citation_data,
            split_train_test,
            build_paragraph_index,
        )

        print("Loading citation pairs data...")
        df = pd.read_csv(self.csv_path).dropna()

        with open(self.metadata_path) as f:
            metadata = json.load(f)

        train_meta, test_meta = split_train_test(metadata, self.train_cutoff_year)

        (
            pid_to_text,
            celex_number_to_pid,
            paragraph_dates,
            paragraph_celex,
            paragraph_number,
            paragraph_set,
        ) = build_paragraph_index(df, train_meta, test_meta)

        # Build paragraphs list for consistency
        paragraphs = []
        for pid in range(len(pid_to_text)):
            celex = paragraph_celex[pid]
            number = paragraph_number[pid]
            paragraphs.append(
                {
                    "text": pid_to_text[pid],
                    "celex": celex,
                    "date": paragraph_dates[pid].astype("O"),
                    "number": int(number),
                    "set_type": paragraph_set[pid],
                }
            )

        # Sort by date for chronological processing
        paragraphs.sort(key=lambda p: (p["date"], p["celex"], p["number"]))

        # Rebuild arrays in sorted order
        self.paragraphs = paragraphs
        self.pid_to_text = np.array([p["text"] for p in paragraphs], dtype=object)
        self.celex_number_to_pid = {
            (p["celex"], p["number"]): pid for pid, p in enumerate(paragraphs)
        }
        self.paragraph_dates = np.array(
            [p["date"] for p in paragraphs], dtype="datetime64[ns]"
        )
        self.paragraph_set = np.array([p["set_type"] for p in paragraphs], dtype=object)

        print(f"Total paragraphs: {len(self.pid_to_text)}")
        print(f"Train: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test: {np.sum(self.paragraph_set == 'test')}")

        # Build citation graph
        self._build_citation_graph(df)

    def _load_all_paragraphs_mode(self) -> None:
        """Load data in all paragraphs mode (using judgments_cleaned.json)"""
        print("Loading judgments...")
        with open(self.judgments_path) as f:
            judgments = json.load(f)

        # Build paragraph index from all judgments
        paragraphs = []
        for celex, judgment in tqdm(judgments.items(), desc="Processing judgments"):

            # Get date from meta (nested structure)
            meta = judgment.get("meta", {}).get("meta", {})
            date_str = meta.get("date")

            try:
                date = dt.strptime(date_str, "%Y-%m-%d")
            except:
                continue

            year = date.year
            set_type = "train" if year < self.train_cutoff_year else "test"

            for par_num, text in judgment["paragraphs"].items():
                paragraphs.append(
                    {
                        "text": text,
                        "celex": celex,
                        "date": date,
                        "number": int(par_num),
                        "set_type": set_type,
                    }
                )

        # Sort paragraphs by date to enable chronological processing
        paragraphs.sort(key=lambda p: (p["date"], p["celex"], p["number"]))

        # Build arrays
        self.paragraphs = paragraphs
        self.pid_to_text = np.array([p["text"] for p in paragraphs], dtype=object)
        self.celex_number_to_pid = {
            (p["celex"], p["number"]): pid for pid, p in enumerate(paragraphs)
        }
        self.paragraph_dates = np.array(
            [p["date"] for p in paragraphs], dtype="datetime64[ns]"
        )
        self.paragraph_set = np.array([p["set_type"] for p in paragraphs], dtype=object)

        print(f"Total paragraphs: {len(self.pid_to_text)}")
        print(f"Train: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test: {np.sum(self.paragraph_set == 'test')}")

        # Load citation pairs for ground truth
        print("Loading citation pairs for ground truth...")
        df = pd.read_csv(self.csv_path).dropna()
        self._build_citation_graph(df)

    def _build_citation_graph(self, df: pd.DataFrame) -> None:
        """Build citation graph from par-to-par CSV."""
        cited_by_pid = defaultdict(set)
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building citations"):
            celex_from = row["CELEX_FROM"]
            number_from = row["NUMBER_FROM"]
            celex_to = row["CELEX_TO"]
            number_to = row["NUMBER_TO"]

            src_key = (str(celex_from), int(number_from))
            tgt_key = (str(celex_to), int(number_to))

            if (
                src_key not in self.celex_number_to_pid
                or tgt_key not in self.celex_number_to_pid
            ):
                skipped += 1
                continue

            src_pid = self.celex_number_to_pid[src_key]
            tgt_pid = self.celex_number_to_pid[tgt_key]
            cited_by_pid[src_pid].add(tgt_pid)

        if skipped > 0:
            print(
                f"Skipped {skipped}/{len(df)} citation pairs (paragraphs not in index)"
            )

        # Convert to sorted lists for determinism
        self.cited_by_pid = {k: sorted(v) for k, v in cited_by_pid.items()}

    def initialize_graph_with_training_data(self) -> None:
        """Initialize graph with all training paragraphs."""
        assert self.pid_to_text is not None
        assert self.paragraph_set is not None

        print("\nInitializing graph with training data...")

        # Get training paragraph IDs
        train_mask = self.paragraph_set == "train"
        train_pids = np.where(train_mask)[0]

        # Try to load cached embeddings
        cached_embeddings = self._load_cached_embeddings()

        if cached_embeddings is not None and len(cached_embeddings) == len(
            self.pid_to_text
        ):
            print(f"Using cached embeddings (shape: {cached_embeddings.shape})")
            all_text_embeddings = cached_embeddings
        else:
            if cached_embeddings is not None:
                print(
                    f"Cache size mismatch: {len(cached_embeddings)} vs {len(self.pid_to_text)}"
                )

            # Encode all texts first (including test, so we can add them incrementally)
            print("Encoding all paragraph texts...")
            all_text_embeddings = self.text_encoder.encode(
                self.pid_to_text.tolist(),
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            print("Encoded all paragraph texts")

            # Save to cache
            self._save_cached_embeddings(all_text_embeddings)

        self.text_embeddings = torch.tensor(all_text_embeddings, dtype=torch.float32)

        # Build initial graph with training data only
        train_edge_list = []
        for src_pid in train_pids:
            if src_pid in self.cited_by_pid:
                for tgt_pid in self.cited_by_pid[src_pid]:
                    # Only include edges where both nodes are in training set
                    if tgt_pid in train_pids:
                        # Edge direction: tgt -> src (citation direction)
                        train_edge_list.append([tgt_pid, src_pid])

        if train_edge_list:
            edge_index = (
                torch.tensor(train_edge_list, dtype=torch.long).t().contiguous()
            )
            # Make undirected for GNN message passing
            edge_index = to_undirected(edge_index, num_nodes=len(self.pid_to_text))
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        print(f"Initial graph: {len(train_pids)} nodes, {edge_index.shape[1]} edges")

        # Create graph data
        self.graph_data = Data(
            x=self.text_embeddings,
            edge_index=edge_index,
            num_nodes=len(self.pid_to_text),
        )

        # Mark training nodes as being in graph
        self.nodes_in_graph = set(train_pids.tolist())

        # Compute initial embeddings
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

        if node_id in self.nodes_in_graph:
            return  # Already in graph

        # Add edges for this node (citations from/to nodes already in graph)
        new_edges = []

        # Add citations FROM this node to older nodes
        if node_id in self.cited_by_pid:
            for tgt_pid in self.cited_by_pid[node_id]:
                if tgt_pid in self.nodes_in_graph:
                    # Citation edge: tgt -> src
                    new_edges.append([tgt_pid, node_id])

        # Add citations TO this node from older nodes
        for src_pid in self.nodes_in_graph:
            if src_pid in self.cited_by_pid and node_id in self.cited_by_pid[src_pid]:
                # Citation edge: node_id -> src_pid (src_pid cites node_id)
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
        """
        assert self.current_embeddings is not None
        assert self.pid_to_text is not None

        # Get query embedding (use query encoder)
        query_text = self.pid_to_text[query_pid]
        with torch.no_grad():
            query_text_emb = self.text_encoder.encode(
                [query_text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            query_tensor = torch.tensor(  # type: ignore[call-overload]
                query_text_emb, dtype=torch.float32, device=self.device
            )
            query_emb = self.gnn_model.encode_query(query_tensor)  # type: ignore[misc]

            if self.normalize_embeddings:
                query_emb = F.normalize(query_emb, p=2, dim=1)

            query_emb = query_emb.cpu().numpy()[0]

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

        # Get ground truth relevant documents
        relevant_pids = np.array(self.cited_by_pid.get(query_pid, []), dtype=np.int64)

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
        """
        assert self.paragraph_set is not None
        assert self.paragraph_dates is not None

        # Get test paragraphs sorted by date (already sorted in load_and_prepare)
        test_mask = self.paragraph_set == "test"
        test_pids = np.where(test_mask)[0]

        print(f"\nEvaluating on {len(test_pids)} test paragraphs incrementally...")

        # Track metrics
        all_ap_scores: list[float] = []
        all_recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        for test_pid in tqdm(test_pids, desc="Incremental evaluation"):
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

            if len(candidate_pids) == 0:
                # Add to graph and continue
                self._add_node_to_graph(int(test_pid))
                continue

            # Only evaluate if there are citations to measure
            if (
                test_pid not in self.cited_by_pid
                or len(self.cited_by_pid[test_pid]) == 0
            ):
                # Add to graph and continue
                self._add_node_to_graph(int(test_pid))
                continue

            # Retrieve and evaluate
            result = self._retrieve_and_evaluate(
                int(test_pid), candidate_pids, k_values
            )

            if result is not None:
                all_ap_scores.append(result["ap"])
                for k in k_values:
                    all_recall_scores[k].append(result["recall"][k])
                self.query_results.append(result)

            # Add node to graph for future queries
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
        self.load_and_prepare()
        self.initialize_graph_with_training_data()
        metrics = self.evaluate_incremental()

        print("\n" + "=" * 80)
        print("Incremental Evaluation Results")
        print("=" * 80)

        metric_name = f"MAP@{self.top_k}" if self.top_k else "MAP"
        map_score = metrics.get(f"map@{self.top_k}" if self.top_k else "map", 0.0)
        print(f"{metric_name}: {map_score:.4f}")

        for k in [5, 10, 100]:
            if f"recall@{k}" in metrics:
                print(f"Recall@{k}: {metrics[f'recall@{k}']:.4f}")

        print(f"\nTotal queries evaluated: {len(self.query_results)}")

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
        in_channels, hidden_dim=512, output_dim=in_channels, num_layers=3
    )
    model.load_state_dict(torch.load("checkpoints/gnn/best_model.pt"))

    # Run incremental evaluation
    evaluator = IncrementalGNNEvaluator(
        gnn_model=model,
        text_encoder_name=encoding_model,
        csv_path="data/par-to-par-cleaned.csv",
        judgments_path="data/judgments_cleaned.json",
        train_cutoff_year=2018,
        k_hops=2,
        top_k=10000,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    metrics = evaluator.run()

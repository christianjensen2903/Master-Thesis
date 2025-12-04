import os

# Fix OpenMP conflict on macOS - MUST be set before importing torch/faiss
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from datetime import datetime

import faiss  # type: ignore

# Set FAISS to single-threaded mode to avoid segmentation faults
faiss.omp_set_num_threads(1)

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
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
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
            confidence: Confidence level for bootstrap intervals (default 0.95)
            n_bootstrap: Number of bootstrap samples (default 1000)
        """
        self.graph_builder = graph_builder
        self.par_to_par_path = par_to_par_path
        self.train_cutoff_year = train_cutoff_year
        self.k_hops = k_hops
        self.mode = mode
        self.top_k = top_k
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap
        self.is_hetero = isinstance(graph_builder, HeterogeneousGraphBuilder)

        # Results storage
        self.map_score: float | None = None
        self.recall_scores: dict[int, float] | None = None
        self.map_ci: tuple[float, float] | None = None
        self.recall_cis: dict[int, tuple[float, float]] | None = None

        # Per-query results for detailed analysis
        self.per_query_results: list[dict] | None = None

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.gnn_model = gnn_model.to(self.device)
        self.gnn_model.eval()
        self._remove_wandb_hooks()

        print(f"Using device: {self.device}")
        print(f"Graph builder: {type(graph_builder).__name__}")
        print(f"Train cutoff year: {self.train_cutoff_year}")

        self.graph_data = self._build_graph()
        self.embeddings = self._compute_initial_embeddings()
        self.citation_mask = self._build_citation_mask()

        # FAISS index for efficient similarity search (built lazily during evaluation)
        self.faiss_index: faiss.IndexIDMap | None = None

    def _remove_wandb_hooks(self) -> None:
        """Remove wandb hooks from the model to prevent errors during evaluation."""
        try:
            import wandb

            # Try to unwatch if wandb is initialized
            if wandb.run is not None:
                wandb.unwatch(self.gnn_model)
        except (ImportError, AttributeError):
            pass

        # Manually remove any forward hooks that might be from wandb
        # wandb registers hooks on forward_pre and forward hooks
        for module in self.gnn_model.modules():
            # Remove forward_pre hooks (wandb uses these)
            if hasattr(module, "_forward_pre_hooks"):
                hooks_to_remove = []
                for hook_id, hook in list(module._forward_pre_hooks.items()):
                    # Check if hook is from wandb by checking various attributes
                    hook_str = str(hook)
                    hook_fn = getattr(hook, "__func__", None) or getattr(
                        hook, "__self__", None
                    )
                    if hook_fn is not None:
                        hook_module = getattr(hook_fn, "__module__", "")
                        if "wandb" in str(hook_module).lower():
                            hooks_to_remove.append(hook_id)
                    elif "wandb" in hook_str.lower():
                        hooks_to_remove.append(hook_id)
                for hook_id in hooks_to_remove:
                    module._forward_pre_hooks.pop(hook_id, None)

            # Remove forward hooks (wandb also uses these)
            if hasattr(module, "_forward_hooks"):
                hooks_to_remove = []
                for hook_id, hook in list(module._forward_hooks.items()):
                    hook_str = str(hook)
                    hook_fn = getattr(hook, "__func__", None) or getattr(
                        hook, "__self__", None
                    )
                    if hook_fn is not None:
                        hook_module = getattr(hook_fn, "__module__", "")
                        if "wandb" in str(hook_module).lower():
                            hooks_to_remove.append(hook_id)
                    elif "wandb" in hook_str.lower():
                        hooks_to_remove.append(hook_id)
                for hook_id in hooks_to_remove:
                    module._forward_hooks.pop(hook_id, None)

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
        Uses batched node encoding + full-graph GNN to avoid OOM.
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

        # Clear CUDA cache before heavy computation
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        return self._compute_homo_embeddings_full(filtered_edges, filtered_attr)

    def _compute_homo_embeddings_full(
        self,
        filtered_edges: torch.Tensor,
        filtered_attr: torch.Tensor | None,
        batch_size: int = 256,
    ) -> torch.Tensor:
        """Compute embeddings using batched inference to avoid OOM.

        Uses NeighborLoader to sample subgraphs for each batch of nodes,
        then extracts embeddings for the seed nodes only.
        """
        num_nodes = self.graph_data.x.size(0)

        print(f"num nodes: {num_nodes}")

        # Create a temporary graph with filtered edges for the loader
        temp_graph = Data(
            x=self.graph_data.x,
            edge_index=filtered_edges.cpu(),
            edge_attr=filtered_attr.cpu() if filtered_attr is not None else None,
            time=self.graph_data.time,
        )
        # Copy optional attributes
        for attr in [
            "date_feature",
            "language",
            "subject_matter",
            "keywords",
            "case_law_about",
        ]:
            if hasattr(self.graph_data, attr):
                setattr(temp_graph, attr, getattr(self.graph_data, attr))

        # Use NeighborLoader to batch through all nodes
        loader = NeighborLoader(
            data=temp_graph,
            num_neighbors=[-1] * self.k_hops,
            input_nodes=torch.arange(num_nodes),
            batch_size=batch_size,
            shuffle=False,
        )

        all_embeddings: torch.Tensor | None = None

        with torch.no_grad():
            for batch in tqdm(loader, desc="Computing initial embeddings"):
                batch = batch.to(self.device)
                batch_size_actual = batch.batch_size

                # Get optional attributes
                date_feature = getattr(batch, "date_feature", None)
                language = getattr(batch, "language", None)
                subject_matter = getattr(batch, "subject_matter", None)
                keywords = getattr(batch, "keywords", None)
                case_law_about = getattr(batch, "case_law_about", None)
                edge_attr = getattr(batch, "edge_attr", None)

                # Compute embeddings
                if hasattr(self.gnn_model, "encode_document"):
                    emb = self.gnn_model.encode_document(
                        batch.x,
                        batch.edge_index,
                        date_feature=date_feature,
                        edge_attr=edge_attr,
                        language=language,
                        subject_matter=subject_matter,
                        keywords=keywords,
                        case_law_about=case_law_about,
                    )
                else:
                    emb = self.gnn_model(
                        batch.x,
                        batch.edge_index,
                        date_feature=date_feature,
                        edge_attr=edge_attr,
                        language=language,
                        subject_matter=subject_matter,
                        keywords=keywords,
                        case_law_about=case_law_about,
                    )

                # Allocate on first batch using actual output dimension
                if all_embeddings is None:
                    all_embeddings = torch.zeros(num_nodes, emb.size(1))

                # Store only seed node embeddings
                all_embeddings[batch.n_id[:batch_size_actual]] = emb[
                    :batch_size_actual
                ].cpu()

                # Clear cache periodically
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        assert all_embeddings is not None, "No batches processed"
        return all_embeddings

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

    def _create_loader(self, input_nodes, batch_size: int = 512) -> NeighborLoader:
        """Create a NeighborLoader for the given input nodes."""
        return NeighborLoader(
            data=self.graph_data,
            shuffle=False,
            input_nodes=input_nodes,
            num_neighbors=[-1] * self.k_hops,
            time_attr="time",
            batch_size=batch_size,
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
            date_feature = getattr(sub, "date_feature", None)
            language = getattr(sub, "language", None)
            subject_matter = getattr(sub, "subject_matter", None)
            keywords = getattr(sub, "keywords", None)
            case_law_about = getattr(sub, "case_law_about", None)

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
                subject_matter=subject_matter,
                keywords=keywords,
                case_law_about=case_law_about,
            )
            return embeddings[:num_nodes].cpu()

    def _get_k_hop_neighbors(
        self, seed_nodes: torch.Tensor, batch_size: int = 256
    ) -> torch.Tensor:
        """Find all nodes within k hops of seed nodes using NeighborLoader."""
        if len(seed_nodes) == 0:
            return torch.tensor([], dtype=torch.long)

        input_nodes = ("paragraph", seed_nodes) if self.is_hetero else seed_nodes
        loader = NeighborLoader(
            data=self.graph_data,
            num_neighbors=[-1] * self.k_hops,
            input_nodes=input_nodes,
            batch_size=batch_size,
            shuffle=False,
        )

        all_neighbors: set[int] = set()
        for sub in loader:
            n_id = sub["paragraph"].n_id if self.is_hetero else sub.n_id
            all_neighbors.update(n_id.tolist())

        return torch.tensor(list(all_neighbors), dtype=torch.long)

    def _update_embeddings(
        self, node_indices: torch.Tensor, batch_size: int = 256
    ) -> None:
        """Update embeddings for given nodes using their k-hop neighborhoods."""
        if len(node_indices) == 0:
            return

        input_nodes = ("paragraph", node_indices) if self.is_hetero else node_indices
        loader = self._create_loader(input_nodes, batch_size=batch_size)

        for batch in loader:
            batch = batch.to(self.device)
            actual_batch_size = batch.batch_size
            n_id = batch["paragraph"].n_id if self.is_hetero else batch.n_id

            with torch.no_grad():
                if self.is_hetero:
                    embeddings = self.gnn_model(batch)["paragraph"]
                else:
                    edge_attr = getattr(batch, "edge_attr", None)
                    date_feature = getattr(batch, "date_feature", None)
                    language = getattr(batch, "language", None)
                    subject_matter = getattr(batch, "subject_matter", None)
                    keywords = getattr(batch, "keywords", None)
                    case_law_about = getattr(batch, "case_law_about", None)
                    # Use document encoder if dual encoder
                    if hasattr(self.gnn_model, "encode_document"):
                        embeddings = self.gnn_model.encode_document(
                            batch.x,
                            batch.edge_index,
                            date_feature=date_feature,
                            edge_attr=edge_attr,
                            language=language,
                            subject_matter=subject_matter,
                            keywords=keywords,
                            case_law_about=case_law_about,
                        )
                    else:
                        embeddings = self.gnn_model(
                            batch.x,
                            batch.edge_index,
                            date_feature=date_feature,
                            edge_attr=edge_attr,
                            language=language,
                            subject_matter=subject_matter,
                            keywords=keywords,
                            case_law_about=case_law_about,
                        )

            # Update only the seed nodes in this batch
            self.embeddings[n_id[:actual_batch_size]] = embeddings[
                :actual_batch_size
            ].cpu()

            # Clear GPU memory
            del batch, embeddings
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

    def _init_faiss_index(self, dim: int) -> None:
        """Initialize FAISS index with ID mapping for updates."""
        base_index = faiss.IndexFlatIP(dim)
        self.faiss_index = faiss.IndexIDMap(base_index)

    def _add_to_faiss_index(
        self, indices: torch.Tensor, batch_size: int = 10000
    ) -> None:
        """Add embeddings at given indices to the FAISS index in batches."""
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialized")

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]

            # Get embeddings and normalize for cosine similarity
            # emb = self.embeddings[batch_indices].numpy()
            # norms = np.linalg.norm(emb, axis=1, keepdims=True)
            # norms = np.where(norms > 1e-10, norms, 1.0)
            # emb_normalized = (emb / norms).astype(np.float32)
            emb_normalized = self.embeddings[batch_indices].numpy()

            # Ensure contiguous array for FAISS
            if not emb_normalized.flags["C_CONTIGUOUS"]:
                emb_normalized = np.ascontiguousarray(emb_normalized)

            ids = batch_indices.numpy().astype(np.int64)
            self.faiss_index.add_with_ids(emb_normalized, ids)

    def _update_faiss_index(
        self, indices: torch.Tensor, batch_size: int = 10000
    ) -> None:
        """Update embeddings for existing indices in FAISS (remove + re-add) in batches."""
        if self.faiss_index is None or len(indices) == 0:
            return

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            ids = batch_indices.numpy().astype(np.int64)
            self.faiss_index.remove_ids(ids)

        self._add_to_faiss_index(indices, batch_size)

    def _search_faiss_index(
        self, query_emb: torch.Tensor, k: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index for top-k similar candidates.

        Args:
            query_emb: Query embeddings [num_queries, dim]
            k: Number of top candidates to return

        Returns:
            similarities: Similarity scores [num_queries, k]
            orig_indices: Original node indices [num_queries, k]
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not initialized")

        # Normalize query embeddings
        emb = query_emb.numpy()
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)
        emb_normalized = (emb / norms).astype(np.float32)

        if not emb_normalized.flags["C_CONTIGUOUS"]:
            emb_normalized = np.ascontiguousarray(emb_normalized)

        # Search FAISS index (IndexIDMap returns IDs directly)
        k_actual = min(k, self.faiss_index.ntotal)
        similarities, orig_indices = self.faiss_index.search(emb_normalized, k_actual)

        return similarities, orig_indices

    def _bootstrap_confidence_interval(
        self,
        values: NDArray,
        confidence: float | None = None,
        n_bootstrap: int | None = None,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence interval for the mean."""
        confidence = confidence or self.confidence
        n_bootstrap = n_bootstrap or self.n_bootstrap

        if len(values) == 0:
            return 0.0, 0.0

        rng = np.random.default_rng()
        means = np.empty(n_bootstrap, dtype=np.float64)
        n = len(values)

        for i in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            sample = values[indices]
            means[i] = float(np.mean(sample))

        alpha = 1.0 - confidence
        lower = float(np.quantile(means, alpha / 2.0))
        upper = float(np.quantile(means, 1.0 - alpha / 2.0))
        return lower, upper

    def get_per_query_results(self) -> list[dict]:
        """Return per-query results for detailed analysis."""
        if self.per_query_results is None:
            raise RuntimeError("Evaluation not run yet. Call run() first.")
        return self.per_query_results

    def save_per_query_results(self, path: str) -> None:
        """Save per-query results to JSON file."""
        import json

        if self.per_query_results is None:
            raise RuntimeError("Evaluation not run yet. Call run() first.")

        with open(path, "w") as f:
            json.dump(self.per_query_results, f, indent=2)
        print(f"Saved {len(self.per_query_results)} per-query results to {path}")

    def run(self, k_values: list[int] = [5, 10, 100]) -> float:
        """Run full incremental evaluation using FAISS for efficient similarity search."""
        df = pd.read_csv(self.par_to_par_path)
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        cutoff_date = datetime.strptime(str(self.train_cutoff_year), "%Y")
        df = df[df["DATE_FROM"] >= cutoff_date]

        times, node_id_hash, source_nodes = self._get_graph_attrs()

        # Initialize FAISS index with embeddings before cutoff
        cutoff_ts = int(cutoff_date.timestamp())
        initial_mask = times < cutoff_ts
        initial_indices = torch.where(initial_mask)[0]

        print(f"citation mask: {self.citation_mask}")

        if self.citation_mask is not None:
            initial_indices = initial_indices[
                torch.isin(initial_indices, self.citation_mask)
            ]

        emb_dim = self.embeddings.size(1)
        self._init_faiss_index(emb_dim)
        self._add_to_faiss_index(initial_indices)
        print(f"Initialized FAISS index with {self.faiss_index.ntotal} embeddings")

        # Track which nodes have been added to the index
        in_index = set(initial_indices.tolist())

        ap_scores = []
        recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        # Per-query results for detailed analysis
        per_query_results: list[dict] = []

        for date, group in tqdm(df.groupby("DATE_FROM"), desc="Evaluating"):
            date_str = date.normalize().strftime("%Y-%m-%d")
            timestamp = int(datetime.fromisoformat(date_str).timestamp())

            # Get nodes at current time with outgoing edges
            nodes_at_time = (times == timestamp).nonzero(as_tuple=True)[0]
            nodes_with_edges = nodes_at_time[torch.isin(nodes_at_time, source_nodes)]
            num_nodes = nodes_with_edges.size(0)

            if num_nodes == 0:
                continue

            if self.faiss_index.ntotal == 0:
                continue

            # Create subgraph and compute query embeddings
            input_nodes = (
                ("paragraph", nodes_with_edges) if self.is_hetero else nodes_with_edges
            )
            sub = next(iter(self._create_loader(input_nodes, batch_size=10000)))
            query_emb = self._process_subgraph(sub, num_nodes)

            # Search FAISS index for top-k candidates
            _, orig_indices = self._search_faiss_index(query_emb, self.top_k)
            del query_emb

            # Get node IDs
            sub_node_id_hash = (
                sub["paragraph"].node_id_hash if self.is_hetero else sub.node_id_hash
            )
            sub_n_id = sub["paragraph"].n_id if self.is_hetero else sub.n_id

            query_ids = [decode_celex(nid) for nid in sub_node_id_hash[:num_nodes]]
            cand_ids_ordered = node_id_hash[orig_indices]
            ranked_ids = [
                [decode_celex(nid) for nid in row] for row in cand_ids_ordered
            ]
            del orig_indices, cand_ids_ordered

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

                ap = compute_ap(ranked_array, relevant_set)
                ap_scores.append(ap)

                query_recalls = {}
                for k in k_values:
                    recall = compute_recall_at_k(
                        ranked_array, relevant_set, min(k, len(ranked_array))
                    )
                    recall_scores[k].append(recall)
                    query_recalls[k] = recall

                # Store per-query result
                per_query_results.append(
                    {
                        "query_celex": celex_from,
                        "query_number": number_from,
                        "ap": ap,
                        "recall": query_recalls,
                    }
                )

            # Add ALL nodes at this timestamp to FAISS and update affected neighbors
            new_indices = [idx for idx in nodes_at_time.tolist() if idx not in in_index]
            if new_indices:
                new_indices_tensor = torch.tensor(new_indices, dtype=torch.long)

                # Find k-hop neighbors of new nodes that are already in index
                # Their embeddings are affected by the new nodes
                k_hop_neighbors = self._get_k_hop_neighbors(new_indices_tensor)
                existing_neighbors = torch.tensor(
                    [idx for idx in k_hop_neighbors.tolist() if idx in in_index],
                    dtype=torch.long,
                )

                # Update embeddings for existing neighbors (affected by new edges)
                if len(existing_neighbors) > 0:
                    self._update_embeddings(existing_neighbors)
                    self._update_faiss_index(existing_neighbors)

                # Update embeddings for new nodes and add to FAISS
                self._update_embeddings(new_indices_tensor)
                self._add_to_faiss_index(new_indices_tensor)
                in_index.update(new_indices)

        # Compute metrics and confidence intervals
        ap_array = np.array(ap_scores, dtype=np.float64)
        self.map_score = float(np.mean(ap_array))
        self.map_ci = self._bootstrap_confidence_interval(ap_array)

        self.recall_scores = {}
        self.recall_cis = {}
        for k, recalls in recall_scores.items():
            recall_array = np.array(recalls, dtype=np.float64)
            self.recall_scores[k] = float(np.mean(recall_array))
            self.recall_cis[k] = self._bootstrap_confidence_interval(recall_array)

        # Store per-query results
        self.per_query_results = per_query_results

        # Print results
        confidence_pct = int(self.confidence * 100)
        print(
            f"\nMAP: {self.map_score:.3f} "
            f"({confidence_pct}% CI [{self.map_ci[0]:.3f}, {self.map_ci[1]:.3f}])"
        )
        for k in sorted(self.recall_scores.keys()):
            recall = self.recall_scores[k]
            ci = self.recall_cis[k]
            print(
                f"Recall@{k}: {recall:.3f} "
                f"({confidence_pct}% CI [{ci[0]:.3f}, {ci[1]:.3f}])"
            )

        return self.map_score


if __name__ == "__main__":
    from models import DualEncoderGNN, SymmetricGNN, MLPBaseline, CaseLinkGNN

    layers = 1

    model = SymmetricGNN(
        input_dim=1024,
        output_dim=1024,
        num_layers=layers,
        fusion_mode="cross_attention",
        use_language=False,
    )
    # in_channels = 1024
    # layers = 1

    # model = CaseLinkGNN(
    #     input_dim=in_channels,
    #     num_layers=layers,
    #     dropout=0.5,
    #     num_heads=4,
    # )

    model.load_state_dict(torch.load("checkpoints/homo_gnn_ablation2/best_model.pt"))

    # Option 1: Citation-based graph (HomogeneousGraphBuilder)
    graph_builder = HomogeneousGraphBuilder(
        preprocessed_dir="data/preprocessed_new",
        # include_only_citing=False,
    )

    # graph_builder = SemanticGraphBuilder(
    #     "data/preprocessed_new",
    #     "data/judgments_cleaned.json",
    #     semantic_cache_path="data/semantic_cache",
    #     semantic_threshold=0.0,
    #     semantic_max_neighbors=3,
    #     include_article_nodes=False,
    #     # include_only_citing=False,
    # )

    evaluator = GNNEvaluator(
        gnn_model=model,
        graph_builder=graph_builder,
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,  # Evaluate on data after this year
        k_hops=layers,
        device="cuda" if torch.cuda.is_available() else "cpu",
        # mode="all_paragraphs",
        top_k=10000,
    )

    evaluator.run()

    # evaluator.save_per_query_results("artifacts/per_query_results/caselink_gnn.json")

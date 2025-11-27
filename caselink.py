"""
CaseLink-inspired implementation for legal case retrieval.

Based on: "CaseLink: Inductive Graph Learning for Legal Case Retrieval" (Tang et al., SIGIR 2024)
https://arxiv.org/abs/2403.17780

Key adaptations:
- Cases → Paragraphs (our main entities)
- Charges → Legal Articles (articles/provisions cited by paragraphs)
- Case-case semantic edges → Paragraph semantic similarity edges
- Case-charge edges → Paragraph-to-article citation edges
"""

import pickle
import os
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GATConv

# Fix OpenMP conflict on macOS (FAISS and PyTorch may use different OpenMP runtimes)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss

# Set FAISS to single-threaded mode to avoid segmentation faults
faiss.omp_set_num_threads(1)


class CaseLinkGraphBuilder:
    """
    CaseLink-inspired graph builder.

    Creates a graph with:
    - Paragraph nodes (main entities)
    - Article nodes (like "charges" in CaseLink)
    - Multiple edge types encoded in edge_attr

    Edge types:
    - 0: paragraph references article
    - 1: article referenced_by paragraph
    - 2: paragraph similar_to paragraph (semantic similarity)
    - 3: article similar_to article (cosine similarity)
    - 4: paragraph cites paragraph (for neighbor sampling only, masked during forward pass)

    Citation pairs are also stored in graph_data.citation_pairs for loss computation.
    Edge type 4 is used for neighbor sampling but should be masked during message passing.
    """

    def __init__(self, preprocessed_dir: str):
        self.preprocessed_dir = Path(preprocessed_dir)

        # Load paragraph embeddings
        doc_emb_path = self.preprocessed_dir / "paragraph_embeddings_doc.npy"
        query_emb_path = self.preprocessed_dir / "paragraph_embeddings_query.npy"
        legacy_emb_path = self.preprocessed_dir / "paragraph_embeddings.npy"

        if doc_emb_path.exists() and query_emb_path.exists():
            self.par_embeddings_doc = np.load(doc_emb_path)
            self.par_embeddings_query = np.load(query_emb_path)
            print("Loaded separate document and query embeddings")
        elif legacy_emb_path.exists():
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

        # Load citations (includes both par->par and par->art)
        with open(self.preprocessed_dir / "citations.pkl", "rb") as f:
            self.citations = pickle.load(f)

        # Create ID mappings
        self.par_id_to_idx = {m["id"]: i for i, m in enumerate(self.par_metadata)}
        self.art_id_to_idx = {m["id"]: i for i, m in enumerate(self.art_metadata)}

        print(
            f"Loaded {len(self.par_metadata)} paragraphs, {len(self.art_metadata)} articles"
        )
        print(f"Loaded {len(self.citations)} citation edges")

    def _date_to_timestamp(self, date_str: str | None) -> int:
        """Convert ISO date string to Unix timestamp."""
        if not date_str:
            return 0
        try:
            dt = datetime.fromisoformat(date_str)
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            return 0

    def _extract_date_features(self, date_str: str | None) -> np.ndarray:
        """Extract normalized date feature."""
        if not date_str:
            return np.array([0.0], dtype=np.float32)
        try:
            dt = datetime.fromisoformat(date_str)
            base_date = datetime(1954, 1, 1)
            max_date = datetime(2025, 12, 31)
            days_since_base = (dt - base_date).days
            max_days = (max_date - base_date).days
            time_norm = max(0.0, min(1.0, days_since_base / max_days))
            return np.array([time_norm], dtype=np.float32)
        except (ValueError, AttributeError):
            return np.array([0.0], dtype=np.float32)

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
            if include_only_citing and meta["id"] not in citing_pars:
                continue
            if train_cutoff_year and meta.get("year"):
                if meta["year"] >= train_cutoff_year:
                    continue
            selected_pars.append(i)

        return selected_pars

    def _compute_semantic_similarity_edges(
        self,
        embeddings: np.ndarray,
        times: np.ndarray | None = None,
        threshold: float = 0.7,
        max_neighbors: int = 10,
        batch_size: int = 1024,
        use_temporal_constraint: bool = True,
    ) -> list[tuple[int, int]]:
        """
        Compute semantic similarity edges using FAISS with temporal constraints.

        Paragraphs can only link to paragraphs that came before them (by time).
        Uses grouping by time for efficiency - processes chronologically and
        incrementally builds the FAISS index.

        Args:
            embeddings: Node embeddings, shape (n, d)
            times: Timestamps for each node (Unix timestamps or any sortable int).
                   If None, temporal constraints are disabled.
            threshold: Cosine similarity threshold for creating an edge
            max_neighbors: Maximum number of neighbors per node
            batch_size: Batch size for FAISS queries (for memory efficiency)
            use_temporal_constraint: If True, only link to earlier nodes

        Returns:
            List of (source_idx, target_idx) edges where target came before source
        """
        n, d = embeddings.shape
        print(f"  Computing semantic edges for {n} nodes using FAISS...")

        # Normalize embeddings for cosine similarity
        # FAISS IndexFlatIP computes inner product = cosine sim for normalized vectors
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms > 1e-10, norms, 1.0)  # Avoid division by zero
        embeddings_normalized = (embeddings / norms).astype(np.float32)

        # Ensure contiguous array for FAISS
        if not embeddings_normalized.flags["C_CONTIGUOUS"]:
            embeddings_normalized = np.ascontiguousarray(embeddings_normalized)

        if times is None or not use_temporal_constraint:
            edges = self._compute_edges_no_temporal(
                embeddings_normalized, threshold, max_neighbors, batch_size
            )
        else:
            edges = self._compute_edges_temporal(
                embeddings_normalized, times, threshold, max_neighbors, batch_size
            )

        print(f"  Found {len(edges)} semantic similarity edges")
        return edges

    def _compute_edges_no_temporal(
        self,
        embeddings: np.ndarray,
        threshold: float,
        max_neighbors: int,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        """Compute edges without temporal constraints."""
        n, d = embeddings.shape

        # Build FAISS index with all embeddings
        index = faiss.IndexFlatIP(d)

        # Ensure embeddings are contiguous
        if not embeddings.flags["C_CONTIGUOUS"]:
            embeddings = np.ascontiguousarray(embeddings)

        index.add(embeddings)

        edges = []

        # Query in batches for memory efficiency
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_embs = embeddings[start:end]

            # +1 because first result is the node itself
            k = min(max_neighbors + 1, n)
            similarities, neighbors = index.search(batch_embs, k)

            for i, orig_idx in enumerate(range(start, end)):
                for j in range(k):
                    neighbor_idx = neighbors[i, j]
                    sim = similarities[i, j]

                    # Skip self-loops
                    if neighbor_idx == orig_idx:
                        continue

                    if sim >= threshold:
                        edges.append((orig_idx, neighbor_idx))

        return edges

    def _compute_edges_temporal(
        self,
        embeddings: np.ndarray,
        times: np.ndarray,
        threshold: float,
        max_neighbors: int,
        batch_size: int,
    ) -> list[tuple[int, int]]:
        """
        Compute edges with temporal constraints.

        Process nodes in chronological order, building up the index incrementally.
        Each node can only find neighbors among nodes with earlier timestamps.
        """
        n, d = embeddings.shape

        # Group indices by time
        time_to_indices: dict[int, list[int]] = defaultdict(list)
        for idx in range(n):
            time_to_indices[times[idx]].append(idx)

        # Sort unique times chronologically
        unique_times = sorted(time_to_indices.keys())
        print(f"  Processing {len(unique_times)} time groups chronologically...")

        # Create FAISS index (will be built incrementally)
        index = faiss.IndexFlatIP(d)

        # Track mapping: FAISS internal index -> original node index
        faiss_to_orig: list[int] = []

        edges = []
        nodes_processed = 0

        for time_idx, t in enumerate(unique_times):
            group_indices = time_to_indices[t]
            group_size = len(group_indices)

            # Search for similar nodes in the index (only earlier nodes)
            if index.ntotal > 0 and group_size > 0:
                group_embs = embeddings[group_indices]

                # Ensure contiguous array
                if not group_embs.flags["C_CONTIGUOUS"]:
                    group_embs = np.ascontiguousarray(group_embs)

                # Process group in batches if large
                for batch_start in range(0, group_size, batch_size):
                    batch_end = min(batch_start + batch_size, group_size)
                    batch_indices = group_indices[batch_start:batch_end]
                    batch_embs = group_embs[batch_start:batch_end]

                    # Ensure batch is contiguous
                    if not batch_embs.flags["C_CONTIGUOUS"]:
                        batch_embs = np.ascontiguousarray(batch_embs)

                    k = min(max_neighbors, index.ntotal)
                    similarities, faiss_neighbors = index.search(batch_embs, k)

                    for i, orig_idx in enumerate(batch_indices):
                        for j in range(k):
                            sim = similarities[i, j]
                            if sim >= threshold:
                                faiss_idx = faiss_neighbors[i, j]
                                neighbor_orig_idx = faiss_to_orig[faiss_idx]
                                edges.append((orig_idx, neighbor_orig_idx))

            # Add current group to the index for future searches
            if group_size > 0:
                group_embs = embeddings[group_indices]

                # Ensure contiguous array before adding to FAISS
                if not group_embs.flags["C_CONTIGUOUS"]:
                    group_embs = np.ascontiguousarray(group_embs)

                index.add(group_embs)
                faiss_to_orig.extend(group_indices)

            nodes_processed += group_size

            # Progress update every 10%
            if (time_idx + 1) % max(1, len(unique_times) // 10) == 0:
                pct = 100 * (time_idx + 1) / len(unique_times)
                print(
                    f"    {pct:.0f}% complete ({nodes_processed}/{n} nodes, {len(edges)} edges)"
                )

        return edges

    def build_graph(
        self,
        train_cutoff_year: int | None = None,
        include_only_citing: bool = True,
        include_semantic_edges: bool = True,
        semantic_threshold: float = 0.7,
        article_threshold: float = 0.9,
        semantic_max_neighbors: int = 10,
        include_article_nodes: bool = True,
        use_temporal_constraint: bool = True,
    ) -> Data:
        """
        Build CaseLink-style graph.

        Args:
            train_cutoff_year: Only include paragraphs before this year
            include_only_citing: Only include paragraphs involved in citations
            include_semantic_edges: Add semantic similarity edges (like CaseLink)
            semantic_threshold: Cosine similarity threshold for paragraph semantic edges
            article_threshold: Cosine similarity threshold for article semantic edges
            semantic_max_neighbors: Max number of semantic neighbors per node
            include_article_nodes: Include article nodes in the graph
            use_temporal_constraint: For semantic edges, only link to earlier paragraphs

        Returns:
            graph_data: PyTorch Geometric Data object with multiple node/edge types
        """
        # Filter paragraphs
        selected_pars = self._filter_paragraphs(include_only_citing, train_cutoff_year)
        print(f"Selected {len(selected_pars)} paragraphs")

        # Build paragraph node mappings
        par_node_id_to_idx: dict[str, int] = {}
        par_doc_embeddings_list = []
        par_query_embeddings_list = []
        par_date_features_list = []
        par_times = []

        for par_idx in selected_pars:
            meta = self.par_metadata[par_idx]
            node_id = meta["id"]
            current_idx = len(par_node_id_to_idx)

            par_node_id_to_idx[node_id] = current_idx
            par_doc_embeddings_list.append(self.par_embeddings_doc[par_idx])
            par_query_embeddings_list.append(self.par_embeddings_query[par_idx])
            par_date_features_list.append(self._extract_date_features(meta.get("date")))
            par_times.append(self._date_to_timestamp(meta.get("date")))

        num_par_nodes = len(par_node_id_to_idx)

        # Build article node mappings (if enabled)
        art_node_id_to_idx: dict[str, int] = {}
        art_embeddings_list = []

        if include_article_nodes:
            # Find which articles are referenced by selected paragraphs
            referenced_articles = set()
            for src_id, tgt_id in self.citations:
                if src_id in par_node_id_to_idx and tgt_id in self.art_id_to_idx:
                    referenced_articles.add(tgt_id)

            for art_id in referenced_articles:
                art_orig_idx = self.art_id_to_idx[art_id]
                current_idx = len(art_node_id_to_idx)
                art_node_id_to_idx[art_id] = current_idx
                art_embeddings_list.append(self.art_embeddings[art_orig_idx])

            print(
                f"Selected {len(art_node_id_to_idx)} articles referenced by paragraphs"
            )

        num_art_nodes = len(art_node_id_to_idx)
        total_nodes = num_par_nodes + num_art_nodes

        # Create combined node features
        par_emb_dim = self.par_embeddings_doc.shape[1]
        art_emb_dim = (
            self.art_embeddings.shape[1]
            if len(art_embeddings_list) > 0
            else par_emb_dim
        )

        # Node type indicator: 0 = paragraph, 1 = article
        node_types = []

        # If dimensions differ, we need to project (or pad)
        # For simplicity, we'll use the paragraph dimension and project articles
        all_doc_embeddings = []
        all_query_embeddings = []

        # Add paragraph embeddings
        for emb in par_doc_embeddings_list:
            all_doc_embeddings.append(emb)
            node_types.append(0)
        for emb in par_query_embeddings_list:
            all_query_embeddings.append(emb)

        # Add article embeddings (with projection if needed)
        if include_article_nodes and len(art_embeddings_list) > 0:
            art_emb_array = np.array(art_embeddings_list)
            if art_emb_dim != par_emb_dim:
                # Simple projection: pad or truncate
                if art_emb_dim < par_emb_dim:
                    padding = np.zeros(
                        (len(art_embeddings_list), par_emb_dim - art_emb_dim)
                    )
                    art_emb_array = np.concatenate([art_emb_array, padding], axis=1)
                else:
                    art_emb_array = art_emb_array[:, :par_emb_dim]

            for emb in art_emb_array:
                all_doc_embeddings.append(emb)
                all_query_embeddings.append(emb)  # Articles use same embedding for both
                node_types.append(1)

        # Build edges with different types
        edge_list = []
        edge_attr_list = []

        # Collect paragraph citation pairs and add as edges (type 4)
        # These edges are used for neighbor sampling but masked during forward pass
        # Also stored separately in citation_pairs for loss computation
        citation_src_list = []
        citation_tgt_list = []
        for src_id, tgt_id in self.citations:
            if src_id in par_node_id_to_idx and tgt_id in par_node_id_to_idx:
                src_idx = par_node_id_to_idx[src_id]
                tgt_idx = par_node_id_to_idx[tgt_id]
                citation_src_list.append(src_idx)
                citation_tgt_list.append(tgt_idx)
                # Add bidirectional citation edges (type 4) for neighbor sampling
                edge_list.append([src_idx, tgt_idx])
                edge_attr_list.append(4)
                edge_list.append([tgt_idx, src_idx])
                edge_attr_list.append(4)

        print(
            f"  Paragraph citation edges (type 4, for sampling): {len(citation_src_list)} (bidirectional)"
        )

        # Edge type 0: paragraph references article
        # Edge type 1: article referenced_by paragraph
        if include_article_nodes and len(art_node_id_to_idx) > 0:
            par_art_edges = 0
            for src_id, tgt_id in self.citations:
                if src_id in par_node_id_to_idx and tgt_id in art_node_id_to_idx:
                    par_idx = par_node_id_to_idx[src_id]
                    art_idx = num_par_nodes + art_node_id_to_idx[tgt_id]
                    edge_list.append([par_idx, art_idx])
                    edge_attr_list.append(0)
                    edge_list.append([art_idx, par_idx])
                    edge_attr_list.append(1)
                    par_art_edges += 1

            print(f"  Paragraph-article edges: {par_art_edges} (bidirectional)")

        # Edge type 2: paragraph similar_to paragraph (semantic)
        if include_semantic_edges:
            print("Computing semantic similarity edges with FAISS...")
            par_embeddings = np.array(par_doc_embeddings_list)
            par_times_array = np.array(par_times)

            semantic_edges = self._compute_semantic_similarity_edges(
                par_embeddings,
                times=par_times_array if use_temporal_constraint else None,
                threshold=semantic_threshold,
                max_neighbors=semantic_max_neighbors,
                use_temporal_constraint=use_temporal_constraint,
            )
            for src_idx, tgt_idx in semantic_edges:
                edge_list.append([src_idx, tgt_idx])
                edge_list.append([tgt_idx, src_idx])
                edge_attr_list.append(2)
                edge_attr_list.append(2)
            print(f"  Semantic similarity edges: {len(semantic_edges)}")

        # Edge type 3: article similar_to article (cosine similarity)
        if include_article_nodes and len(art_node_id_to_idx) > 1:
            print("Computing article similarity edges with FAISS...")
            art_emb_array = np.array(art_embeddings_list)

            # Use same method as paragraph similarity (no temporal constraint for articles)
            article_semantic_edges = self._compute_semantic_similarity_edges(
                art_emb_array,
                times=None,  # Articles don't have temporal constraints
                threshold=article_threshold,
                max_neighbors=semantic_max_neighbors,
                use_temporal_constraint=False,
            )
            for src_idx, tgt_idx in article_semantic_edges:
                global_src = num_par_nodes + src_idx
                global_tgt = num_par_nodes + tgt_idx
                edge_list.append([global_src, global_tgt])
                edge_attr_list.append(3)
            print(
                f"  Article-article cosine similarity edges: {len(article_semantic_edges)}"
            )

        # Create tensors
        x_doc = torch.tensor(np.array(all_doc_embeddings), dtype=torch.float32)
        x_query = torch.tensor(np.array(all_query_embeddings), dtype=torch.float32)
        node_type = torch.tensor(node_types, dtype=torch.long)

        # Date features only for paragraphs, zeros for articles
        if include_article_nodes and num_art_nodes > 0:
            art_date_features = [
                np.array([0.0], dtype=np.float32) for _ in range(num_art_nodes)
            ]
            all_date_features = par_date_features_list + art_date_features
        else:
            all_date_features = par_date_features_list
        date_features = torch.tensor(np.array(all_date_features), dtype=torch.float32)

        # Time for temporal sampling (articles get time 0)
        if include_article_nodes and num_art_nodes > 0:
            art_times = [0] * num_art_nodes
            all_times = par_times + art_times
        else:
            all_times = par_times
        time_tensor = torch.tensor(all_times, dtype=torch.long)

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr_list, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0,), dtype=torch.long)

        # Store citation pairs for training (not as graph edges)
        if citation_src_list:
            citation_pairs = torch.tensor(
                [citation_src_list, citation_tgt_list], dtype=torch.long
            )
        else:
            citation_pairs = torch.empty((2, 0), dtype=torch.long)

        graph_data = Data(
            x=x_doc,
            x_query=x_query,
            date_feature=date_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_type=node_type,  # 0=paragraph, 1=article
            num_nodes=total_nodes,
            num_par_nodes=num_par_nodes,
            num_art_nodes=num_art_nodes,
            time=time_tensor,
            citation_pairs=citation_pairs,  # [2, num_citations] - src cites tgt
        )

        print(f"\nBuilt CaseLink-style graph:")
        print(
            f"  Total nodes: {total_nodes} ({num_par_nodes} paragraphs, {num_art_nodes} articles)"
        )
        print(f"  Total edges: {edge_index.shape[1]}")
        print(f"  Citation pairs for training: {citation_pairs.shape[1]}")
        print(
            f"  Edge types: 0=references, 1=referenced_by, 2=par_similar, 3=art_similar, 4=cites (masked)"
        )

        return graph_data


# =============================================================================
# GNN Model
# =============================================================================


class CaseLinkGNN(nn.Module):
    """
    CaseLink-inspired GNN for legal case retrieval.

    Key features:
    - Multiple edge type handling
    - Residual connections (like CaseLink)
    - Degree-aware embeddings
    - Optional attention mechanism
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_heads: int = 4,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        # GNN layers - use separate convolutions per edge type or shared
        self.convs = nn.ModuleList()

        self.convs.append(
            GATConv(
                input_dim,
                hidden_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
        )

        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(
                    hidden_dim,
                    hidden_dim,
                    heads=num_heads,
                    concat=False,
                    dropout=dropout,
                    add_self_loops=False,
                )
            )

        self.convs.append(
            GATConv(
                hidden_dim,
                output_dim,
                heads=num_heads,
                concat=False,
                dropout=dropout,
                add_self_loops=False,
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:

        in_feat = x  # Store input for residual connection
        h = x

        # Apply all layers with activation and normalization
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)

            # Apply activation and norm for all layers except the last
            if i < self.num_layers - 1:
                h = F.relu(h)

        h = h + in_feat
        h = F.normalize(h, p=2, dim=1)

        return h


class CaseLinkGNNRelational(nn.Module):
    """
    CaseLink-inspired GNN with relation-aware message passing.

    Uses different transformations for different edge types,
    similar to R-GCN but simpler.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | None = None,
        output_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.3,
        num_edge_types: int = 6,
        degree_embed_dim: int = 32,
    ):
        super().__init__()

        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.num_layers = num_layers
        self.num_edge_types = num_edge_types
        self.hidden_dim = hidden_dim

        # Node type embedding
        self.node_type_embedding = nn.Embedding(2, hidden_dim)

        # Degree encoder
        self.degree_encoder = nn.Sequential(
            nn.Linear(2, degree_embed_dim),
            nn.GELU(),
            nn.Linear(degree_embed_dim, hidden_dim),
        )

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Relation-specific message transforms
        # Each edge type has its own linear transform
        self.relation_transforms = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Linear(hidden_dim, hidden_dim, bias=False)
                        for _ in range(num_edge_types)
                    ]
                )
                for _ in range(num_layers)
            ]
        )

        # Layer norms
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Output projection
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def compute_degree_features(
        self, edge_index: torch.Tensor, num_nodes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        in_degree = torch.zeros(num_nodes, device=edge_index.device)
        out_degree = torch.zeros(num_nodes, device=edge_index.device)
        ones = torch.ones(edge_index.size(1), device=edge_index.device)
        in_degree.scatter_add_(0, edge_index[1], ones)
        out_degree.scatter_add_(0, edge_index[0], ones)
        return in_degree, out_degree

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        date_feature: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_nodes = x.size(0)

        # Input projection
        h = self.input_proj(x)

        # Add node type embedding
        if node_type is not None:
            h = h + self.node_type_embedding(node_type)

        # Add degree features
        in_deg, out_deg = self.compute_degree_features(edge_index, num_nodes)
        degree_feats = torch.stack([in_deg, out_deg], dim=1)
        degree_feats = torch.log1p(degree_feats)
        degree_embedding = self.degree_encoder(degree_feats)
        h = h + degree_embedding

        # Relational message passing layers
        for layer_idx in range(self.num_layers):
            h_norm = self.norms[layer_idx](h)

            # Aggregate messages per edge type
            aggregated = torch.zeros_like(h)
            edge_count = torch.zeros(num_nodes, 1, device=h.device)

            if edge_attr is not None:
                for edge_type in range(self.num_edge_types):
                    # Get edges of this type
                    mask = edge_attr == edge_type
                    if mask.sum() == 0:
                        continue

                    type_edges = edge_index[:, mask]
                    src, tgt = type_edges

                    # Transform source node features
                    transform = self.relation_transforms[layer_idx][edge_type]
                    messages = transform(h_norm[src])

                    # Aggregate to target nodes
                    aggregated.scatter_add_(
                        0, tgt.unsqueeze(1).expand(-1, self.hidden_dim), messages
                    )
                    edge_count.scatter_add_(
                        0, tgt.unsqueeze(1), torch.ones_like(tgt.unsqueeze(1).float())
                    )
            else:
                # Fallback: single edge type
                src, tgt = edge_index
                transform = self.relation_transforms[layer_idx][0]
                messages = transform(h_norm[src])
                aggregated.scatter_add_(
                    0, tgt.unsqueeze(1).expand(-1, self.hidden_dim), messages
                )
                edge_count.scatter_add_(
                    0, tgt.unsqueeze(1), torch.ones_like(tgt.unsqueeze(1).float())
                )

            # Mean aggregation
            edge_count = edge_count.clamp(min=1)
            h_new = aggregated / edge_count
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)

            # Residual connection
            h = h + h_new

        # Output projection
        out = self.projector(h)
        out = F.normalize(out, p=2, dim=1)

        return out


# =============================================================================
# Degree Regularization Loss (CaseLink feature)
# =============================================================================


def degree_regularization_loss(
    embeddings: torch.Tensor,
    edge_index: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Degree regularization loss from CaseLink.

    Encourages node embeddings to be less dependent on node degree,
    preventing popular nodes from dominating the representation space.
    """
    num_nodes = embeddings.size(0)

    # Compute degree
    degree = torch.zeros(num_nodes, device=embeddings.device)
    ones = torch.ones(edge_index.size(1), device=embeddings.device)
    degree.scatter_add_(0, edge_index[0], ones)
    degree.scatter_add_(0, edge_index[1], ones)

    # Normalize embeddings
    emb_norm = F.normalize(embeddings, p=2, dim=1)

    # Compute similarity between nodes and their degree-weighted versions
    # This encourages embeddings to be independent of degree
    degree_weight = torch.log1p(degree).unsqueeze(1)
    degree_weighted_emb = emb_norm * degree_weight

    # The loss encourages decorrelation between embedding magnitude and degree
    correlation = (emb_norm * degree_weighted_emb).sum(dim=1).mean()

    return correlation


def info_nce_loss_with_degree_reg(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    temperature: float = 0.07,
    edge_index: torch.Tensor | None = None,
    degree_reg_weight: float = 0.1,
    anchor_times: torch.Tensor | None = None,
    positive_times: torch.Tensor | None = None,
    anchor_indices: torch.Tensor | None = None,
    positive_indices: torch.Tensor | None = None,
    return_stats: bool = False,
    all_embeddings: torch.Tensor | None = None,  # ADD THIS PARAMETER
):
    """
    InfoNCE loss with optional degree regularization (CaseLink style).
    """
    # Compute similarity matrix
    sim_matrix = torch.mm(anchor, positive.t()) / temperature
    batch_size = sim_matrix.size(0)
    diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)

    # False negative masking
    false_negative_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)
    if anchor_indices is not None and positive_indices is not None:
        same_anchor = anchor_indices.unsqueeze(1) == anchor_indices.unsqueeze(0)
        same_target = positive_indices.unsqueeze(1) == positive_indices.unsqueeze(0)
        false_negative_mask = (same_anchor.float() @ same_target.float()) > 0

    final_mask = false_negative_mask & ~diagonal_mask
    sim_matrix = sim_matrix.masked_fill(final_mask, float("-inf"))

    # Temporal masking
    if anchor_times is not None and positive_times is not None:
        time_mask = positive_times.unsqueeze(0) < anchor_times.unsqueeze(1)
        time_mask = time_mask | diagonal_mask
        sim_matrix = sim_matrix.masked_fill(~time_mask, float("-inf"))

    # Compute loss
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)
    nce_loss = F.cross_entropy(sim_matrix, labels)

    # Add degree regularization if edge_index is provided
    total_loss = nce_loss
    deg_reg = None
    if edge_index is not None and degree_reg_weight > 0:
        # USE all_embeddings if provided, otherwise fall back to concat
        if all_embeddings is not None:
            deg_reg = degree_regularization_loss(all_embeddings, edge_index)
        else:
            # Fallback (may cause index errors if edge_index is from larger graph)
            combined = torch.cat([anchor, positive], dim=0)
            deg_reg = degree_regularization_loss(combined, edge_index)
        total_loss = nce_loss + degree_reg_weight * deg_reg

    if not return_stats:
        return total_loss

    # Compute statistics
    stats = {}
    stats["nce_loss"] = nce_loss.item()
    if deg_reg is not None:
        stats["deg_reg"] = deg_reg.item()
        stats["deg_reg_weighted"] = (degree_reg_weight * deg_reg).item()
    stats["total_loss"] = total_loss.item()
    positive_sims = torch.diagonal(sim_matrix)
    stats["pos_sim_mean"] = positive_sims.mean().item()
    stats["pos_sim_std"] = positive_sims.std().item()

    valid_mask = ~torch.isinf(sim_matrix) & ~diagonal_mask
    if valid_mask.any():
        negative_sims = sim_matrix[valid_mask]
        stats["neg_sim_mean"] = negative_sims.mean().item()
        stats["neg_sim_std"] = negative_sims.std().item()

        num_valid_negatives = valid_mask.sum(dim=1).float()
        stats["num_negatives_mean"] = num_valid_negatives.mean().item()

        ranks = (sim_matrix > positive_sims.unsqueeze(1)).sum(dim=1) + 1
        stats["pos_rank_mean"] = ranks.float().mean().item()

        for k in [1, 5, 10]:
            if k <= sim_matrix.size(1):
                acc_at_k = (ranks <= k).float().mean().item()
                stats[f"acc@{k}"] = acc_at_k

    return total_loss, stats

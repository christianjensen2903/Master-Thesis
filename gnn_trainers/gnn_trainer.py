import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.data import Data, HeteroData  # type: ignore
from tqdm import tqdm  # type: ignore
from collections import defaultdict
import hashlib
import pickle
from pathlib import Path
from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
)


def info_nce_loss_fast(anchor, positive, negatives, temperature=0.07):
    """
    Optimized contrastive loss with explicit hard negatives.

    Args:
        anchor: [batch_size, dim]
        positive: [batch_size, dim]
        negatives: [batch_size, num_negatives, dim]
        temperature: temperature parameter

    Returns:
        loss: scalar
    """
    batch_size = anchor.size(0)

    # Normalize once
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Positive similarities: [batch_size, 1]
    pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / temperature

    # Negative similarities: [batch_size, num_neg]
    neg_sim = (anchor.unsqueeze(1) * negatives).sum(dim=-1) / temperature

    # Concatenate: [batch_size, 1 + num_neg]
    logits = torch.cat([pos_sim, neg_sim], dim=1)

    # Labels: positive at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)

    return F.cross_entropy(logits, labels)


class GNNTrainer:
    def __init__(
        self,
        preprocessed_dir: str,
        output_path: str = "output/gnn",
        batch_size: int = 16,
        epochs: int = 5,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_hops: int = 2,
        graph_type: str = "heterogeneous",
        patience: int = 3,
        num_hard_negatives: int = 5,
        hard_negative_pool_size: int = 300,  # Increased to accommodate range
        hard_negative_start_rank: int = 100,  # Start sampling from rank 100
        hard_negative_end_rank: int = 300,  # End sampling at rank 300
        include_only_citing: bool = True,
    ):
        self.preprocessed_dir = preprocessed_dir
        self.output_path = output_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.num_hops = num_hops
        self.graph_type = graph_type
        self.patience = patience
        self.num_hard_negatives = num_hard_negatives
        self.hard_negative_pool_size = hard_negative_pool_size
        self.hard_negative_start_rank = hard_negative_start_rank
        self.hard_negative_end_rank = hard_negative_end_rank or hard_negative_pool_size
        self.include_only_citing = include_only_citing

        # Validate parameters
        if self.hard_negative_start_rank >= self.hard_negative_end_rank:
            raise ValueError(
                "hard_negative_start_rank must be less than hard_negative_end_rank"
            )
        if self.hard_negative_end_rank > self.hard_negative_pool_size:
            raise ValueError(
                "hard_negative_end_rank cannot exceed hard_negative_pool_size"
            )

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")
        print(
            f"Hard negatives: sampling {num_hard_negatives} from ranks "
            f"{hard_negative_start_rank}-{self.hard_negative_end_rank} "
            f"(pool size: {hard_negative_pool_size})"
        )
        print(f"Include only citing: {include_only_citing}")

    def _get_hard_negatives_cache_path(
        self, num_nodes, train_cutoff_year, include_only_citing
    ):
        """Generate cache path for hard negatives based on configuration."""
        # Create cache directory
        cache_dir = Path(self.preprocessed_dir) / "hard_negatives_cache"
        cache_dir.mkdir(exist_ok=True)

        # Create unique hash based on configuration (now includes temporal flag)
        config_str = f"{num_nodes}_{train_cutoff_year}_{include_only_citing}_{self.hard_negative_pool_size}_{self.graph_type}_temporal"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        cache_file = cache_dir / f"hard_neg_{config_hash}.pt"
        return cache_file

    def _compute_hard_negatives(
        self, embeddings, edge_index=None, train_cutoff_year=None, time_attr=None
    ):
        """
        Pre-compute hard negatives based on initial embeddings.
        Uses caching to avoid recomputation.
        Optimized for large graphs with vectorized operations.
        Now includes temporal filtering: only papers published BEFORE the anchor can be negatives.

        Args:
            embeddings: Node embeddings
            edge_index: Graph edges (for positive pair filtering)
            train_cutoff_year: Training cutoff year
            time_attr: Tensor of timestamps/years for each node (required for temporal filtering)
        """
        num_nodes = embeddings.shape[0]

        # Check cache first
        cache_path = self._get_hard_negatives_cache_path(
            num_nodes, train_cutoff_year, self.include_only_citing
        )

        if cache_path.exists():
            print(f"\nLoading cached hard negatives from {cache_path}")
            hard_negatives = torch.load(cache_path, map_location=self.device)
            print(f"Loaded hard negatives: {hard_negatives.shape}")
            return hard_negatives

        print(f"\nComputing hard negatives (will cache to {cache_path})...")
        print(f"Number of nodes: {num_nodes}")
        print(
            f"Temporal filtering: {'ENABLED' if time_attr is not None else 'DISABLED'}"
        )
        embeddings = embeddings.to(self.device)
        embeddings_norm = F.normalize(embeddings, dim=-1)

        # Move time attributes to device if provided
        if time_attr is not None:
            time_attr = time_attr.to(self.device)

        # Build positive edge lookup (dict for fast access)
        positive_edges = None
        if edge_index is not None and edge_index.numel() > 0:
            print("Building positive pairs index...")
            edge_index = edge_index.to(self.device)

            num_edges = edge_index.size(1)

            # Create bidirectional edge dictionary for O(1) lookup
            # Key: source node, Value: tensor of target nodes
            positive_edges = {}

            src = edge_index[0].cpu()
            tgt = edge_index[1].cpu()

            # Forward edges
            for s, t in zip(src.tolist(), tgt.tolist()):
                if s not in positive_edges:
                    positive_edges[s] = []
                positive_edges[s].append(t)

            # Backward edges (make bidirectional)
            for s, t in zip(src.tolist(), tgt.tolist()):
                if t not in positive_edges:
                    positive_edges[t] = []
                positive_edges[t].append(s)

            # Convert lists to tensors for faster operations
            for node in positive_edges:
                positive_edges[node] = torch.tensor(
                    list(set(positive_edges[node])), device=self.device
                )

            print(f"Positive pairs: {num_edges} edges (bidirectional index built)")

        # Compute hard negatives in batches
        hard_neg_list = []
        batch_size = 500  # Smaller batches for memory efficiency

        print("Computing similarity and selecting hard negatives...")
        for start_idx in tqdm(
            range(0, num_nodes, batch_size), desc="Processing batches"
        ):
            end_idx = min(start_idx + batch_size, num_nodes)
            current_batch_size = end_idx - start_idx
            batch_emb = embeddings_norm[start_idx:end_idx]

            # Compute similarity with all nodes
            sim = torch.mm(batch_emb, embeddings_norm.t())  # [batch_size, num_nodes]

            # Mask out self-connections (diagonal for this batch)
            batch_indices = torch.arange(start_idx, end_idx, device=self.device)
            sim[torch.arange(current_batch_size, device=self.device), batch_indices] = (
                -1e9
            )

            # TEMPORAL FILTERING: Mask out papers published AFTER the anchor
            if time_attr is not None:
                # Get times for current batch of anchors
                batch_times = time_attr[start_idx:end_idx]  # [batch_size]

                # Compare with all node times
                # Only keep candidates where candidate_time <= anchor_time
                # Shape: [batch_size, num_nodes]
                temporal_mask = time_attr.unsqueeze(0) > batch_times.unsqueeze(1)

                # Mask out future papers (set their similarity to very low)
                sim[temporal_mask] = -1e9

            # Mask out positive pairs efficiently using dictionary lookup
            if positive_edges is not None:
                for i in range(current_batch_size):
                    node_idx = start_idx + i
                    if node_idx in positive_edges:
                        # Mask all positive neighbors for this node
                        positive_neighbors = positive_edges[node_idx]
                        sim[i, positive_neighbors] = -1e9

            # Get top-k (need full pool size to sample from range)
            k = min(self.hard_negative_pool_size, num_nodes - 1)
            _, topk_indices = torch.topk(sim, k=k, dim=-1)
            hard_neg_list.append(topk_indices.cpu())  # Move to CPU to save GPU memory

            # Clear cache periodically
            if (start_idx // batch_size) % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Concatenating results...")
        hard_negatives = torch.cat(hard_neg_list, dim=0)
        print(f"Computed hard negatives: {hard_negatives.shape}")

        # Cache to disk
        print(f"Caching hard negatives to {cache_path}")
        torch.save(hard_negatives, cache_path)

        # Move back to device
        hard_negatives = hard_negatives.to(self.device)
        return hard_negatives

    def _sample_random_negatives_with_temporal_filter(
        self, embeddings, batch_size, node_ids, time_attr
    ):
        """
        Sample random negatives with temporal filtering.
        Only samples nodes published before each anchor.

        Args:
            embeddings: All node embeddings in the batch
            batch_size: Number of anchor nodes
            node_ids: Global node IDs for all nodes in batch
            time_attr: Global time attributes for all nodes

        Returns:
            hard_neg_emb: [batch_size, num_hard_negatives, dim]
        """
        hard_neg_indices = torch.zeros(
            (batch_size, self.num_hard_negatives), dtype=torch.long, device=self.device
        )

        if time_attr is not None and node_ids is not None:
            # Sample with temporal filtering
            anchor_global_ids = node_ids[:batch_size]
            anchor_times = time_attr[anchor_global_ids]
            batch_node_times = time_attr[node_ids]

            for i in range(batch_size):
                # Find nodes published before this anchor
                valid_mask = batch_node_times <= anchor_times[i]
                valid_mask[i] = False  # Exclude self

                # Prefer non-anchor nodes if available
                if embeddings.size(0) > batch_size:
                    valid_mask[:batch_size] = False

                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) >= self.num_hard_negatives:
                    # Enough valid candidates
                    selected = valid_indices[
                        torch.randperm(len(valid_indices))[: self.num_hard_negatives]
                    ]
                    hard_neg_indices[i] = selected
                elif len(valid_indices) > 0:
                    # Some valid candidates, sample with replacement
                    hard_neg_indices[i] = valid_indices[
                        torch.randint(0, len(valid_indices), (self.num_hard_negatives,))
                    ]
                else:
                    # No valid candidates (edge case), fall back to any non-self
                    fallback_mask = torch.ones(
                        embeddings.size(0), dtype=torch.bool, device=self.device
                    )
                    fallback_mask[i] = False
                    fallback_indices = torch.where(fallback_mask)[0]
                    if len(fallback_indices) > 0:
                        hard_neg_indices[i] = fallback_indices[
                            torch.randint(
                                0, len(fallback_indices), (self.num_hard_negatives,)
                            )
                        ]
                    else:
                        # Absolute fallback (single node case)
                        hard_neg_indices[i] = 0
        else:
            # No temporal filtering, use original random sampling
            if embeddings.size(0) > batch_size:
                hard_neg_indices = torch.randint(
                    batch_size,
                    embeddings.size(0),
                    (batch_size, self.num_hard_negatives),
                    device=self.device,
                )
            else:
                hard_neg_indices = torch.randint(
                    0,
                    embeddings.size(0),
                    (batch_size, self.num_hard_negatives),
                    device=self.device,
                )

        return embeddings[hard_neg_indices]

    def _process_batch(
        self, batch, is_hetero, hard_negatives_tensor, graph_time_attr=None
    ):
        """Process batch - optimized version with temporal info extraction."""
        if is_hetero:
            batch_size = batch["paragraph"].batch_size
            x = batch["paragraph"].x.clone()

            if hasattr(batch["paragraph"], "x_query"):
                x[:batch_size] = batch["paragraph"].x_query[:batch_size]

            node_ids = (
                batch["paragraph"].n_id if hasattr(batch["paragraph"], "n_id") else None
            )
            cite_edge_index = batch.get(("paragraph", "cites", "paragraph"), {}).get(
                "edge_index", None
            )

            # Simplified masking
            masked_cite_edges = None
            if cite_edge_index is not None:
                src, tgt = cite_edge_index
                # Only mask edges between anchor nodes
                leakage_mask = (src < batch_size) & (tgt < batch_size)
                masked_cite_edges = cite_edge_index[:, ~leakage_mask]

            modified_batch = batch.clone()
            if (
                masked_cite_edges is not None
                and ("paragraph", "cites", "paragraph") in batch.edge_types
            ):
                modified_batch["paragraph", "cites", "paragraph"].edge_index = (
                    masked_cite_edges
                )
            modified_batch["paragraph"].x = x

            return {
                "batch_size": batch_size,
                "modified_batch": modified_batch,
                "edge_index": cite_edge_index,
                "x": x,
                "node_ids": node_ids,
            }
        else:
            batch_size = batch.batch_size
            x = batch.x.clone()

            if hasattr(batch, "x_query"):
                x[:batch_size] = batch.x_query[:batch_size]

            node_ids = batch.n_id if hasattr(batch, "n_id") else None
            edge_index = batch.edge_index

            # Mask edges
            src, tgt = edge_index
            leakage_mask = (src < batch_size) & (tgt < batch_size)
            masked_edge_index = edge_index[:, ~leakage_mask]

            return {
                "batch_size": batch_size,
                "modified_batch": None,
                "edge_index": edge_index,
                "x": x,
                "masked_edge_index": masked_edge_index,
                "node_ids": node_ids,
            }

    def _compute_loss(
        self, model, batch_data, is_hetero, hard_negatives_tensor, time_attr=None
    ):
        """Optimized loss computation with configurable hard negative sampling range and temporal filtering."""
        batch_size = batch_data["batch_size"]
        edge_index = batch_data["edge_index"]
        node_ids = batch_data["node_ids"]

        # Get embeddings
        if is_hetero:
            out = model(batch_data["modified_batch"])
            embeddings = out["paragraph"] if isinstance(out, dict) else out
        else:
            out = model(batch_data["x"], batch_data["masked_edge_index"])
            embeddings = out["paragraph"] if isinstance(out, dict) else out

        anchor_emb = embeddings[:batch_size]

        # Get positive samples efficiently
        positive_indices = torch.arange(batch_size, device=self.device)

        if edge_index is not None and edge_index.numel() > 0:
            src, tgt = edge_index
            input_mask = src < batch_size

            if input_mask.sum() > 0:
                batch_src = src[input_mask]
                batch_tgt = tgt[input_mask]

                # Use scatter to efficiently select one positive per anchor
                # Group by source and take first occurrence
                unique_src, inverse = torch.unique(batch_src, return_inverse=True)

                # For each unique source, find first target
                first_occurrence = torch.zeros_like(unique_src)
                for i, src_idx in enumerate(unique_src):
                    mask = batch_src == src_idx
                    first_occurrence[i] = batch_tgt[mask][0]

                positive_indices[unique_src] = first_occurrence

        positive_emb = embeddings[positive_indices]

        # Optimized hard negative sampling with configurable range and temporal filtering
        if node_ids is not None and hard_negatives_tensor is not None:
            anchor_global_ids = node_ids[:batch_size]

            # Calculate the valid sampling range
            start_idx = self.hard_negative_start_rank
            end_idx = min(self.hard_negative_end_rank, hard_negatives_tensor.size(1))
            sampling_range = end_idx - start_idx

            if sampling_range <= 0:
                # Fallback to random negatives if range is invalid
                print("Warning: Invalid sampling range, using random negatives")
                hard_neg_emb = self._sample_random_negatives_with_temporal_filter(
                    embeddings, batch_size, node_ids, time_attr
                )
            else:
                # Sample from the specified rank range (e.g., ranks 100-300)
                num_to_sample = min(self.num_hard_negatives, sampling_range)

                # Random sampling from the specified range for all anchors at once
                random_indices = torch.randint(
                    start_idx,
                    end_idx,
                    (batch_size, num_to_sample),
                    device=self.device,
                )

                # Gather hard negative global IDs from the specified range
                hard_neg_global_ids = torch.gather(
                    hard_negatives_tensor[anchor_global_ids], 1, random_indices
                )  # [batch_size, num_hard_negatives]

                # TEMPORAL FILTERING: Only keep hard negatives published before anchor
                if time_attr is not None and node_ids is not None:
                    # Get anchor times
                    anchor_times = time_attr[anchor_global_ids]  # [batch_size]

                    # Get hard negative times
                    hard_neg_times = time_attr[
                        hard_neg_global_ids
                    ]  # [batch_size, num_hard_negatives]

                    # Mask: keep only negatives where neg_time <= anchor_time
                    temporal_valid = hard_neg_times <= anchor_times.unsqueeze(
                        1
                    )  # [batch_size, num_hard_negatives]

                    # For invalid (future) negatives, we'll need to replace them
                    # Count valid negatives per anchor
                    num_valid = temporal_valid.sum(dim=1)  # [batch_size]

                    # If some anchors don't have enough valid negatives, we need to handle this
                    min_valid = num_valid.min().item()
                    if min_valid == 0:
                        # Some anchors have no valid hard negatives from the precomputed list
                        # Fall back to sampling from in-batch negatives with temporal filter
                        hard_neg_emb = (
                            self._sample_random_negatives_with_temporal_filter(
                                embeddings, batch_size, node_ids, time_attr
                            )
                        )
                    else:
                        # We have at least some valid negatives for each anchor
                        # For simplicity, take the first k valid ones
                        # Create a list to store final hard negative IDs
                        final_hard_neg_ids = torch.zeros(
                            (batch_size, num_to_sample),
                            dtype=torch.long,
                            device=self.device,
                        )

                        for i in range(batch_size):
                            valid_mask = temporal_valid[i]
                            valid_ids = hard_neg_global_ids[i][valid_mask]

                            if len(valid_ids) >= num_to_sample:
                                # We have enough valid negatives
                                final_hard_neg_ids[i] = valid_ids[:num_to_sample]
                            else:
                                # Not enough valid negatives, use what we have and pad with in-batch
                                final_hard_neg_ids[i, : len(valid_ids)] = valid_ids

                                # Fill remaining with random in-batch negatives (with temporal filter)
                                num_needed = num_to_sample - len(valid_ids)
                                anchor_time = anchor_times[i]

                                # Find in-batch nodes published before this anchor
                                batch_node_times = time_attr[node_ids]
                                valid_batch_mask = batch_node_times <= anchor_time

                                # Exclude the anchor itself and positives
                                valid_batch_mask[i] = False  # Exclude self
                                if edge_index is not None and edge_index.numel() > 0:
                                    # Also exclude positive samples (rough approximation)
                                    valid_batch_mask[:batch_size] = False

                                valid_batch_indices = torch.where(valid_batch_mask)[0]

                                if len(valid_batch_indices) >= num_needed:
                                    random_fill = valid_batch_indices[
                                        torch.randperm(len(valid_batch_indices))[
                                            :num_needed
                                        ]
                                    ]
                                    final_hard_neg_ids[i, len(valid_ids) :] = node_ids[
                                        random_fill
                                    ]
                                else:
                                    # Not enough valid in-batch either, use what we have
                                    if len(valid_batch_indices) > 0:
                                        final_hard_neg_ids[
                                            i,
                                            len(valid_ids) : len(valid_ids)
                                            + len(valid_batch_indices),
                                        ] = node_ids[valid_batch_indices]

                        hard_neg_global_ids = final_hard_neg_ids

                # Create mapping from global ID to batch position (vectorized)
                batch_positions = torch.arange(embeddings.size(0), device=self.device)

                # For each hard negative, find its position in the batch
                matches = hard_neg_global_ids.unsqueeze(-1) == node_ids.unsqueeze(
                    0
                ).unsqueeze(0)

                # Get first match for each (should only be one)
                hard_neg_batch_indices = matches.long().argmax(dim=-1)

                # Handle cases where hard negative is not in batch (use random fallback with temporal filter)
                not_found = ~matches.any(dim=-1)
                if not_found.any():
                    # Use temporally valid random nodes from the batch as fallback
                    if time_attr is not None and node_ids is not None:
                        for i in range(batch_size):
                            for j in range(num_to_sample):
                                if not_found[i, j]:
                                    # Find valid candidates for this anchor
                                    anchor_time = time_attr[anchor_global_ids[i]]
                                    batch_node_times = time_attr[node_ids]
                                    valid_mask = batch_node_times <= anchor_time
                                    valid_mask[i] = False  # Exclude self
                                    if embeddings.size(0) > batch_size:
                                        valid_mask[:batch_size] = (
                                            False  # Prefer non-anchor nodes
                                        )

                                    valid_indices = torch.where(valid_mask)[0]
                                    if len(valid_indices) > 0:
                                        hard_neg_batch_indices[i, j] = valid_indices[
                                            torch.randint(0, len(valid_indices), (1,))
                                        ].item()
                                    else:
                                        # Absolute fallback: any non-self node
                                        fallback_idx = (
                                            (i + 1) % embeddings.size(0)
                                            if embeddings.size(0) > 1
                                            else 0
                                        )
                                        hard_neg_batch_indices[i, j] = fallback_idx
                    else:
                        # No temporal info, use original random fallback
                        if embeddings.size(0) > batch_size:
                            random_fallback = torch.randint(
                                batch_size,
                                embeddings.size(0),
                                (not_found.sum(),),
                                device=self.device,
                            )
                        else:
                            random_fallback = torch.randint(
                                0,
                                embeddings.size(0),
                                (not_found.sum(),),
                                device=self.device,
                            )
                        hard_neg_batch_indices[not_found] = random_fallback

                # Gather hard negative embeddings
                hard_neg_emb = embeddings[hard_neg_batch_indices]
        else:
            # Fallback: random negatives with temporal filtering if available
            hard_neg_emb = self._sample_random_negatives_with_temporal_filter(
                embeddings, batch_size, node_ids, time_attr
            )

        # Compute loss
        loss = info_nce_loss_fast(
            anchor_emb, positive_emb, hard_neg_emb, self.temperature
        )
        return loss

    def train_epoch(
        self, model, loader, optimizer, is_hetero, hard_negatives_tensor, time_attr=None
    ):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            batch_data = self._process_batch(
                batch, is_hetero, hard_negatives_tensor, time_attr
            )
            if batch_data is None:
                continue

            loss = self._compute_loss(
                model, batch_data, is_hetero, hard_negatives_tensor, time_attr
            )
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def validate(self, model, loader, is_hetero, hard_negatives_tensor, time_attr=None):
        """Validate the model."""
        model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = batch.to(self.device)
            batch_data = self._process_batch(
                batch, is_hetero, hard_negatives_tensor, time_attr
            )
            if batch_data is None:
                continue

            loss = self._compute_loss(
                model, batch_data, is_hetero, hard_negatives_tensor, time_attr
            )
            if loss is None:
                continue

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self,
        gnn_model: nn.Module,
        train_cutoff_year: int | None = None,
        val_cutoff_year: int | None = None,
    ) -> torch.nn.Module:
        """Train GNN model."""
        os.makedirs(self.output_path, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Training GNN with {self.graph_type.capitalize()} Graph")
        print(f"Include only citing: {self.include_only_citing}")
        print(f"Temporal filtering for hard negatives: ENABLED")
        print(
            f"Hard negatives: {self.num_hard_negatives} sampled from ranks "
            f"{self.hard_negative_start_rank}-{self.hard_negative_end_rank}"
        )
        print("=" * 80)

        is_hetero = self.graph_type == "heterogeneous"

        # Build graphs
        if is_hetero:
            builder = HeterogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=self.include_only_citing,
            )

            # Extract time attribute for temporal filtering
            time_attr = (
                train_graph_data["paragraph"].time
                if hasattr(train_graph_data["paragraph"], "time")
                else None
            )

            # Compute hard negatives WITH temporal filtering
            train_hard_negatives = self._compute_hard_negatives(
                train_graph_data["paragraph"].x,
                train_graph_data.get(("paragraph", "cites", "paragraph"), {}).get(
                    "edge_index", None
                ),
                train_cutoff_year,
                time_attr=time_attr,
            )

            train_graph_data = train_graph_data.to(self.device)

            if val_cutoff_year is not None:
                val_graph_data = builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=self.include_only_citing,
                )
                val_time_attr = (
                    val_graph_data["paragraph"].time
                    if hasattr(val_graph_data["paragraph"], "time")
                    else None
                )
                val_hard_negatives = self._compute_hard_negatives(
                    val_graph_data["paragraph"].x,
                    val_graph_data.get(("paragraph", "cites", "paragraph"), {}).get(
                        "edge_index", None
                    ),
                    val_cutoff_year,
                    time_attr=val_time_attr,
                )
                val_graph_data = val_graph_data.to(self.device)
            else:
                val_graph_data = None
                val_hard_negatives = None
                val_time_attr = None

            input_nodes = ("paragraph", None)
        else:
            builder = HomogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=self.include_only_citing,
            )

            # Extract time attribute for temporal filtering
            time_attr = (
                train_graph_data.time if hasattr(train_graph_data, "time") else None
            )

            train_hard_negatives = self._compute_hard_negatives(
                train_graph_data.x,
                (
                    train_graph_data.edge_index
                    if train_graph_data.edge_index.numel() > 0
                    else None
                ),
                train_cutoff_year,
                time_attr=time_attr,
            )

            train_graph_data = train_graph_data.to(self.device)

            if val_cutoff_year is not None:
                val_graph_data = builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=self.include_only_citing,
                )
                val_time_attr = (
                    val_graph_data.time if hasattr(val_graph_data, "time") else None
                )
                val_hard_negatives = self._compute_hard_negatives(
                    val_graph_data.x,
                    (
                        val_graph_data.edge_index
                        if val_graph_data.edge_index.numel() > 0
                        else None
                    ),
                    val_cutoff_year,
                    time_attr=val_time_attr,
                )
                val_graph_data = val_graph_data.to(self.device)
            else:
                val_graph_data = None
                val_hard_negatives = None
                val_time_attr = None

            input_nodes = None

        # Initialize model
        print("\nInitializing model...")
        model = gnn_model.to(self.device)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Create loaders
        num_neighbors = [-1] * (self.num_hops + 1) if self.num_hops > 0 else [-1]

        train_loader = NeighborLoader(
            train_graph_data,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=True,
            time_attr="time",
            subgraph_type="bidirectional",
            num_workers=0,  # Set to 0 to avoid multiprocessing overhead
        )

        val_loader = None
        if val_graph_data is not None:
            val_loader = NeighborLoader(
                val_graph_data,
                num_neighbors=num_neighbors,
                batch_size=self.batch_size,
                input_nodes=input_nodes,
                shuffle=False,
                time_attr="time",
                subgraph_type="bidirectional",
                num_workers=0,
            )

        print(f"\nTraining for {self.epochs} epochs...")

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(
                model,
                train_loader,
                optimizer,
                is_hetero,
                train_hard_negatives,
                time_attr,
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss = self.validate(
                    model, val_loader, is_hetero, val_hard_negatives, val_time_attr
                )
                print(f"  Val Loss:   {val_loss:.4f}")

                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(f"  ✓ Improved by {improvement:.4f}")
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(
                        f"  ✗ No improvement ({epochs_without_improvement}/{self.patience})"
                    )

                if epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break
            else:
                if train_loss < best_val_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(f"  ✓ New best: {train_loss:.4f}")
                    best_val_loss = train_loss

            scheduler.step()

        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")
        model.load_state_dict(torch.load(f"{self.output_path}/best_model.pt"))

        print(f"\nTraining complete! Best loss: {best_val_loss:.4f}")
        return model

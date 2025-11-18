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


def info_nce_loss_combined(
    anchor, positive, in_batch_negatives, hard_negatives=None, temperature=0.07
):
    """
    Contrastive loss combining all in-batch negatives with shared hard negatives.

    Args:
        anchor: [batch_size, dim]
        positive: [batch_size, dim]
        in_batch_negatives: [batch_size, dim] - all anchors (we'll mask self/positive)
        hard_negatives: [num_hard, dim] - shared hard negatives for all anchors
        temperature: temperature parameter

    Returns:
        loss: scalar
    """
    batch_size = anchor.size(0)
    device = anchor.device

    # Normalize all embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    in_batch_negatives = F.normalize(in_batch_negatives, dim=-1)

    # Positive similarities: [batch_size, 1]
    pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / temperature

    # In-batch negative similarities: [batch_size, batch_size]
    # Each anchor compares against all in-batch samples
    in_batch_sim = torch.mm(anchor, in_batch_negatives.t()) / temperature

    # Mask out self-similarities (diagonal)
    mask = torch.eye(batch_size, dtype=torch.bool, device=device)
    in_batch_sim = in_batch_sim.masked_fill(mask, -1e9)

    # Hard negative similarities (shared across all anchors)
    if hard_negatives is not None and hard_negatives.size(0) > 0:
        hard_negatives = F.normalize(hard_negatives, dim=-1)
        # [batch_size, num_hard]
        hard_neg_sim = torch.mm(anchor, hard_negatives.t()) / temperature

        # Concatenate: [batch_size, 1 + batch_size + num_hard]
        # (positive, in-batch negatives, hard negatives)
        logits = torch.cat([pos_sim, in_batch_sim, hard_neg_sim], dim=1)
    else:
        # Just positives and in-batch negatives
        logits = torch.cat([pos_sim, in_batch_sim], dim=1)

    # Labels: positive is always at index 0
    labels = torch.zeros(batch_size, dtype=torch.long, device=device)

    return F.cross_entropy(logits, labels)


# MAP: 0.5630222945192006
# Recall@5: 0.7196769491781009
# Recall@10: 0.8077791927893228
# Recall@100: 0.9582581336365732


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
        # Hard negative parameters
        num_hard_negatives: int = 1,  # Shared across all anchors in batch
        hard_negative_pool_size: int = 300,
        hard_negative_start_rank: int = 100,
        hard_negative_end_rank: int = 300,
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
            f"Shared hard negatives: {num_hard_negatives} sampled from ranks "
            f"{hard_negative_start_rank}-{self.hard_negative_end_rank}"
        )
        print(f"In-batch negatives: all (batch_size - 1 = {batch_size - 1})")
        print(f"Total negatives per anchor: ~{batch_size - 1 + num_hard_negatives}")
        print(f"Include only citing: {include_only_citing}")

    def _get_hard_negatives_cache_path(
        self, num_nodes, train_cutoff_year, include_only_citing
    ):
        """Generate cache path for hard negatives based on configuration."""
        cache_dir = Path(self.preprocessed_dir) / "hard_negatives_cache"
        cache_dir.mkdir(exist_ok=True)

        config_str = f"{num_nodes}_{train_cutoff_year}_{include_only_citing}_{self.hard_negative_pool_size}_{self.hard_negative_start_rank}_{self.hard_negative_end_rank}_{self.num_hard_negatives}_{self.graph_type}_presampled"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        cache_file = cache_dir / f"hard_neg_{config_hash}.pt"
        return cache_file

    def _compute_hard_negatives(
        self, embeddings, edge_index=None, train_cutoff_year=None, time_attr=None
    ):
        """
        Pre-compute hard negatives based on initial embeddings.
        Uses caching to avoid recomputation.
        """
        num_nodes = embeddings.shape[0]

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

        if time_attr is not None:
            time_attr = time_attr.to(self.device)

        # Build positive edge lookup
        positive_edges = None
        if edge_index is not None and edge_index.numel() > 0:
            print("Building positive pairs index...")
            edge_index = edge_index.to(self.device)
            num_edges = edge_index.size(1)

            positive_edges = {}
            src = edge_index[0].cpu()
            tgt = edge_index[1].cpu()

            for s, t in zip(src.tolist(), tgt.tolist()):
                if s not in positive_edges:
                    positive_edges[s] = []
                positive_edges[s].append(t)

            for s, t in zip(src.tolist(), tgt.tolist()):
                if t not in positive_edges:
                    positive_edges[t] = []
                positive_edges[t].append(s)

            for node in positive_edges:
                positive_edges[node] = torch.tensor(
                    list(set(positive_edges[node])), device=self.device
                )

            print(f"Positive pairs: {num_edges} edges (bidirectional index built)")

        # Compute hard negatives in batches
        hard_neg_list = []
        batch_size = 500

        print("Computing similarity and selecting hard negatives...")
        for start_idx in tqdm(
            range(0, num_nodes, batch_size), desc="Processing batches"
        ):
            end_idx = min(start_idx + batch_size, num_nodes)
            current_batch_size = end_idx - start_idx
            batch_emb = embeddings_norm[start_idx:end_idx]

            sim = torch.mm(batch_emb, embeddings_norm.t())

            batch_indices = torch.arange(start_idx, end_idx, device=self.device)
            sim[torch.arange(current_batch_size, device=self.device), batch_indices] = (
                -1e9
            )

            if time_attr is not None:
                batch_times = time_attr[start_idx:end_idx]
                temporal_mask = time_attr.unsqueeze(0) > batch_times.unsqueeze(1)
                sim[temporal_mask] = -1e9

            if positive_edges is not None:
                for i in range(current_batch_size):
                    node_idx = start_idx + i
                    if node_idx in positive_edges:
                        positive_neighbors = positive_edges[node_idx]
                        sim[i, positive_neighbors] = -1e9

            k = min(self.hard_negative_pool_size, num_nodes - 1)
            _, topk_indices = torch.topk(sim, k=k, dim=-1)

            # Pre-sample N hard negatives from the specified rank range
            start_rank = self.hard_negative_start_rank
            end_rank = min(self.hard_negative_end_rank, k)

            if end_rank > start_rank:
                sampling_range = end_rank - start_rank
                num_to_sample = min(self.num_hard_negatives, sampling_range)

                # Random sample indices for each node in batch
                random_indices = torch.randint(
                    start_rank,
                    end_rank,
                    (current_batch_size, num_to_sample),
                    device=self.device,
                )

                # Gather the selected hard negatives
                selected_hard_negs = torch.gather(topk_indices, 1, random_indices)
                hard_neg_list.append(selected_hard_negs.cpu())
            else:
                # Fallback: just use top N
                hard_neg_list.append(topk_indices[:, : self.num_hard_negatives].cpu())

            if (start_idx // batch_size) % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        print("Concatenating results...")
        hard_negatives = torch.cat(hard_neg_list, dim=0)
        print(f"Computed hard negatives: {hard_negatives.shape}")

        print(f"Caching hard negatives to {cache_path}")
        torch.save(hard_negatives, cache_path)

        hard_negatives = hard_negatives.to(self.device)
        return hard_negatives

    def _process_batch(
        self, batch, is_hetero, hard_negatives_tensor, graph_time_attr=None
    ):
        """Process batch - optimized version."""
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

            masked_cite_edges = None
            if cite_edge_index is not None:
                src, tgt = cite_edge_index
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
        """
        Efficient loss computation using:
        - All in-batch samples as negatives (via matrix multiply)
        - Shared hard negatives for all anchors
        """
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

        # Get positive samples
        positive_indices = torch.arange(batch_size, device=self.device)

        if edge_index is not None and edge_index.numel() > 0:
            src, tgt = edge_index
            input_mask = src < batch_size

            if input_mask.sum() > 0:
                batch_src = src[input_mask]
                batch_tgt = tgt[input_mask]

                unique_src, inverse = torch.unique(batch_src, return_inverse=True)

                first_occurrence = torch.zeros_like(unique_src)
                for i, src_idx in enumerate(unique_src):
                    mask = batch_src == src_idx
                    first_occurrence[i] = batch_tgt[mask][0]

                positive_indices[unique_src] = first_occurrence

        positive_emb = embeddings[positive_indices]

        # ========== HARD NEGATIVES (pre-sampled, just look them up) ==========
        hard_neg_emb = None

        if (
            node_ids is not None
            and hard_negatives_tensor is not None
            and self.num_hard_negatives > 0
        ):
            anchor_global_ids = node_ids[:batch_size]

            # Get pre-sampled hard negative IDs for each anchor
            # hard_negatives_tensor is [num_nodes, num_hard_negatives]
            hard_neg_global_ids = hard_negatives_tensor[
                anchor_global_ids
            ]  # [batch_size, num_hard_negatives]

            # Flatten to get all hard negative IDs
            all_hard_neg_ids = (
                hard_neg_global_ids.flatten()
            )  # [batch_size * num_hard_negatives]
            unique_hard_neg_ids = torch.unique(all_hard_neg_ids)

            # Map to batch positions (find which are in the current batch)
            # Create lookup: global_id -> batch_position
            matches = unique_hard_neg_ids.unsqueeze(1) == node_ids.unsqueeze(
                0
            )  # [num_unique, batch_nodes]
            in_batch_mask = matches.any(dim=1)

            in_batch_hard_ids = unique_hard_neg_ids[in_batch_mask]

            if len(in_batch_hard_ids) > 0:
                # Get batch indices for in-batch hard negatives
                matches_in_batch = in_batch_hard_ids.unsqueeze(1) == node_ids.unsqueeze(
                    0
                )
                batch_indices = matches_in_batch.long().argmax(dim=1)
                hard_neg_emb = embeddings[batch_indices]
            else:
                # Fallback: use non-anchor nodes from batch
                if embeddings.size(0) > batch_size:
                    num_available = embeddings.size(0) - batch_size
                    num_to_use = min(self.num_hard_negatives, num_available)
                    indices = (
                        torch.randperm(num_available, device=self.device)[:num_to_use]
                        + batch_size
                    )
                    hard_neg_emb = embeddings[indices]

        # Compute loss with all in-batch negatives + hard negatives
        loss = info_nce_loss_combined(
            anchor_emb,
            positive_emb,
            positive_emb,  # Positives as in-batch negatives (standard approach)
            hard_neg_emb,  # Shared hard negatives
            self.temperature,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        print(f"Temporal filtering: ENABLED")
        print(f"In-batch negatives: ALL (batch_size - 1)")
        print(
            f"Shared hard negatives: {self.num_hard_negatives} from ranks "
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

            time_attr = (
                train_graph_data["paragraph"].time
                if hasattr(train_graph_data["paragraph"], "time")
                else None
            )

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
            num_workers=0,
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

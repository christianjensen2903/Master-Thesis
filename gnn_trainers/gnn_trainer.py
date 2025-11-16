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
        hard_negative_pool_size: int = 100,
        include_only_citing: bool = True,  # Add option to control this
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
        self.include_only_citing = include_only_citing

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")
        print(
            f"Hard negatives: {num_hard_negatives} from pool of {hard_negative_pool_size}"
        )
        print(f"Include only citing: {include_only_citing}")

    def _get_hard_negatives_cache_path(
        self, num_nodes, train_cutoff_year, include_only_citing
    ):
        """Generate cache path for hard negatives based on configuration."""
        # Create cache directory
        cache_dir = Path(self.preprocessed_dir) / "hard_negatives_cache"
        cache_dir.mkdir(exist_ok=True)

        # Create unique hash based on configuration
        config_str = f"{num_nodes}_{train_cutoff_year}_{include_only_citing}_{self.hard_negative_pool_size}_{self.graph_type}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        cache_file = cache_dir / f"hard_neg_{config_hash}.pt"
        return cache_file

    def _compute_hard_negatives(
        self, embeddings, edge_index=None, train_cutoff_year=None
    ):
        """
        Pre-compute hard negatives based on initial embeddings.
        Uses caching to avoid recomputation.
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
        embeddings = embeddings.to(self.device)
        embeddings_norm = F.normalize(embeddings, dim=-1)

        # Build set of positive pairs (citations)
        positive_pairs = set()
        if edge_index is not None:
            edge_index = edge_index.to(self.device)
            # Convert to set of tuples for fast lookup
            edges = edge_index.t().cpu().numpy()
            positive_pairs = set(map(tuple, edges))

        # Compute hard negatives in batches
        hard_neg_list = []
        batch_size = 1000

        for start_idx in tqdm(
            range(0, num_nodes, batch_size), desc="Computing similarity"
        ):
            end_idx = min(start_idx + batch_size, num_nodes)
            batch_emb = embeddings_norm[start_idx:end_idx]

            # Compute similarity with all nodes
            sim = torch.mm(batch_emb, embeddings_norm.t())

            # Mask out self and positives efficiently
            for i in range(batch_emb.shape[0]):
                node_idx = start_idx + i
                sim[i, node_idx] = -1e9  # Self

                # Mask positives - check both directions
                for j in range(num_nodes):
                    if (node_idx, j) in positive_pairs or (
                        j,
                        node_idx,
                    ) in positive_pairs:
                        sim[i, j] = -1e9

            # Get top-k
            _, topk_indices = torch.topk(
                sim, k=min(self.hard_negative_pool_size, num_nodes - 1), dim=-1
            )
            hard_neg_list.append(topk_indices)

        hard_negatives = torch.cat(hard_neg_list, dim=0)
        print(f"Computed hard negatives: {hard_negatives.shape}")

        # Cache to disk
        print(f"Caching hard negatives to {cache_path}")
        torch.save(hard_negatives, cache_path)

        return hard_negatives  # Keep on GPU

    def _process_batch(self, batch, is_hetero, hard_negatives_tensor):
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

    def _compute_loss(self, model, batch_data, is_hetero, hard_negatives_tensor):
        """Optimized loss computation."""
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

        # Optimized hard negative sampling
        if node_ids is not None and hard_negatives_tensor is not None:
            anchor_global_ids = node_ids[:batch_size]

            # Vectorized lookup of hard negatives
            # Sample indices efficiently
            num_to_sample = min(self.num_hard_negatives, hard_negatives_tensor.size(1))

            # Random sampling from pool for all anchors at once
            random_indices = torch.randint(
                0,
                hard_negatives_tensor.size(1),
                (batch_size, num_to_sample),
                device=self.device,
            )

            # Gather hard negative global IDs
            hard_neg_global_ids = torch.gather(
                hard_negatives_tensor[anchor_global_ids], 1, random_indices
            )  # [batch_size, num_hard_negatives]

            # Create mapping from global ID to batch position (vectorized)
            # This is the key optimization - use broadcasting instead of loops
            batch_positions = torch.arange(embeddings.size(0), device=self.device)

            # For each hard negative, find its position in the batch
            # Use broadcasting: [batch_size, num_hard_neg, 1] vs [1, 1, num_nodes_in_batch]
            matches = hard_neg_global_ids.unsqueeze(-1) == node_ids.unsqueeze(
                0
            ).unsqueeze(0)

            # Get first match for each (should only be one)
            hard_neg_batch_indices = matches.long().argmax(dim=-1)

            # Handle cases where hard negative is not in batch (use random fallback)
            not_found = ~matches.any(dim=-1)
            if not_found.any():
                # Use random nodes from the batch as fallback
                # Fix: Handle case when batch_size >= embeddings.shape[0]
                if embeddings.size(0) > batch_size:
                    # Sample from nodes outside the anchor batch
                    random_fallback = torch.randint(
                        batch_size,
                        embeddings.size(0),
                        (not_found.sum(),),
                        device=self.device,
                    )
                else:
                    # All nodes are anchors, sample from any node
                    random_fallback = torch.randint(
                        0, embeddings.size(0), (not_found.sum(),), device=self.device
                    )
                hard_neg_batch_indices[not_found] = random_fallback

            # Gather hard negative embeddings
            hard_neg_emb = embeddings[hard_neg_batch_indices]
        else:
            # Fallback: random negatives
            # Fix: Handle case when batch_size >= embeddings.shape[0]
            if embeddings.size(0) > batch_size:
                # Sample from nodes outside anchor batch
                random_indices = torch.randint(
                    batch_size,
                    embeddings.size(0),
                    (batch_size, self.num_hard_negatives),
                    device=self.device,
                )
            else:
                # Sample from any nodes (including anchors)
                random_indices = torch.randint(
                    0,
                    embeddings.size(0),
                    (batch_size, self.num_hard_negatives),
                    device=self.device,
                )
            hard_neg_emb = embeddings[random_indices]

        # Compute loss
        loss = info_nce_loss_fast(
            anchor_emb, positive_emb, hard_neg_emb, self.temperature
        )
        return loss

    def train_epoch(self, model, loader, optimizer, is_hetero, hard_negatives_tensor):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Training", leave=False):
            batch = batch.to(self.device)
            batch_data = self._process_batch(batch, is_hetero, hard_negatives_tensor)
            if batch_data is None:
                continue

            loss = self._compute_loss(
                model, batch_data, is_hetero, hard_negatives_tensor
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
    def validate(self, model, loader, is_hetero, hard_negatives_tensor):
        """Validate the model."""
        model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Validation", leave=False):
            batch = batch.to(self.device)
            batch_data = self._process_batch(batch, is_hetero, hard_negatives_tensor)
            if batch_data is None:
                continue

            loss = self._compute_loss(
                model, batch_data, is_hetero, hard_negatives_tensor
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
        print(f"Hard negatives: {self.num_hard_negatives}")
        print("=" * 80)

        is_hetero = self.graph_type == "heterogeneous"

        # Build graphs
        if is_hetero:
            builder = HeterogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=self.include_only_citing,
            )

            # Compute hard negatives
            train_hard_negatives = self._compute_hard_negatives(
                train_graph_data["paragraph"].x,
                train_graph_data.get(("paragraph", "cites", "paragraph"), {}).get(
                    "edge_index", None
                ),
                train_cutoff_year,
            )

            train_graph_data = train_graph_data.to(self.device)

            if val_cutoff_year is not None:
                val_graph_data = builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=self.include_only_citing,
                )
                val_hard_negatives = self._compute_hard_negatives(
                    val_graph_data["paragraph"].x,
                    val_graph_data.get(("paragraph", "cites", "paragraph"), {}).get(
                        "edge_index", None
                    ),
                    val_cutoff_year,
                )
                val_graph_data = val_graph_data.to(self.device)
            else:
                val_graph_data = None
                val_hard_negatives = None

            input_nodes = ("paragraph", None)
        else:
            builder = HomogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=self.include_only_citing,
            )

            train_hard_negatives = self._compute_hard_negatives(
                train_graph_data.x,
                (
                    train_graph_data.edge_index
                    if train_graph_data.edge_index.numel() > 0
                    else None
                ),
                train_cutoff_year,
            )

            train_graph_data = train_graph_data.to(self.device)

            if val_cutoff_year is not None:
                val_graph_data = builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=self.include_only_citing,
                )
                val_hard_negatives = self._compute_hard_negatives(
                    val_graph_data.x,
                    (
                        val_graph_data.edge_index
                        if val_graph_data.edge_index.numel() > 0
                        else None
                    ),
                    val_cutoff_year,
                )
                val_graph_data = val_graph_data.to(self.device)
            else:
                val_graph_data = None
                val_hard_negatives = None

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
                model, train_loader, optimizer, is_hetero, train_hard_negatives
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            if val_loader is not None:
                val_loss = self.validate(
                    model, val_loader, is_hetero, val_hard_negatives
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

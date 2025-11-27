import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.data import Data, HeteroData  # type: ignore
from tqdm import tqdm  # type: ignore
import wandb  # type: ignore
from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
)
import math
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.transforms import ToUndirected


def info_nce_loss(
    anchor,
    positive,
    temperature=0.07,
    anchor_times=None,
    positive_times=None,
    anchor_indices=None,
    positive_indices=None,
    hard_negatives=None,
    hard_negative_times=None,
    return_stats=False,
):
    """
    In-batch negative contrastive loss with temporal filtering, False Negative masking, and hard negatives.

    anchor: [batch_size, dim]
    positive: [batch_size, dim]
    anchor_indices: [batch_size] - IDs of source nodes
    positive_indices: [batch_size] - IDs of target nodes
    hard_negatives: [num_hard_negatives, dim] - Hard negative embeddings shared across all anchors
    hard_negative_times: [num_hard_negatives] - Timestamps for hard negatives (for temporal filtering)
    return_stats: if True, return (loss, stats_dict), else return loss only
    """
    # Compute similarity matrix using unnormalized dot product: [batch_size, batch_size]
    sim_matrix = torch.mm(anchor, positive.t()) / temperature

    # If hard negatives provided, compute similarities and concatenate
    num_hard_negatives = 0
    if hard_negatives is not None and hard_negatives.size(0) > 0:
        num_hard_negatives = hard_negatives.size(0)
        hard_neg_sim = (
            torch.mm(anchor, hard_negatives.t()) / temperature
        )  # [batch_size, num_hard_negatives]
        sim_matrix = torch.cat(
            [sim_matrix, hard_neg_sim], dim=1
        )  # [batch_size, batch_size + num_hard_negatives]

    batch_size = sim_matrix.size(0)
    total_cols = sim_matrix.size(1)  # batch_size + num_hard_negatives

    # Diagonal mask for the in-batch positives part only
    diagonal_mask = torch.eye(
        batch_size, total_cols, dtype=torch.bool, device=sim_matrix.device
    )

    # Initialize mask for False Negatives (True = mask out/ignore)
    false_negative_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)

    if anchor_indices is not None and positive_indices is not None:
        # same_anchor[i,k] = True if anchor_i == anchor_k
        same_anchor = anchor_indices.unsqueeze(1) == anchor_indices.unsqueeze(0)
        # same_target[k,j] = True if positive_k == positive_j
        same_target = positive_indices.unsqueeze(1) == positive_indices.unsqueeze(0)

        # Mask (i,j) if there exists ANY k where:
        #   anchor_i == anchor_k  AND  positive_k == positive_j
        # This means positive_j is a true positive for anchor_i
        fn_mask_in_batch = (same_anchor.float() @ same_target.float()) > 0

        # Pad with zeros for hard negatives (they are pre-filtered to not be false negatives)
        if num_hard_negatives > 0:
            hard_neg_fn_mask = torch.zeros(
                batch_size,
                num_hard_negatives,
                dtype=torch.bool,
                device=sim_matrix.device,
            )
            false_negative_mask = torch.cat([fn_mask_in_batch, hard_neg_fn_mask], dim=1)
        else:
            false_negative_mask = fn_mask_in_batch

    # Ensure diagonal is NOT masked (we need it for the loss)
    final_mask = false_negative_mask & ~diagonal_mask

    # Apply mask: set false negatives to -inf so Softmax ignores them
    sim_matrix = sim_matrix.masked_fill(final_mask, float("-inf"))

    # Apply temporal masking if time information is provided
    if anchor_times is not None and positive_times is not None:
        # positive_j is valid negative for anchor_i only if positive_time_j < anchor_time_i
        time_mask = positive_times.unsqueeze(0) < anchor_times.unsqueeze(1)

        # Extend time mask for hard negatives if provided
        if num_hard_negatives > 0 and hard_negative_times is not None:
            hard_neg_time_mask = hard_negative_times.unsqueeze(
                0
            ) < anchor_times.unsqueeze(1)
            time_mask = torch.cat([time_mask, hard_neg_time_mask], dim=1)
        elif num_hard_negatives > 0:
            # No time info for hard negatives, assume they're all valid
            hard_neg_time_mask = torch.ones(
                batch_size,
                num_hard_negatives,
                dtype=torch.bool,
                device=sim_matrix.device,
            )
            time_mask = torch.cat([time_mask, hard_neg_time_mask], dim=1)

        # Ensure diagonal (positive pairs) are always valid
        time_mask = time_mask | diagonal_mask

        # Apply mask: set invalid negatives to very low value
        sim_matrix = sim_matrix.masked_fill(~time_mask, float("-inf"))

    # Labels: for each anchor_i, the positive is at position i (diagonal)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    loss = F.cross_entropy(sim_matrix, labels)

    if not return_stats:
        return loss

    # Compute statistics for monitoring
    stats = {}
    stats["num_hard_negatives"] = num_hard_negatives

    # Get positive similarities (diagonal)
    positive_sims = torch.diagonal(sim_matrix)
    stats["pos_sim_mean"] = positive_sims.mean().item()
    stats["pos_sim_std"] = positive_sims.std().item()
    stats["pos_sim_min"] = positive_sims.min().item()
    stats["pos_sim_max"] = positive_sims.max().item()

    # Get negative similarities (off-diagonal valid entries)
    # Create a mask for valid negatives (not masked to -inf)
    valid_mask = ~torch.isinf(sim_matrix) & ~diagonal_mask

    if valid_mask.any():
        negative_sims = sim_matrix[valid_mask]
        stats["neg_sim_mean"] = negative_sims.mean().item()
        stats["neg_sim_std"] = negative_sims.std().item()
        stats["neg_sim_max"] = negative_sims.max().item()

        # Number of valid negatives per sample
        num_valid_negatives = valid_mask.sum(dim=1).float()
        stats["num_negatives_mean"] = num_valid_negatives.mean().item()
        stats["num_negatives_min"] = num_valid_negatives.min().item()
        stats["num_negatives_max"] = num_valid_negatives.max().item()

        # Margin: difference between positive and max negative similarity
        max_neg_per_sample = torch.where(
            valid_mask,
            sim_matrix,
            torch.tensor(float("-inf"), device=sim_matrix.device),
        ).max(dim=1)[0]
        margin = positive_sims - max_neg_per_sample
        stats["margin_mean"] = margin.mean().item()
        stats["margin_min"] = margin.min().item()

        # Positive rank: where does the positive rank among all samples?
        # Lower is better (rank 1 means positive is the highest similarity)
        ranks = (sim_matrix > positive_sims.unsqueeze(1)).sum(dim=1) + 1
        stats["pos_rank_mean"] = ranks.float().mean().item()
        stats["pos_rank_median"] = ranks.float().median().item()

        # Accuracy@k: percentage of positives in top-k
        for k in [1, 5, 10]:
            if k <= sim_matrix.size(1):
                acc_at_k = (ranks <= k).float().mean().item()
                stats[f"acc@{k}"] = acc_at_k
    else:
        # No valid negatives case
        stats["neg_sim_mean"] = 0.0
        stats["neg_sim_std"] = 0.0
        stats["neg_sim_max"] = 0.0
        stats["num_negatives_mean"] = 0.0
        stats["num_negatives_min"] = 0.0
        stats["num_negatives_max"] = 0.0
        stats["margin_mean"] = 0.0
        stats["margin_min"] = 0.0
        stats["pos_rank_mean"] = 1.0
        stats["pos_rank_median"] = 1.0

    return loss, stats


class GNNTrainer:
    def __init__(
        self,
        preprocessed_dir: str,
        output_path: str = "output/gnn",
        batch_size: int = 16,
        epochs: int = 5,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_hops: int = 2,
        graph_type: str = "heterogeneous",
        checkpoint_interval: int = 25,
        wandb_project: str | None = "gnn-training",
        wandb_name: str | None = None,
        warmup_epochs: int = 3,
        eval_every_n_epochs: int = 1,
        gradient_clip_val: float | None = None,
        log_every_n_batches: int = 100,
        # Semantic edge parameters (for homogeneous graphs)
        include_semantic_edges: bool = False,
        semantic_threshold: float = 0.7,
        semantic_max_neighbors: int = 10,
        # Hard negative mining
        num_hard_negatives: int = 0,
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
        self.checkpoint_interval = checkpoint_interval
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.warmup_epochs = warmup_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_batches = log_every_n_batches
        # Semantic edge settings
        self.include_semantic_edges = include_semantic_edges
        self.semantic_threshold = semantic_threshold
        self.semantic_max_neighbors = semantic_max_neighbors
        # Hard negative mining
        self.num_hard_negatives = num_hard_negatives

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")
        if include_semantic_edges and graph_type == "homogeneous":
            print(
                f"  Semantic edges enabled (threshold={semantic_threshold}, max_neighbors={semantic_max_neighbors})"
            )
        if num_hard_negatives > 0:
            print(f"  Semantic hard negatives enabled (max={num_hard_negatives})")

    def _process_batch(
        self,
        batch,
        is_hetero: bool,
    ):
        """Process a batch and extract necessary components."""
        if is_hetero:
            batch_size = batch["paragraph"].batch_size
            x = batch["paragraph"].x.clone()

            if hasattr(batch["paragraph"], "x_query"):
                x[:batch_size] = batch["paragraph"].x_query[:batch_size]

            if hasattr(batch["paragraph"], "time"):
                anchor_times = batch["paragraph"].time[:batch_size]
            else:
                anchor_times = None

            if ("paragraph", "cites", "paragraph") in batch.edge_types:
                cite_edge_index = batch["paragraph", "cites", "paragraph"].edge_index
            else:
                return None

            # Mask citation edges to prevent leakage
            if ("paragraph", "belongs_to", "case") in batch.edge_types:
                par_to_case = batch["paragraph", "belongs_to", "case"].edge_index
                case_to_par = batch["case", "contains", "paragraph"].edge_index

                anchor_mask = par_to_case[0] < batch_size
                anchor_cases = par_to_case[1, anchor_mask].unique()

                case_mask = torch.isin(case_to_par[0], anchor_cases)
                paragraphs_in_anchor_cases = case_to_par[1, case_mask].unique()
            else:
                paragraphs_in_anchor_cases = torch.arange(
                    batch_size, device=self.device
                )

            cite_src, cite_tgt = cite_edge_index
            leakage_mask = torch.isin(
                cite_src, paragraphs_in_anchor_cases
            ) | torch.isin(cite_tgt, paragraphs_in_anchor_cases)
            masked_cite_edges = cite_edge_index[:, ~leakage_mask]

            modified_batch = batch.clone()
            modified_batch["paragraph", "cites", "paragraph"].edge_index = (
                masked_cite_edges
            )
            modified_batch["paragraph"].x = x

            return {
                "batch_size": batch_size,
                "modified_batch": modified_batch,
                "edge_index": cite_edge_index,
                "x": x,
                "anchor_times": anchor_times,
                "all_times": (
                    batch["paragraph"].time
                    if hasattr(batch["paragraph"], "time")
                    else None
                ),
            }

        else:
            # For homogeneous graphs
            batch_size = batch.batch_size
            x = batch.x.clone()

            if hasattr(batch, "x_query"):
                x[:batch_size] = batch.x_query[:batch_size]

            if hasattr(batch, "time"):
                anchor_times = batch.time[:batch_size]
            else:
                anchor_times = None

            edge_index = batch.edge_index
            edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None

            src, tgt = edge_index

            # Mask edges to prevent leakage:
            # 1. All outgoing edges from anchors (src < batch_size)
            # 2. Citation edges (type 0, 1) incoming to anchors (tgt < batch_size)
            # Keep semantic edges (type 2) incoming to anchors
            outgoing_from_anchor = src < batch_size
            incoming_to_anchor = tgt < batch_size

            if edge_attr is not None:
                # Citation edges (type 0=cites, 1=cited_by): mask both directions
                is_citation_edge = (edge_attr == 0) | (edge_attr == 1)
                # Semantic edges (type 2): only mask outgoing from anchor
                leakage_mask = outgoing_from_anchor | (
                    incoming_to_anchor & is_citation_edge
                )
            else:
                # No edge types, mask all edges involving anchors
                leakage_mask = outgoing_from_anchor | incoming_to_anchor

            keep_mask = ~leakage_mask
            masked_edge_index = edge_index[:, keep_mask]
            masked_edge_attr = edge_attr[keep_mask] if edge_attr is not None else None

            return {
                "batch_size": batch_size,
                "modified_batch": None,
                "edge_index": edge_index,
                "x": x,
                "masked_edge_index": masked_edge_index,
                "masked_edge_attr": masked_edge_attr,
                "edge_attr": edge_attr,
                "date_feature": batch.date_feature,
                "anchor_times": anchor_times,
                "all_times": batch.time if hasattr(batch, "time") else None,
            }

    def _compute_loss(
        self,
        model: nn.Module,
        batch_data: dict,
        is_hetero: bool,
        return_stats: bool = False,
    ):
        """Compute loss for a processed batch."""
        batch_size = batch_data["batch_size"]
        edge_index = batch_data["edge_index"]
        all_times = batch_data.get("all_times")

        # Get embeddings
        if is_hetero:
            out = model(batch_data["modified_batch"])
            embeddings = out["paragraph"] if isinstance(out, dict) else out
        else:
            date_feature = batch_data.get("date_feature")
            masked_edge_attr = batch_data.get("masked_edge_attr")
            out = model(
                batch_data["x"],
                batch_data["masked_edge_index"],
                date_feature=date_feature,
                edge_attr=masked_edge_attr,
            )
            embeddings = out["paragraph"] if isinstance(out, dict) else out

        # Find edges where source is in the input batch
        # We only want "cites" edges (edge_attr == 0) for training
        src, tgt = edge_index
        edge_attr = batch_data.get("edge_attr")

        if edge_attr is not None:
            # Only use forward citation edges (type 0 = "cites")
            cites_mask = edge_attr == 0
            input_mask = (src < batch_size) & cites_mask
        else:
            input_mask = src < batch_size

        if input_mask.sum() == 0:
            return (None, None) if return_stats else None

        batch_src = src[input_mask]
        batch_tgt = tgt[input_mask]

        anchor_emb = embeddings[batch_src]
        positive_emb = embeddings[batch_tgt]

        # Get times for all pairs
        pair_anchor_times = None
        pair_positive_times = None
        if all_times is not None:
            pair_anchor_times = all_times[batch_src]
            pair_positive_times = all_times[batch_tgt]

        # Extract hard negatives from semantic similarity edges if enabled
        hard_negatives = None
        hard_negative_times = None
        if self.num_hard_negatives > 0 and edge_attr is not None:
            hard_negatives, hard_negative_times = self._get_semantic_hard_negatives(
                embeddings=embeddings,
                edge_index=edge_index,
                edge_attr=edge_attr,
                batch_size=batch_size,
                positive_indices=batch_tgt,
                all_times=all_times,
            )

        # Compute loss with in-batch negatives
        result = info_nce_loss(
            anchor_emb,
            positive_emb,
            self.temperature,
            anchor_times=pair_anchor_times,
            positive_times=pair_positive_times,
            anchor_indices=batch_src,  # Mask Same Source
            positive_indices=batch_tgt,  # Mask Same Target
            hard_negatives=hard_negatives,
            hard_negative_times=hard_negative_times,
            return_stats=return_stats,
        )

        if return_stats:
            loss, stats = result
            # Add embedding statistics
            stats["emb_norm_mean"] = embeddings.norm(dim=1).mean().item()
            stats["emb_norm_std"] = embeddings.norm(dim=1).std().item()
            stats["num_pairs"] = input_mask.sum().item()
            return loss, stats
        else:
            return result

    def _get_semantic_hard_negatives(
        self,
        embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        batch_size: int,
        positive_indices: torch.Tensor,
        all_times: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Get hard negatives from semantic similarity edges (edge_attr == 2).

        These are nodes connected to anchors via pre-computed semantic similarity
        edges from the graph builder, excluding any positive targets.

        Returns hard negatives that will be used as negatives for ALL anchor-positive pairs.
        """
        # Find semantic edges (edge_attr == 2) where source is an anchor (< batch_size)
        # Semantic edges point FROM anchor TO similar node (src -> tgt)
        src, tgt = edge_index
        semantic_mask = edge_attr == 2
        anchor_mask = src < batch_size

        # Get semantic edges from anchors
        valid_mask = semantic_mask & anchor_mask
        if not valid_mask.any():
            return None, None

        # Get target nodes of semantic edges from anchors
        semantic_targets = tgt[valid_mask]

        # Exclude positive targets (we don't want false negatives)
        positive_set = positive_indices.unique()
        is_not_positive = ~torch.isin(semantic_targets, positive_set)
        semantic_targets = semantic_targets[is_not_positive]

        if len(semantic_targets) == 0:
            return None, None

        # Get unique hard negative indices
        unique_hard_neg_indices = semantic_targets.unique()

        # Limit to num_hard_negatives if specified
        if len(unique_hard_neg_indices) > self.num_hard_negatives:
            unique_hard_neg_indices = unique_hard_neg_indices[: self.num_hard_negatives]

        hard_negatives = embeddings[unique_hard_neg_indices]
        hard_negative_times = (
            all_times[unique_hard_neg_indices] if all_times is not None else None
        )

        return hard_negatives, hard_negative_times

    def train_epoch(
        self,
        model: nn.Module,
        loader: NeighborLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        is_hetero: bool,
        epoch: int = 0,
        global_batch_counter: int = 0,
    ) -> tuple[float, int]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        batch_counter = global_batch_counter

        # Accumulators for batch statistics
        batch_stats_accum = {}

        for batch_idx, batch in enumerate(
            tqdm(loader, desc="Training batches", leave=False)
        ):
            batch_data = self._process_batch(batch, is_hetero)
            if batch_data is None:
                continue

            # Get loss and stats
            result = self._compute_loss(model, batch_data, is_hetero, return_stats=True)
            if result[0] is None:
                continue

            loss, batch_stats = result

            optimizer.zero_grad()
            loss.backward()

            total_norm = 0.0
            max_grad = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    max_grad = max(max_grad, p.grad.data.abs().max().item())
            total_norm = total_norm**0.5

            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.gradient_clip_val
                )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            batch_counter += 1

            # Accumulate statistics
            for key, value in batch_stats.items():
                if key not in batch_stats_accum:
                    batch_stats_accum[key] = []
                batch_stats_accum[key].append(value)

            if (
                self.wandb_project is not None
                and batch_idx % self.log_every_n_batches == 0
            ):
                log_dict = {
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/batch": batch_counter,
                    "train/scheduler_lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": total_norm,
                    "train/max_grad": max_grad,
                }

                # Add batch statistics to wandb
                for key, value in batch_stats.items():
                    log_dict[f"train/{key}"] = value

                wandb.log(log_dict)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute epoch averages for statistics
        epoch_stats = {}
        for key, values in batch_stats_accum.items():
            if values:
                epoch_stats[key] = sum(values) / len(values)

        return avg_loss, batch_counter, epoch_stats

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        loader: NeighborLoader,
        is_hetero: bool,
    ) -> tuple[float, dict]:
        """Validate the model on validation set."""
        model.eval()
        total_loss = 0
        num_batches = 0

        # Accumulators for validation statistics
        val_stats_accum = {}

        for batch in tqdm(loader, desc="Validation batches", leave=False):
            batch_data = self._process_batch(batch, is_hetero)
            if batch_data is None:
                continue

            result = self._compute_loss(model, batch_data, is_hetero, return_stats=True)
            if result[0] is None:
                continue

            loss, batch_stats = result

            total_loss += loss.item()
            num_batches += 1

            # Accumulate statistics
            for key, value in batch_stats.items():
                if key not in val_stats_accum:
                    val_stats_accum[key] = []
                val_stats_accum[key].append(value)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

        # Compute validation averages for statistics
        val_stats = {}
        for key, values in val_stats_accum.items():
            if values:
                val_stats[key] = sum(values) / len(values)

        return avg_loss, val_stats

    def train(
        self,
        gnn_model: nn.Module,
        train_cutoff_year: int | None = None,
        val_cutoff_year: int | None = None,
    ) -> torch.nn.Module:
        """Train GNN model with optional validation."""
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Training GNN with {self.graph_type.capitalize()} Graph Builder")
        print("=" * 80)

        is_hetero = self.graph_type == "heterogeneous"

        # Build training graph
        if is_hetero:
            hetero_builder = HeterogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = hetero_builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
            ).to(self.device)
        else:
            homo_builder = HomogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = homo_builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
                add_reverse_edges=True,
                include_semantic_edges=self.include_semantic_edges,
                semantic_threshold=self.semantic_threshold,
                semantic_max_neighbors=self.semantic_max_neighbors,
            ).to(self.device)

        print("\nFiltering nodes with positive examples...")
        if is_hetero:
            cite_edge_index = train_graph_data[
                "paragraph", "cites", "paragraph"
            ].edge_index
            nodes_with_positives = cite_edge_index[0].unique()
            print(
                f"  Paragraph nodes with citations: {len(nodes_with_positives)} / {train_graph_data['paragraph'].num_nodes}"
            )
            input_nodes = ("paragraph", nodes_with_positives)
        else:
            edge_index = train_graph_data.edge_index
            edge_attr = train_graph_data.edge_attr
            # Only consider "cites" edges (type 0) for finding nodes with positives
            cites_mask = edge_attr == 0
            cites_edges = edge_index[:, cites_mask]
            nodes_with_positives = cites_edges[0].unique()
            print(
                f"  Nodes with citations: {len(nodes_with_positives)} / {train_graph_data.num_nodes}"
            )
            input_nodes = nodes_with_positives

        # Don't use ToUndirected for homogeneous graphs anymore since we handle it ourselves
        if is_hetero:
            train_graph_data = ToUndirected()(train_graph_data)

        # Build validation graph if val_cutoff_year is provided
        val_loader = None
        if val_cutoff_year is not None:
            print("\nBuilding validation graph...")
            if is_hetero:
                val_graph_data = hetero_builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=True,
                ).to(self.device)
                val_cite_edge_index = val_graph_data[
                    "paragraph", "cites", "paragraph"
                ].edge_index
                val_nodes_with_positives = val_cite_edge_index[0].unique()
                print(
                    f"  Val paragraph nodes with citations: {len(val_nodes_with_positives)} / {val_graph_data['paragraph'].num_nodes}"
                )
                val_input_nodes = ("paragraph", val_nodes_with_positives)
            else:
                val_graph_data = homo_builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=True,
                    add_reverse_edges=True,
                    include_semantic_edges=self.include_semantic_edges,
                    semantic_threshold=self.semantic_threshold,
                    semantic_max_neighbors=self.semantic_max_neighbors,
                ).to(self.device)
                val_edge_index = val_graph_data.edge_index
                val_edge_attr = val_graph_data.edge_attr
                val_cites_mask = val_edge_attr == 0
                val_cites_edges = val_edge_index[:, val_cites_mask]
                val_nodes_with_positives = val_cites_edges[0].unique()
                train_cutoff_time_stamp = homo_builder._date_to_timestamp(
                    f"{train_cutoff_year}-01-01"
                )
                node_times = val_graph_data.time[val_nodes_with_positives]
                time_mask = node_times > train_cutoff_time_stamp
                val_nodes_with_positives = val_nodes_with_positives[time_mask]
                print(
                    f"  Val nodes with citations: {len(val_nodes_with_positives)} / {val_graph_data.num_nodes}"
                )
                val_input_nodes = val_nodes_with_positives

            if is_hetero:
                val_graph_data = ToUndirected()(val_graph_data)

            num_neighbors = [-1] * (self.num_hops + 1) if self.num_hops > 0 else [-1]
            val_loader = NeighborLoader(
                val_graph_data,
                num_neighbors=num_neighbors,
                batch_size=self.batch_size,
                input_nodes=val_input_nodes,
                shuffle=True,
                time_attr="time",
                subgraph_type="bidirectional",
            )

        if self.wandb_project is not None:
            config = {
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "temperature": self.temperature,
                "num_hops": self.num_hops,
                "graph_type": self.graph_type,
                "checkpoint_interval": self.checkpoint_interval,
                "train_cutoff_year": train_cutoff_year,
                "val_cutoff_year": val_cutoff_year,
                "device": str(self.device),
                "num_nodes_with_positives": len(nodes_with_positives),
                "gradient_clip_val": self.gradient_clip_val,
                "log_every_n_batches": self.log_every_n_batches,
                "include_semantic_edges": self.include_semantic_edges,
                "semantic_threshold": self.semantic_threshold,
                "semantic_max_neighbors": self.semantic_max_neighbors,
                "num_hard_negatives": self.num_hard_negatives,
            }
            if val_cutoff_year is not None:
                config["num_val_nodes_with_positives"] = len(val_nodes_with_positives)

            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name,
                config=config,
            )

        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        if self.wandb_project is not None:
            wandb.watch(model, log="all", log_freq=100)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        num_neighbors = [-1] * (self.num_hops + 1) if self.num_hops > 0 else [-1]

        train_loader = NeighborLoader(
            train_graph_data,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            input_nodes=input_nodes,
            shuffle=True,
            time_attr="time",
            subgraph_type="bidirectional",
        )

        steps_per_epoch = len(train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        print(f"\nStarting training for {self.epochs} epochs...")
        best_train_loss = float("inf")
        best_val_loss = float("inf")
        global_batch_counter = 0

        for epoch in range(self.epochs):
            train_loss, global_batch_counter, train_stats = self.train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                is_hetero,
                epoch,
                global_batch_counter,
            )

            # Enhanced console output with key metrics
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            # Print key training statistics
            if train_stats:
                print(
                    f"  Pos Sim:    {train_stats.get('pos_sim_mean', 0):.3f} ± {train_stats.get('pos_sim_std', 0):.3f}"
                )
                print(
                    f"  Neg Sim:    {train_stats.get('neg_sim_mean', 0):.3f} ± {train_stats.get('neg_sim_std', 0):.3f}"
                )
                print(f"  Margin:     {train_stats.get('margin_mean', 0):.3f}")
                print(f"  Acc@1:      {train_stats.get('acc@1', 0):.2%}")
                print(f"  Num Negs:   {train_stats.get('num_negatives_mean', 0):.1f}")

            # Run validation if enabled and it's time to evaluate
            val_loss = None
            val_stats = None
            if val_loader is not None and (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss, val_stats = self.validate(model, val_loader, is_hetero)
                print(f"  Val Loss:   {val_loss:.4f}")

                # Print key validation statistics
                if val_stats:
                    print(f"  Val Pos Sim: {val_stats.get('pos_sim_mean', 0):.3f}")
                    print(f"  Val Margin:  {val_stats.get('margin_mean', 0):.3f}")
                    print(f"  Val Acc@1:   {val_stats.get('acc@1', 0):.2%}")

            # Log to wandb
            if self.wandb_project is not None:
                log_dict = {
                    "train/epoch_loss": train_loss,
                    "epoch": epoch + 1,
                }

                # Add epoch-level training statistics
                for key, value in train_stats.items():
                    log_dict[f"train_epoch/{key}"] = value

                if val_loss is not None:
                    log_dict["val/epoch_loss"] = val_loss
                    # Add epoch-level validation statistics
                    if val_stats:
                        for key, value in val_stats.items():
                            log_dict[f"val_epoch/{key}"] = value

                wandb.log(log_dict)

            if val_loss is not None and val_loss < best_val_loss:
                torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                print(
                    f"  ✓ New best validation loss: {best_val_loss:.4f} -> {val_loss:.4f}"
                )
                best_val_loss = val_loss
                if self.wandb_project is not None:
                    wandb.run.summary["best_val_loss"] = best_val_loss

            if (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                torch.save(
                    model.state_dict(),
                    checkpoint_path,
                )
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")

        if val_loader is not None:
            print(f"\nLoading best model (val loss: {best_val_loss:.4f})...")
        else:
            print(f"\nLoading best model (train loss: {best_train_loss:.4f})...")

        model.load_state_dict(torch.load(f"{self.output_path}/best_model.pt"))

        if self.wandb_project is not None:
            wandb.finish()

        return model

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
    return_stats=False,
    # NEW: Similarity-based false negative masking
    gnn_similarity_threshold=None,  # Mask if GNN embedding cosine sim > threshold
    anchor_x=None,  # Original text embeddings for anchors [batch_size, dim]
    positive_x=None,  # Original text embeddings for positives [batch_size, dim]
    x_similarity_threshold=None,  # Mask if text embedding cosine sim > threshold
):
    """
    In-batch negative contrastive loss with temporal filtering and False Negative masking.

    anchor: [batch_size, dim]
    positive: [batch_size, dim]
    anchor_indices: [batch_size] - IDs of source nodes
    positive_indices: [batch_size] - IDs of target nodes
    gnn_similarity_threshold: float - Mask negatives with GNN cosine sim > threshold
    anchor_x: [batch_size, dim] - Original text embeddings for anchors (before GNN)
    positive_x: [batch_size, dim] - Original text embeddings for positives (before GNN)
    x_similarity_threshold: float - Mask negatives with text cosine sim > threshold
    return_stats: if True, return (loss, stats_dict), else return loss only
    """
    # Compute similarity matrix using unnormalized dot product: [batch_size, batch_size]
    sim_matrix = torch.mm(anchor, positive.t()) / temperature

    batch_size = sim_matrix.size(0)
    diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)

    # Initialize mask for False Negatives (True = mask out/ignore)
    false_negative_mask = torch.zeros_like(sim_matrix, dtype=torch.bool)

    # Track statistics about masking
    mask_stats = {}

    # 1. Same source/target masking (existing logic)
    if anchor_indices is not None and positive_indices is not None:
        # same_anchor[i,k] = True if anchor_i == anchor_k
        same_anchor = anchor_indices.unsqueeze(1) == anchor_indices.unsqueeze(0)
        # same_target[k,j] = True if positive_k == positive_j
        same_target = positive_indices.unsqueeze(1) == positive_indices.unsqueeze(0)

        # Mask (i,j) if there exists ANY k where:
        #   anchor_i == anchor_k  AND  positive_k == positive_j
        # This means positive_j is a true positive for anchor_i
        index_based_mask = (same_anchor.float() @ same_target.float()) > 0
        false_negative_mask = false_negative_mask | index_based_mask

        if return_stats:
            # Count how many were masked by index logic (excluding diagonal)
            mask_stats["num_masked_by_index"] = (
                (index_based_mask & ~diagonal_mask).sum().item()
            )

    # 2. NEW: GNN embedding similarity-based masking
    if gnn_similarity_threshold is not None:
        # Compute cosine similarity (normalized) for consistent thresholding
        anchor_norm = F.normalize(anchor, p=2, dim=1)
        positive_norm = F.normalize(positive, p=2, dim=1)
        gnn_cosine_sim = torch.mm(anchor_norm, positive_norm.t())

        # Mask negatives that are too similar (potential false negatives)
        # But never mask the diagonal (true positives)
        high_gnn_sim_mask = (gnn_cosine_sim > gnn_similarity_threshold) & ~diagonal_mask

        if return_stats:
            # Count newly masked (not already masked by other criteria)
            newly_masked = high_gnn_sim_mask & ~false_negative_mask
            mask_stats["num_masked_by_gnn_sim"] = newly_masked.sum().item()
            mask_stats["gnn_sim_threshold"] = gnn_similarity_threshold
            # Track the similarity distribution of masked pairs
            if newly_masked.any():
                mask_stats["masked_gnn_sim_mean"] = (
                    gnn_cosine_sim[newly_masked].mean().item()
                )
                mask_stats["masked_gnn_sim_max"] = (
                    gnn_cosine_sim[newly_masked].max().item()
                )

        false_negative_mask = false_negative_mask | high_gnn_sim_mask

    # 3. NEW: Text embedding similarity-based masking
    if (
        x_similarity_threshold is not None
        and anchor_x is not None
        and positive_x is not None
    ):
        # Compute cosine similarity of original text embeddings
        anchor_x_norm = F.normalize(anchor_x, p=2, dim=1)
        positive_x_norm = F.normalize(positive_x, p=2, dim=1)
        x_cosine_sim = torch.mm(anchor_x_norm, positive_x_norm.t())

        # Mask negatives with highly similar text (potential false negatives)
        high_x_sim_mask = (x_cosine_sim > x_similarity_threshold) & ~diagonal_mask

        if return_stats:
            newly_masked = high_x_sim_mask & ~false_negative_mask
            mask_stats["num_masked_by_x_sim"] = newly_masked.sum().item()
            mask_stats["x_sim_threshold"] = x_similarity_threshold
            if newly_masked.any():
                mask_stats["masked_x_sim_mean"] = (
                    x_cosine_sim[newly_masked].mean().item()
                )
                mask_stats["masked_x_sim_max"] = x_cosine_sim[newly_masked].max().item()

        false_negative_mask = false_negative_mask | high_x_sim_mask

    # Ensure diagonal is NOT masked (we need it for the loss)
    final_mask = false_negative_mask & ~diagonal_mask

    if return_stats:
        mask_stats["total_masked"] = final_mask.sum().item()
        mask_stats["mask_ratio"] = final_mask.sum().item() / (
            batch_size * (batch_size - 1)
        )

    # Apply mask: set false negatives to -inf so Softmax ignores them
    sim_matrix = sim_matrix.masked_fill(final_mask, float("-inf"))

    # Apply temporal masking if time information is provided
    if anchor_times is not None and positive_times is not None:
        # positive_j is valid negative for anchor_i only if positive_time_j < anchor_time_i
        time_mask = positive_times.unsqueeze(0) < anchor_times.unsqueeze(1)

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

    # Add masking statistics
    stats.update(mask_stats)

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
        # NEW: Similarity-based false negative masking thresholds
        gnn_similarity_threshold: float | None = None,  # e.g., 0.8
        x_similarity_threshold: float | None = None,  # e.g., 0.9
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
        self.gnn_similarity_threshold = gnn_similarity_threshold
        self.x_similarity_threshold = x_similarity_threshold

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")
        if gnn_similarity_threshold is not None:
            print(
                f"GNN similarity threshold for FN masking: {gnn_similarity_threshold}"
            )
        if x_similarity_threshold is not None:
            print(f"Text similarity threshold for FN masking: {x_similarity_threshold}")

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
                "original_x": batch[
                    "paragraph"
                ].x,  # NEW: Keep original x for similarity masking
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
            leakage_mask = ~((src < batch_size) | (tgt < batch_size))
            masked_edge_index = edge_index[:, leakage_mask]
            masked_edge_attr = (
                edge_attr[leakage_mask] if edge_attr is not None else None
            )

            return {
                "batch_size": batch_size,
                "modified_batch": None,
                "edge_index": edge_index,
                "x": x,
                "original_x": batch.x,  # NEW: Keep original x for similarity masking
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
        original_x = batch_data.get("original_x")  # NEW: Get original text embeddings

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

        # NEW: Get original text embeddings for similarity-based masking
        anchor_x = None
        positive_x = None
        if self.x_similarity_threshold is not None and original_x is not None:
            anchor_x = original_x[batch_src]
            positive_x = original_x[batch_tgt]

        # Get times for all pairs
        pair_anchor_times = None
        pair_positive_times = None
        if all_times is not None:
            pair_anchor_times = all_times[batch_src]
            pair_positive_times = all_times[batch_tgt]

        # Compute loss with in-batch negatives
        result = info_nce_loss(
            anchor_emb,
            positive_emb,
            self.temperature,
            anchor_times=pair_anchor_times,
            positive_times=pair_positive_times,
            anchor_indices=batch_src,  # Mask Same Source
            positive_indices=batch_tgt,  # Mask Same Target
            return_stats=return_stats,
            # NEW: Pass similarity thresholds
            gnn_similarity_threshold=self.gnn_similarity_threshold,
            anchor_x=anchor_x,
            positive_x=positive_x,
            x_similarity_threshold=self.x_similarity_threshold,
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
                add_reverse_edges=True,  # Enable edge direction features
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
                # NEW: Log similarity thresholds
                "gnn_similarity_threshold": self.gnn_similarity_threshold,
                "x_similarity_threshold": self.x_similarity_threshold,
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
                # NEW: Print masking statistics
                if train_stats.get("total_masked", 0) > 0:
                    print(
                        f"  Masked FN:  {train_stats.get('total_masked', 0):.1f} ({train_stats.get('mask_ratio', 0):.1%})"
                    )
                    if "num_masked_by_gnn_sim" in train_stats:
                        print(
                            f"    By GNN:   {train_stats.get('num_masked_by_gnn_sim', 0):.1f}"
                        )
                    if "num_masked_by_x_sim" in train_stats:
                        print(
                            f"    By Text:  {train_stats.get('num_masked_by_x_sim', 0):.1f}"
                        )

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

            # Save best model based on validation loss if available, otherwise training loss
            if val_loss is not None:
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best validation loss: {best_val_loss:.4f} -> {val_loss:.4f}"
                    )
                    best_val_loss = val_loss
                    if self.wandb_project is not None:
                        wandb.run.summary["best_val_loss"] = best_val_loss
            else:
                if train_loss < best_train_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best training loss: {best_train_loss:.4f} -> {train_loss:.4f}"
                    )
                    best_train_loss = train_loss
                    if self.wandb_project is not None:
                        wandb.run.summary["best_train_loss"] = best_train_loss

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

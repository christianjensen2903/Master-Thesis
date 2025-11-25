"""
Conventional Link Prediction Baseline Trainer

This implements standard negative sampling with BCE loss, which is the
conventional approach for GNN-based link prediction (as seen in papers
like GAE/VGAE, SEAL, etc.)
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data, HeteroData
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import wandb

from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
)


class LinkPredictionTrainer:
    """
    Conventional link prediction trainer using:
    - Explicit negative sampling (not in-batch)
    - BCE loss with dot-product decoder
    - Optional temporal filtering for negatives

    This serves as a baseline comparison to contrastive learning approaches.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        output_path: str = "output/link_pred_baseline",
        batch_size: int = 16,
        epochs: int = 5,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-5,
        num_hops: int = 2,
        graph_type: str = "heterogeneous",
        checkpoint_interval: int = 25,
        wandb_project: str | None = "gnn-training",
        wandb_name: str | None = None,
        warmup_epochs: int = 3,
        eval_every_n_epochs: int = 1,
        gradient_clip_val: float | None = None,
        log_every_n_batches: int = 100,
        neg_sampling_ratio: float = 1.0,  # Ratio of negatives to positives
        use_temporal_negatives: bool = True,  # Only sample negatives from past
    ):
        self.preprocessed_dir = preprocessed_dir
        self.output_path = output_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_hops = num_hops
        self.graph_type = graph_type
        self.checkpoint_interval = checkpoint_interval
        self.wandb_project = wandb_project
        self.wandb_name = wandb_name
        self.warmup_epochs = warmup_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_batches = log_every_n_batches
        self.neg_sampling_ratio = neg_sampling_ratio
        self.use_temporal_negatives = use_temporal_negatives

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")

    def _sample_negatives(
        self,
        edge_index: torch.Tensor,
        num_nodes: int,
        num_neg_samples: int,
        node_times: torch.Tensor | None = None,
        source_nodes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Sample negative edges using PyG's negative_sampling or temporal-aware sampling.

        Args:
            edge_index: Positive edges [2, num_edges]
            num_nodes: Total number of nodes
            num_neg_samples: Number of negative samples to generate
            node_times: Optional node timestamps for temporal filtering
            source_nodes: Source nodes of positive edges (for temporal sampling)

        Returns:
            neg_edge_index: [2, num_neg_samples]
        """
        if not self.use_temporal_negatives or node_times is None:
            # Standard negative sampling
            neg_edge_index = negative_sampling(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_neg_samples,
                method="sparse",  # More efficient for large graphs
            )
            return neg_edge_index

        # Temporal-aware negative sampling
        # For each source node, only sample targets with time < source_time
        neg_src = []
        neg_dst = []

        if source_nodes is None:
            source_nodes = edge_index[0]

        unique_sources = source_nodes.unique()
        samples_per_source = max(1, num_neg_samples // len(unique_sources))

        for src in unique_sources:
            src_time = node_times[src]
            # Valid targets: nodes with time < src_time
            valid_mask = node_times < src_time
            valid_targets = torch.where(valid_mask)[0]

            if len(valid_targets) == 0:
                continue

            # Exclude existing edges from this source
            existing_targets = edge_index[1, edge_index[0] == src]
            valid_targets = valid_targets[~torch.isin(valid_targets, existing_targets)]

            if len(valid_targets) == 0:
                continue

            # Sample random targets
            n_samples = min(samples_per_source, len(valid_targets))
            perm = torch.randperm(len(valid_targets), device=valid_targets.device)[
                :n_samples
            ]
            sampled_targets = valid_targets[perm]

            neg_src.extend([src.item()] * n_samples)
            neg_dst.extend(sampled_targets.tolist())

        if len(neg_src) == 0:
            # Fallback to standard sampling if temporal sampling fails
            return negative_sampling(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_neg_samples=num_neg_samples,
            )

        neg_edge_index = torch.tensor([neg_src, neg_dst], device=edge_index.device)
        return neg_edge_index

    def _compute_link_prediction_loss(
        self,
        embeddings: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute BCE loss for link prediction.

        Uses dot-product decoder: score(u, v) = σ(z_u · z_v)
        """
        # Positive scores
        pos_src, pos_dst = pos_edge_index
        pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

        # Negative scores
        neg_src, neg_dst = neg_edge_index
        neg_scores = (embeddings[neg_src] * embeddings[neg_dst]).sum(dim=1)

        # BCE loss
        pos_loss = F.binary_cross_entropy_with_logits(
            pos_scores, torch.ones_like(pos_scores)
        )
        neg_loss = F.binary_cross_entropy_with_logits(
            neg_scores, torch.zeros_like(neg_scores)
        )

        loss = pos_loss + neg_loss

        # Compute metrics for logging
        with torch.no_grad():
            pos_acc = (pos_scores > 0).float().mean()
            neg_acc = (neg_scores < 0).float().mean()
            auc_approx = (
                (pos_scores.unsqueeze(1) > neg_scores.unsqueeze(0)).float().mean()
            )

        stats = {
            "pos_loss": pos_loss.item(),
            "neg_loss": neg_loss.item(),
            "pos_acc": pos_acc.item(),
            "neg_acc": neg_acc.item(),
            "auc_approx": auc_approx.item(),
        }

        return loss, stats

    def _process_batch(self, batch, is_hetero: bool) -> dict | None:
        """Process a batch and extract necessary components."""
        if is_hetero:
            batch_size = batch["paragraph"].batch_size
            x = batch["paragraph"].x.clone()

            if hasattr(batch["paragraph"], "x_query"):
                x[:batch_size] = batch["paragraph"].x_query[:batch_size]

            node_times = (
                batch["paragraph"].time if hasattr(batch["paragraph"], "time") else None
            )

            if ("paragraph", "cites", "paragraph") in batch.edge_types:
                cite_edge_index = batch["paragraph", "cites", "paragraph"].edge_index
            else:
                return None

            # Mask citation edges from anchor nodes (prevent leakage)
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

            # Get positive edges (only from input batch nodes)
            input_mask = cite_src < batch_size
            pos_edge_index = cite_edge_index[:, input_mask]

            modified_batch = batch.clone()
            modified_batch["paragraph", "cites", "paragraph"].edge_index = (
                masked_cite_edges
            )
            modified_batch["paragraph"].x = x

            return {
                "batch_size": batch_size,
                "modified_batch": modified_batch,
                "pos_edge_index": pos_edge_index,
                "full_edge_index": cite_edge_index,
                "x": x,
                "node_times": node_times,
                "num_nodes": batch["paragraph"].num_nodes,
            }
        else:
            # Homogeneous graph
            batch_size = batch.batch_size
            x = batch.x.clone()

            if hasattr(batch, "x_query"):
                x[:batch_size] = batch.x_query[:batch_size]

            node_times = batch.time if hasattr(batch, "time") else None
            edge_index = batch.edge_index

            # Mask edges from anchor nodes
            src, tgt = edge_index
            leakage_mask = ~((src < batch_size) | (tgt < batch_size))
            masked_edge_index = edge_index[:, leakage_mask]

            # Get positive edges
            input_mask = src < batch_size
            pos_edge_index = edge_index[:, input_mask]

            return {
                "batch_size": batch_size,
                "modified_batch": None,
                "pos_edge_index": pos_edge_index,
                "full_edge_index": edge_index,
                "x": x,
                "masked_edge_index": masked_edge_index,
                "date_feature": batch.date_feature,
                "node_times": node_times,
                "num_nodes": batch.num_nodes,
            }

    def _compute_loss(
        self,
        model: nn.Module,
        batch_data: dict,
        is_hetero: bool,
    ) -> tuple[torch.Tensor | None, dict | None]:
        """Compute link prediction loss for a processed batch."""
        pos_edge_index = batch_data["pos_edge_index"]

        if pos_edge_index.size(1) == 0:
            return None, None

        # Get embeddings
        if is_hetero:
            out = model(batch_data["modified_batch"])
            embeddings = out["paragraph"] if isinstance(out, dict) else out
        else:
            out = model(
                batch_data["x"],
                batch_data["masked_edge_index"],
                date_feature=batch_data.get("date_feature"),
            )
            embeddings = out["paragraph"] if isinstance(out, dict) else out

        # Sample negatives
        num_pos = pos_edge_index.size(1)
        num_neg = int(num_pos * self.neg_sampling_ratio)

        neg_edge_index = self._sample_negatives(
            edge_index=batch_data["full_edge_index"],
            num_nodes=batch_data["num_nodes"],
            num_neg_samples=num_neg,
            node_times=batch_data.get("node_times"),
            source_nodes=pos_edge_index[0],
        )

        # Compute loss
        loss, stats = self._compute_link_prediction_loss(
            embeddings, pos_edge_index, neg_edge_index
        )

        return loss, stats

    def train_epoch(
        self,
        model: nn.Module,
        loader: NeighborLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        is_hetero: bool,
        epoch: int = 0,
        global_batch_counter: int = 0,
    ) -> tuple[float, int, dict]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        batch_counter = global_batch_counter

        epoch_stats = {
            "pos_loss": [],
            "neg_loss": [],
            "pos_acc": [],
            "neg_acc": [],
            "auc_approx": [],
        }

        for batch_idx, batch in enumerate(
            tqdm(loader, desc="Training batches", leave=False)
        ):
            batch_data = self._process_batch(batch, is_hetero)
            if batch_data is None:
                continue

            loss, stats = self._compute_loss(model, batch_data, is_hetero)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()

            if self.gradient_clip_val is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.gradient_clip_val
                )

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            batch_counter += 1

            # Accumulate stats
            for key in epoch_stats:
                if key in stats:
                    epoch_stats[key].append(stats[key])

            if (
                self.wandb_project is not None
                and batch_idx % self.log_every_n_batches == 0
            ):
                log_dict = {
                    "train/batch_loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]["lr"],
                    "train/batch": batch_counter,
                    **{f"train/{k}": v for k, v in stats.items()},
                }
                wandb.log(log_dict)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_stats = {k: sum(v) / len(v) if v else 0.0 for k, v in epoch_stats.items()}

        return avg_loss, batch_counter, avg_stats

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        loader: NeighborLoader,
        is_hetero: bool,
    ) -> tuple[float, dict]:
        """Validate the model."""
        model.eval()
        total_loss = 0
        num_batches = 0

        val_stats = {
            "pos_loss": [],
            "neg_loss": [],
            "pos_acc": [],
            "neg_acc": [],
            "auc_approx": [],
        }

        for batch in tqdm(loader, desc="Validation batches", leave=False):
            batch_data = self._process_batch(batch, is_hetero)
            if batch_data is None:
                continue

            loss, stats = self._compute_loss(model, batch_data, is_hetero)
            if loss is None:
                continue

            total_loss += loss.item()
            num_batches += 1

            for key in val_stats:
                if key in stats:
                    val_stats[key].append(stats[key])

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_stats = {k: sum(v) / len(v) if v else 0.0 for k, v in val_stats.items()}

        return avg_loss, avg_stats

    def train(
        self,
        gnn_model: nn.Module,
        train_cutoff_year: int,
        val_cutoff_year: int,
    ) -> nn.Module:
        """Train GNN model with link prediction objective."""
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("Training GNN with Link Prediction Baseline (BCE + Negative Sampling)")
        print("=" * 80)

        is_hetero = self.graph_type == "heterogeneous"

        # Build training graph
        if is_hetero:
            builder = HeterogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
            ).to(self.device)
        else:
            builder = HomogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
            ).to(self.device)

        # Filter nodes with positive examples
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
            nodes_with_positives = edge_index[0].unique()
            print(
                f"  Nodes with citations: {len(nodes_with_positives)} / {train_graph_data.num_nodes}"
            )
            input_nodes = nodes_with_positives

        train_graph_data = ToUndirected()(train_graph_data)

        # Build validation graph
        print("\nBuilding validation graph...")
        print(f"  Training cutoff: {train_cutoff_year}")
        print(f"  Validation cutoff: {val_cutoff_year}")

        if is_hetero:
            val_graph_data = builder.build_graph(
                train_cutoff_year=val_cutoff_year,
                include_only_citing=True,
            ).to(self.device)

            val_cite_edge_index = val_graph_data[
                "paragraph", "cites", "paragraph"
            ].edge_index

            if hasattr(val_graph_data["paragraph"], "time"):
                citing_nodes = val_cite_edge_index[0].unique()
                node_times = val_graph_data["paragraph"].time
                post_cutoff_mask = node_times > train_cutoff_year
                post_cutoff_nodes = torch.where(post_cutoff_mask)[0]
                val_nodes_with_positives = citing_nodes[
                    torch.isin(citing_nodes, post_cutoff_nodes)
                ]
            else:
                val_nodes_with_positives = val_cite_edge_index[0].unique()

            print(
                f"  Val paragraph nodes (post-cutoff with citations): {len(val_nodes_with_positives)}"
            )
            val_input_nodes = ("paragraph", val_nodes_with_positives)
        else:
            val_graph_data = builder.build_graph(
                train_cutoff_year=val_cutoff_year,
                include_only_citing=True,
            ).to(self.device)

            val_edge_index = val_graph_data.edge_index
            citing_nodes = val_edge_index[0].unique()
            node_times = val_graph_data.time
            post_cutoff_mask = node_times > train_cutoff_year
            post_cutoff_nodes = torch.where(post_cutoff_mask)[0]
            val_nodes_with_positives = citing_nodes[
                torch.isin(citing_nodes, post_cutoff_nodes)
            ]

            print(
                f"  Val nodes (post-cutoff with citations): {len(val_nodes_with_positives)}"
            )
            val_input_nodes = val_nodes_with_positives

        val_graph_data = ToUndirected()(val_graph_data)

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

        val_loader = NeighborLoader(
            val_graph_data,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            input_nodes=val_input_nodes,
            shuffle=False,
            time_attr="time",
            subgraph_type="bidirectional",
        )

        if self.wandb_project is not None:
            config = {
                "method": "link_prediction_baseline",
                "loss": "BCE",
                "negative_sampling": "explicit",
                "neg_sampling_ratio": self.neg_sampling_ratio,
                "use_temporal_negatives": self.use_temporal_negatives,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "num_hops": self.num_hops,
                "graph_type": self.graph_type,
                "train_cutoff_year": train_cutoff_year,
                "val_cutoff_year": val_cutoff_year,
                "device": str(self.device),
                "num_nodes_with_positives": len(nodes_with_positives),
                "num_val_nodes_with_positives": len(val_nodes_with_positives),
            }
            wandb.init(project=self.wandb_project, name=self.wandb_name, config=config)

        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        if self.wandb_project is not None:
            wandb.watch(model, log="all", log_freq=100)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
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
        best_val_loss = float("inf")
        best_train_loss = float("inf")
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

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(
                f"  Train Loss: {train_loss:.4f} | AUC≈{train_stats['auc_approx']:.4f} | "
                f"Pos Acc: {train_stats['pos_acc']:.4f} | Neg Acc: {train_stats['neg_acc']:.4f}"
            )

            val_loss = None
            if (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss, val_stats = self.validate(model, val_loader, is_hetero)
                print(
                    f"  Val Loss:   {val_loss:.4f} | AUC≈{val_stats['auc_approx']:.4f} | "
                    f"Pos Acc: {val_stats['pos_acc']:.4f} | Neg Acc: {val_stats['neg_acc']:.4f}"
                )

            if self.wandb_project is not None:
                log_dict = {"train/epoch_loss": train_loss, "epoch": epoch + 1}
                log_dict.update({f"train/epoch_{k}": v for k, v in train_stats.items()})
                if val_loss is not None:
                    log_dict["val/epoch_loss"] = val_loss
                    log_dict.update({f"val/epoch_{k}": v for k, v in val_stats.items()})
                wandb.log(log_dict)

            # Save best model
            if val_loss is not None:
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best validation loss: {best_val_loss:.4f} -> {val_loss:.4f}"
                    )
                    best_val_loss = val_loss
            else:
                if train_loss < best_train_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best training loss: {best_train_loss:.4f} -> {train_loss:.4f}"
                    )
                    best_train_loss = train_loss

            if (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")

        print(f"\nLoading best model...")
        model.load_state_dict(torch.load(f"{self.output_path}/best_model.pt"))

        if self.wandb_project is not None:
            wandb.finish()

        return model


# Alternative: BPR Loss Trainer (Margin-based, common in recommendations)
class BPRLinkPredictionTrainer(LinkPredictionTrainer):
    """
    Variant using Bayesian Personalized Ranking (BPR) loss.

    BPR is a margin-based loss that directly optimizes for ranking:
    L = -log(σ(score_pos - score_neg))

    Often preferred for retrieval/ranking tasks.
    """

    def _compute_link_prediction_loss(
        self,
        embeddings: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Compute BPR loss."""
        # Positive scores
        pos_src, pos_dst = pos_edge_index
        pos_scores = (embeddings[pos_src] * embeddings[pos_dst]).sum(dim=1)

        # Negative scores (sample same number as positives)
        neg_src, neg_dst = neg_edge_index

        # Handle case where we have different numbers of pos/neg samples
        min_samples = min(len(pos_scores), neg_edge_index.size(1))
        pos_scores = pos_scores[:min_samples]
        neg_scores = (
            embeddings[neg_src[:min_samples]] * embeddings[neg_dst[:min_samples]]
        ).sum(dim=1)

        # BPR loss: -log(σ(pos - neg))
        loss = -F.logsigmoid(pos_scores - neg_scores).mean()

        with torch.no_grad():
            pos_acc = (pos_scores > 0).float().mean()
            neg_acc = (neg_scores < 0).float().mean()
            margin_positive = (pos_scores > neg_scores).float().mean()

        stats = {
            "pos_score_mean": pos_scores.mean().item(),
            "neg_score_mean": neg_scores.mean().item(),
            "margin_positive_rate": margin_positive.item(),
            "pos_acc": pos_acc.item(),
            "neg_acc": neg_acc.item(),
            "auc_approx": margin_positive.item(),  # For BPR, this is exact
        }

        return loss, stats


if __name__ == "__main__":
    from models import CitationGNN

    model = CitationGNN(input_dim=384, output_dim=384, num_layers=2, dropout=0.1)
    trainer = BPRLinkPredictionTrainer(
        preprocessed_dir="data/preprocessed",
        output_path="checkpoints/link_pred_baseline",
        batch_size=512,
        epochs=150,
        learning_rate=0.1,
        weight_decay=1e-4,
        graph_type="homogeneous",
        eval_every_n_epochs=5,
        neg_sampling_ratio=1.0,
    )
    trainer.train(model, 2018, 2022)

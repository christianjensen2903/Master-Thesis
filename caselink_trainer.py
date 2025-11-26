"""
CaseLink-specific trainer with degree regularization.

Extends the base GNN trainer with CaseLink-specific features:
- Degree regularization loss
- Support for multi-relation graphs
- Paragraph-only loss computation (excluding article nodes)
"""

import os

# Fix OpenMP conflict on macOS (FAISS and PyTorch may use different OpenMP runtimes)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from caselink import (
    CaseLinkGraphBuilder,
    CaseLinkGNN,
    CaseLinkGNNRelational,
    info_nce_loss_with_degree_reg,
)


class CaseLinkTrainer:
    """
    Trainer for CaseLink-style GNN models.
    """

    def __init__(
        self,
        preprocessed_dir: str,
        output_path: str = "output/caselink",
        batch_size: int = 64,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_hops: int = 2,
        checkpoint_interval: int = 25,
        wandb_project: str | None = "caselink-training",
        wandb_name: str | None = None,
        warmup_epochs: int = 2,
        eval_every_n_epochs: int = 1,
        gradient_clip_val: float | None = 1.0,
        log_every_n_batches: int = 50,
        # CaseLink-specific
        degree_reg_weight: float = 0.1,
        include_semantic_edges: bool = True,
        semantic_threshold: float = 0.7,
        semantic_max_neighbors: int = 10,
        include_article_nodes: bool = True,
    ):
        self.preprocessed_dir = preprocessed_dir
        self.output_path = output_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.num_hops = num_hops
        self.checkpoint_interval = checkpoint_interval
        self.wandb_project = wandb_project if HAS_WANDB else None
        self.wandb_name = wandb_name
        self.warmup_epochs = warmup_epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.gradient_clip_val = gradient_clip_val
        self.log_every_n_batches = log_every_n_batches

        # CaseLink-specific
        self.degree_reg_weight = degree_reg_weight
        self.include_semantic_edges = include_semantic_edges
        self.semantic_threshold = semantic_threshold
        self.semantic_max_neighbors = semantic_max_neighbors
        self.include_article_nodes = include_article_nodes

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

    def _process_batch(self, batch, num_par_nodes_total: int):
        """
        Process a batch for CaseLink-style training.

        Key difference: we only compute loss on paragraph nodes,
        not article nodes.
        """
        batch_size = batch.batch_size
        x = batch.x.clone()

        # Replace input node features with query embeddings
        if hasattr(batch, "x_query"):
            x[:batch_size] = batch.x_query[:batch_size]

        # Get timestamps for temporal filtering
        anchor_times = batch.time[:batch_size] if hasattr(batch, "time") else None

        edge_index = batch.edge_index
        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        node_type = batch.node_type if hasattr(batch, "node_type") else None

        # For CaseLink, we only want citation edges (type 0) for the contrastive loss
        # Other edges help with message passing but aren't used for loss

        # Mask edges to prevent leakage:
        # 1. Mask all outgoing edges from anchor (src is anchor)
        # 2. Mask incoming citation edges to anchor (tgt is anchor AND citation edge)
        # Keep incoming non-citation edges for one-directional info flow
        src, tgt = edge_index
        outgoing_from_anchor = src < batch_size
        incoming_to_anchor = tgt < batch_size
        is_citation = (
            (edge_attr == 0)
            if edge_attr is not None
            else torch.ones(
                edge_index.size(1), dtype=torch.bool, device=edge_index.device
            )
        )
        incoming_citation_to_anchor = incoming_to_anchor & is_citation
        leakage_mask = ~(outgoing_from_anchor | incoming_citation_to_anchor)
        masked_edge_index = edge_index[:, leakage_mask]
        masked_edge_attr = edge_attr[leakage_mask] if edge_attr is not None else None

        return {
            "batch_size": batch_size,
            "x": x,
            "edge_index": batch.edge_index,  # Full edges for finding positives
            "masked_edge_index": masked_edge_index,  # Masked edges for GNN
            "edge_attr": edge_attr,
            "masked_edge_attr": masked_edge_attr,
            "node_type": node_type,
            "anchor_times": anchor_times,
            "all_times": batch.time if hasattr(batch, "time") else None,
            "date_feature": (
                batch.date_feature if hasattr(batch, "date_feature") else None
            ),
            "num_par_nodes_total": num_par_nodes_total,
        }

    def _compute_loss(
        self,
        model: nn.Module,
        batch_data: dict,
        return_stats: bool = False,
    ):
        """Compute CaseLink-style loss with degree regularization."""
        batch_size = batch_data["batch_size"]
        edge_index = batch_data["edge_index"]
        edge_attr = batch_data.get("edge_attr")
        all_times = batch_data.get("all_times")
        num_par_nodes = batch_data.get("num_par_nodes_total", batch_data["x"].size(0))

        # Forward pass through GNN
        embeddings = model(
            batch_data["x"],
            batch_data["masked_edge_index"],
            date_feature=batch_data.get("date_feature"),
            edge_attr=batch_data.get("masked_edge_attr"),
            node_type=batch_data.get("node_type"),
        )

        # Find citation edges where source is in input batch
        # Only use "cites" edges (type 0) for contrastive loss
        src, tgt = edge_index

        if edge_attr is not None:
            # Only use forward citation edges (type 0)
            cites_mask = edge_attr == 0
            input_mask = (src < batch_size) & cites_mask
        else:
            input_mask = src < batch_size

        # Also ensure we're only looking at paragraph-to-paragraph edges
        node_type = batch_data.get("node_type")
        if node_type is not None:
            # Both source and target should be paragraphs (type 0)
            src_is_par = node_type[src] == 0
            tgt_is_par = node_type[tgt] == 0
            input_mask = input_mask & src_is_par & tgt_is_par

        if input_mask.sum() == 0:
            return (None, None) if return_stats else None

        batch_src = src[input_mask]
        batch_tgt = tgt[input_mask]

        anchor_emb = embeddings[batch_src]
        positive_emb = embeddings[batch_tgt]

        # Get times for temporal filtering
        pair_anchor_times = None
        pair_positive_times = None
        if all_times is not None:
            pair_anchor_times = all_times[batch_src]
            pair_positive_times = all_times[batch_tgt]

        # Compute loss with degree regularization
        result = info_nce_loss_with_degree_reg(
            anchor_emb,
            positive_emb,
            temperature=self.temperature,
            edge_index=batch_data["masked_edge_index"],
            degree_reg_weight=self.degree_reg_weight,
            anchor_times=pair_anchor_times,
            positive_times=pair_positive_times,
            anchor_indices=batch_src,
            positive_indices=batch_tgt,
            return_stats=return_stats,
            all_embeddings=embeddings,
        )

        if return_stats:
            loss, stats = result
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
        num_par_nodes: int,
        epoch: int = 0,
        global_batch_counter: int = 0,
    ) -> tuple[float, int, dict]:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0
        batch_counter = global_batch_counter
        batch_stats_accum = {}

        for batch_idx, batch in enumerate(
            tqdm(loader, desc="Training batches", leave=False)
        ):
            batch_data = self._process_batch(batch, num_par_nodes)
            result = self._compute_loss(model, batch_data, return_stats=True)

            if result[0] is None:
                continue

            loss, batch_stats = result

            optimizer.zero_grad()
            loss.backward()

            # Compute gradient stats
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
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
                    "train/grad_norm": total_norm,
                }
                for key, value in batch_stats.items():
                    log_dict[f"train/{key}"] = value
                wandb.log(log_dict)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

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
        num_par_nodes: int,
    ) -> tuple[float, dict]:
        """Validate the model."""
        model.eval()
        total_loss = 0
        num_batches = 0
        val_stats_accum = {}

        for batch in tqdm(loader, desc="Validation batches", leave=False):
            batch_data = self._process_batch(batch, num_par_nodes)
            result = self._compute_loss(model, batch_data, return_stats=True)

            if result[0] is None:
                continue

            loss, batch_stats = result
            total_loss += loss.item()
            num_batches += 1

            for key, value in batch_stats.items():
                if key not in val_stats_accum:
                    val_stats_accum[key] = []
                val_stats_accum[key].append(value)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0

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
    ) -> nn.Module:
        """Train CaseLink-style GNN model."""
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print("Training CaseLink-style GNN")
        print("=" * 80)

        # Build training graph
        builder = CaseLinkGraphBuilder(self.preprocessed_dir)
        train_graph = builder.build_graph(
            train_cutoff_year=train_cutoff_year,
            include_only_citing=True,
            include_semantic_edges=self.include_semantic_edges,
            semantic_threshold=self.semantic_threshold,
            semantic_max_neighbors=self.semantic_max_neighbors,
            include_article_nodes=self.include_article_nodes,
        ).to(self.device)

        num_par_nodes = train_graph.num_par_nodes

        # Filter to nodes with citation edges (for training)
        edge_attr = train_graph.edge_attr
        cites_mask = edge_attr == 0
        cites_edges = train_graph.edge_index[:, cites_mask]
        nodes_with_positives = cites_edges[0].unique()

        # Only include paragraph nodes (not articles)
        node_type = train_graph.node_type
        par_mask = node_type[nodes_with_positives] == 0
        nodes_with_positives = nodes_with_positives[par_mask]

        print(
            f"Paragraph nodes with citations: {len(nodes_with_positives)} / {num_par_nodes}"
        )

        # Build validation graph if needed
        val_loader = None
        val_num_par_nodes = None
        if val_cutoff_year is not None:
            print("\nBuilding validation graph...")
            val_graph = builder.build_graph(
                train_cutoff_year=val_cutoff_year,
                include_only_citing=True,
                include_semantic_edges=self.include_semantic_edges,
                semantic_threshold=self.semantic_threshold,
                semantic_max_neighbors=self.semantic_max_neighbors,
                include_article_nodes=self.include_article_nodes,
            ).to(self.device)

            val_num_par_nodes = val_graph.num_par_nodes
            val_edge_attr = val_graph.edge_attr
            val_cites_mask = val_edge_attr == 0
            val_cites_edges = val_graph.edge_index[:, val_cites_mask]
            val_nodes_with_positives = val_cites_edges[0].unique()

            val_node_type = val_graph.node_type
            val_par_mask = val_node_type[val_nodes_with_positives] == 0
            val_nodes_with_positives = val_nodes_with_positives[val_par_mask]

            # Filter to nodes after training cutoff
            if train_cutoff_year is not None:
                train_cutoff_timestamp = builder._date_to_timestamp(
                    f"{train_cutoff_year}-01-01"
                )
                node_times = val_graph.time[val_nodes_with_positives]
                time_mask = node_times > train_cutoff_timestamp
                val_nodes_with_positives = val_nodes_with_positives[time_mask]

            print(
                f"Val paragraph nodes with citations: {len(val_nodes_with_positives)} / {val_num_par_nodes}"
            )

            num_neighbors = [-1] * (self.num_hops + 1) if self.num_hops > 0 else [-1]
            val_loader = NeighborLoader(
                val_graph,
                num_neighbors=num_neighbors,
                batch_size=self.batch_size,
                input_nodes=val_nodes_with_positives,
                shuffle=True,
                time_attr="time",
                subgraph_type="bidirectional",
            )

        # Initialize wandb
        if self.wandb_project is not None:
            config = {
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "weight_decay": self.weight_decay,
                "temperature": self.temperature,
                "num_hops": self.num_hops,
                "train_cutoff_year": train_cutoff_year,
                "val_cutoff_year": val_cutoff_year,
                "degree_reg_weight": self.degree_reg_weight,
                "include_semantic_edges": self.include_semantic_edges,
                "include_article_nodes": self.include_article_nodes,
                "device": str(self.device),
            }
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name,
                config=config,
            )

        # Initialize model
        print("\nInitializing CaseLink GNN model...")
        model = gnn_model.to(self.device)

        if self.wandb_project is not None:
            wandb.watch(model, log="all", log_freq=100)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create training loader
        num_neighbors = [-1] * (self.num_hops + 1) if self.num_hops > 0 else [-1]
        train_loader = NeighborLoader(
            train_graph,
            num_neighbors=num_neighbors,
            batch_size=self.batch_size,
            input_nodes=nodes_with_positives,
            shuffle=True,
            time_attr="time",
            subgraph_type="bidirectional",
        )

        # Learning rate scheduler
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

        # Training loop
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
                num_par_nodes,
                epoch,
                global_batch_counter,
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            if train_stats:
                print(f"  Pos Sim: {train_stats.get('pos_sim_mean', 0):.3f}")
                print(f"  Neg Sim: {train_stats.get('neg_sim_mean', 0):.3f}")
                print(f"  Acc@1: {train_stats.get('acc@1', 0):.2%}")

            # Validation
            val_loss = None
            val_stats = None
            if val_loader is not None and (epoch + 1) % self.eval_every_n_epochs == 0:
                val_loss, val_stats = self.validate(
                    model, val_loader, val_num_par_nodes
                )
                print(f"  Val Loss: {val_loss:.4f}")
                if val_stats:
                    print(f"  Val Acc@1: {val_stats.get('acc@1', 0):.2%}")

            # Log to wandb
            if self.wandb_project is not None:
                log_dict = {
                    "train/epoch_loss": train_loss,
                    "epoch": epoch + 1,
                }
                for key, value in train_stats.items():
                    log_dict[f"train_epoch/{key}"] = value
                if val_loss is not None:
                    log_dict["val/epoch_loss"] = val_loss
                    if val_stats:
                        for key, value in val_stats.items():
                            log_dict[f"val_epoch/{key}"] = value
                wandb.log(log_dict)

            # Save best model
            if val_loss is not None:
                if val_loss < best_val_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best val loss: {best_val_loss:.4f} -> {val_loss:.4f}"
                    )
                    best_val_loss = val_loss
            else:
                if train_loss < best_train_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best train loss: {best_train_loss:.4f} -> {train_loss:.4f}"
                    )
                    best_train_loss = train_loss

            # Checkpoints
            if (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Save final model and load best
        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")
        model.load_state_dict(torch.load(f"{self.output_path}/best_model.pt"))

        if self.wandb_project is not None:
            wandb.finish()

        return model


if __name__ == "__main__":

    from caselink import CaseLinkGNN, CaseLinkGNNRelational

    # Initialize trainer
    trainer = CaseLinkTrainer(
        preprocessed_dir="data/preprocessed",
        output_path="output/caselink",
        batch_size=2048,
        epochs=50,
        learning_rate=1e-3,
        degree_reg_weight=1e-3,
        include_semantic_edges=True,
        include_article_nodes=True,
        num_hops=1,
        semantic_threshold=0.0,
        semantic_max_neighbors=5,
        gradient_clip_val=None,
        eval_every_n_epochs=1,
    )

    # Initialize model
    model = CaseLinkGNN(
        input_dim=384,
        hidden_dim=384,
        output_dim=384,
        num_layers=1,
        num_edge_types=6,
    )

    # Train
    trained_model = trainer.train(
        model,
        train_cutoff_year=2018,
        val_cutoff_year=2022,
    )

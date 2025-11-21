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
):
    """
    In-batch negative contrastive loss with temporal filtering.

    anchor: [batch_size, dim] - query embeddings
    positive: [batch_size, dim] - positive document embeddings
    anchor_times: [batch_size] - timestamps for anchor nodes (optional)
    positive_times: [batch_size] - timestamps for positive nodes (optional)
    anchor_indices: [batch_size] - original anchor node indices (optional, for masking same-anchor positives)

    For each anchor_i, positive_i is the target, and only positives that come
    BEFORE anchor_i in time are used as negatives.
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    # Compute similarity matrix: [batch_size, batch_size]
    # sim_matrix[i, j] = similarity between anchor_i and positive_j
    sim_matrix = torch.mm(anchor, positive.t()) / temperature

    batch_size = sim_matrix.size(0)
    diagonal_mask = torch.eye(batch_size, dtype=torch.bool, device=sim_matrix.device)

    # Mask out same-anchor positives (they're not negatives)
    if anchor_indices is not None:
        # same_anchor[i, j] = True if anchor_indices[i] == anchor_indices[j]
        same_anchor = anchor_indices.unsqueeze(1) == anchor_indices.unsqueeze(0)
        # Keep diagonal, mask out other same-anchor pairs
        positive_mask = same_anchor & ~diagonal_mask
        sim_matrix = sim_matrix.masked_fill(positive_mask, float("-inf"))

    # Apply temporal masking if time information is provided
    if anchor_times is not None and positive_times is not None:
        # Create mask: positive_j is valid negative for anchor_i only if positive_time_j < anchor_time_i
        # Shape: [batch_size_anchor, batch_size_positive]
        time_mask = positive_times.unsqueeze(0) < anchor_times.unsqueeze(1)

        # Ensure diagonal (positive pairs) are always valid
        time_mask = time_mask | diagonal_mask

        # Apply mask: set invalid negatives to very low value
        sim_matrix = sim_matrix.masked_fill(~time_mask, float("-inf"))

    # Labels: for each anchor_i, the positive is at position i (diagonal)
    labels = torch.arange(sim_matrix.size(0), device=sim_matrix.device)

    # Cross entropy loss treats each row as logits where correct class is on diagonal
    loss = F.cross_entropy(sim_matrix, labels)

    return loss


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
        checkpoint_interval: int = 25,  # Save checkpoint every N epochs
        wandb_project: str | None = "gnn-training",
        wandb_name: str | None = None,
        warmup_epochs: int = 3,
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

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")

    def _process_batch(
        self,
        batch,
        is_hetero: bool,
    ):
        """Process a batch and extract necessary components."""
        # Handle heterogeneous vs homogeneous graphs
        if is_hetero:
            # For heterogeneous graphs, work with paragraph nodes
            batch_size = batch["paragraph"].batch_size
            x = batch["paragraph"].x.clone()

            # Use query embeddings for anchor nodes
            if hasattr(batch["paragraph"], "x_query"):
                x[:batch_size] = batch["paragraph"].x_query[:batch_size]

            # Extract time information for anchor nodes
            if hasattr(batch["paragraph"], "time"):
                anchor_times = batch["paragraph"].time[:batch_size]
            else:
                anchor_times = None

            # Get citation edges (for positive sampling)
            if ("paragraph", "cites", "paragraph") in batch.edge_types:
                cite_edge_index = batch["paragraph", "cites", "paragraph"].edge_index
            else:
                # No citation edges in this batch, skip
                return None

            # Mask citation edges to prevent leakage
            # 1. Identify paragraphs in the same case as anchor nodes (batch_size)
            # Get belongs_to edges to find which cases the anchor paragraphs belong to
            if ("paragraph", "belongs_to", "case") in batch.edge_types:
                par_to_case = batch["paragraph", "belongs_to", "case"].edge_index
                case_to_par = batch["case", "contains", "paragraph"].edge_index

                # Find cases that anchor paragraphs belong to
                anchor_mask = par_to_case[0] < batch_size
                anchor_cases = par_to_case[1, anchor_mask].unique()

                # Find all paragraphs in those cases
                case_mask = torch.isin(case_to_par[0], anchor_cases)
                paragraphs_in_anchor_cases = case_to_par[1, case_mask].unique()
            else:
                # If no case structure, only mask anchor paragraphs
                paragraphs_in_anchor_cases = torch.arange(
                    batch_size, device=self.device
                )

            # 2. Mask citation edges where src or tgt involve anchor cases
            cite_src, cite_tgt = cite_edge_index
            # Mask edges where source AND target are in anchor cases
            leakage_mask = torch.isin(
                cite_src, paragraphs_in_anchor_cases
            ) | torch.isin(cite_tgt, paragraphs_in_anchor_cases)
            masked_cite_edges = cite_edge_index[:, ~leakage_mask]

            # Create modified batch with masked citation edges and updated features
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

            # Use query embeddings for anchor nodes
            if hasattr(batch, "x_query"):
                x[:batch_size] = batch.x_query[:batch_size]

            # Extract time information for anchor nodes
            if hasattr(batch, "time"):
                anchor_times = batch.time[:batch_size]
            else:
                anchor_times = None

            edge_index = batch.edge_index

            # Mask edges to prevent leakage
            # Remove edges where BOTH src and tgt are in anchor batch
            src, tgt = edge_index
            leakage_mask = ~((src < batch_size) | (tgt < batch_size))
            masked_edge_index = edge_index[:, leakage_mask]

            return {
                "batch_size": batch_size,
                "modified_batch": None,
                "edge_index": edge_index,
                "x": x,
                "masked_edge_index": masked_edge_index,
                "anchor_times": anchor_times,
                "all_times": batch.time if hasattr(batch, "time") else None,
            }

    def _compute_loss(
        self,
        model: nn.Module,
        batch_data: dict,
        is_hetero: bool,
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
            out = model(batch_data["x"], batch_data["masked_edge_index"])
            embeddings = out["paragraph"] if isinstance(out, dict) else out

        # Find edges where source is in the input batch (for positive sampling)
        src, tgt = edge_index
        input_mask = src < batch_size
        if input_mask.sum() == 0:
            # No edges for this batch, skip
            return None

        batch_src = src[input_mask]
        batch_tgt = tgt[input_mask]

        # Use all positive pairs instead of sampling one
        anchor_emb = embeddings[batch_src]
        positive_emb = embeddings[batch_tgt]

        # Get times for all pairs
        pair_anchor_times = None
        pair_positive_times = None
        if all_times is not None:
            pair_anchor_times = all_times[batch_src]
            pair_positive_times = all_times[batch_tgt]

        # Compute loss with in-batch negatives (temporally filtered)
        # Pass anchor_indices to mask same-anchor positives
        loss = info_nce_loss(
            anchor_emb,
            positive_emb,
            self.temperature,
            anchor_times=pair_anchor_times,
            positive_times=pair_positive_times,
            anchor_indices=batch_src,
        )

        return loss

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

        for batch_idx, batch in enumerate(
            tqdm(loader, desc="Training batches", leave=False)
        ):
            batch_data = self._process_batch(batch, is_hetero)
            if batch_data is None:
                continue

            loss = self._compute_loss(model, batch_data, is_hetero)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            batch_counter += 1

            # Log batch-level metrics to wandb
            if self.wandb_project is not None and batch_idx % 100 == 0:
                wandb.log(
                    {
                        "train/batch_loss": loss.item(),
                        "train/learning_rate": optimizer.param_groups[0]["lr"],
                        "train/batch": batch_counter,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                    }
                )

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss, batch_counter

    def train(
        self,
        gnn_model: nn.Module,
        train_cutoff_year: int | None = None,
    ) -> torch.nn.Module:
        """
        Train GNN model using preprocessed data from graph builder.

        Args:
            gnn_model: The GNN model to train
            train_cutoff_year: Cutoff year for training data (e.g., 2018)
        """
        os.makedirs(self.output_path, exist_ok=True)
        checkpoint_dir = os.path.join(self.output_path, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Training GNN with {self.graph_type.capitalize()} Graph Builder")
        print("=" * 80)

        # Build graph based on type
        is_hetero = self.graph_type == "heterogeneous"

        # Type declarations
        train_graph_data: Data | HeteroData
        input_nodes: tuple[str, torch.Tensor] | torch.Tensor | None

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
            ).to(self.device)

        # Filter to only sample nodes with at least one positive (citation) edge
        print("\nFiltering nodes with positive examples...")
        if is_hetero:
            # Get citation edge index
            cite_edge_index = train_graph_data[
                "paragraph", "cites", "paragraph"
            ].edge_index
            # Find unique source nodes (nodes that cite others)
            nodes_with_positives = cite_edge_index[0].unique()
            print(
                f"  Paragraph nodes with citations: {len(nodes_with_positives)} / {train_graph_data['paragraph'].num_nodes}"
            )
            input_nodes = ("paragraph", nodes_with_positives)
        else:
            # For homogeneous graphs, find nodes with outgoing edges
            edge_index = train_graph_data.edge_index
            nodes_with_positives = edge_index[0].unique()
            print(
                f"  Nodes with citations: {len(nodes_with_positives)} / {train_graph_data.num_nodes}"
            )
            input_nodes = nodes_with_positives

        train_graph_data = ToUndirected()(train_graph_data)

        # Initialize wandb
        if self.wandb_project is not None:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_name,
                config={
                    "batch_size": self.batch_size,
                    "epochs": self.epochs,
                    "learning_rate": self.learning_rate,
                    "weight_decay": self.weight_decay,
                    "temperature": self.temperature,
                    "num_hops": self.num_hops,
                    "graph_type": self.graph_type,
                    "checkpoint_interval": self.checkpoint_interval,
                    "train_cutoff_year": train_cutoff_year,
                    "device": str(self.device),
                    "num_nodes_with_positives": len(nodes_with_positives),
                },
            )

        # Initialize model
        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        # Log model architecture to wandb
        if self.wandb_project is not None:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            wandb.config.update(
                {
                    "model/total_parameters": total_params,
                    "model/trainable_parameters": trainable_params,
                }
            )
            # Log model architecture
            wandb.watch(model, log="all", log_freq=100)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Create data loaders
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

        # Calculate steps
        steps_per_epoch = len(train_loader)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warmup_epochs * steps_per_epoch

        # Simple warmup + cosine decay
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / warmup_steps

            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

        print(f"\nStarting training for {self.epochs} epochs...")
        print(f"Checkpoints will be saved every {self.checkpoint_interval} epochs")

        best_loss = float("inf")
        global_batch_counter = 0

        for epoch in range(self.epochs):
            train_loss, global_batch_counter = self.train_epoch(
                model,
                train_loader,
                optimizer,
                scheduler,
                is_hetero,
                epoch,
                global_batch_counter,
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            # Log epoch-level metrics to wandb
            if self.wandb_project is not None:
                wandb.log(
                    {
                        "train/epoch_loss": train_loss,
                        "epoch": epoch + 1,
                    }
                )

            # Track best model
            if train_loss < best_loss:
                torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                print(
                    f"  ✓ New best training loss: {best_loss:.4f} -> {train_loss:.4f}"
                )
                best_loss = train_loss

                # Log best loss to wandb
                if self.wandb_project is not None:
                    wandb.run.summary["best_train_loss"] = best_loss

            # Save checkpoint at specified intervals
            if (epoch + 1) % self.checkpoint_interval == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pt")
                torch.save(
                    model.state_dict(),
                    checkpoint_path,
                )
                print(f"  ✓ Checkpoint saved: {checkpoint_path}")

                # Log checkpoint to wandb
                if self.wandb_project is not None:
                    wandb.save(checkpoint_path)

        # Save final model
        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")

        # Load best model
        print(f"\nLoading best model (train loss: {best_loss:.4f})...")
        model.load_state_dict(torch.load(f"{self.output_path}/best_model.pt"))

        print(
            f"Training complete! Best model saved to {self.output_path}/best_model.pt"
        )
        print(f"Checkpoints saved to {checkpoint_dir}/")

        # Finish wandb run
        if self.wandb_project is not None:
            wandb.finish()

        return model

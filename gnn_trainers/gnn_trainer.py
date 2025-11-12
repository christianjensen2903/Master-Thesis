import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.data import Data, HeteroData  # type: ignore
from tqdm import tqdm  # type: ignore
from preprocessing.graph_builder import HomogeneousGraphBuilder, HeterogeneousGraphBuilder


def info_nce_loss(anchor, positive, temperature=0.07):
    """
    In-batch negative contrastive loss.

    anchor: [batch_size, dim] - query embeddings
    positive: [batch_size, dim] - positive document embeddings

    For each anchor_i, positive_i is the target, and all other positives
    in the batch are used as negatives.
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)

    # Compute similarity matrix: [batch_size, batch_size]
    # sim_matrix[i, j] = similarity between anchor_i and positive_j
    sim_matrix = torch.mm(anchor, positive.t()) / temperature

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
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_hops: int = 2,
        graph_type: str = "heterogeneous",
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

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")
        print(f"Using graph type: {self.graph_type}")

    def train_epoch(
        self, model: nn.Module, loader: NeighborLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Training batches", leave=False):
            # Get batch data
            batch_size = batch.batch_size
            edge_index = batch.edge_index

            # Create combined feature matrix:
            # - Anchor nodes (first batch_size nodes) use query embeddings (masked)
            # - All other nodes (positives, neighbors) use document embeddings
            x = batch.x
            if hasattr(batch, "x_query"):
                x[:batch_size] = batch.x_query[:batch_size]

            # Mask edges to prevent leakage
            src, tgt = edge_index
            edge_mask = (src < batch_size) | (tgt < batch_size)
            masked_edge_index = edge_index[:, edge_mask]

            # Get embeddings for nodes in this batch with masked edges
            embeddings = model(x, masked_edge_index)

            anchor_emb = embeddings[:batch_size]

            # Find edges where source is in the input batch
            input_mask = src < batch_size
            if input_mask.sum() == 0:
                # No edges for this batch, skip
                continue

            batch_src = src[input_mask]
            batch_tgt = tgt[input_mask]

            # Sort edges by source for efficient grouping
            sorted_idx = torch.argsort(batch_src)
            src_sorted = batch_src[sorted_idx]
            tgt_sorted = batch_tgt[sorted_idx]

            # Find unique sources and their edge counts
            unique_src, counts = torch.unique_consecutive(
                src_sorted, return_counts=True
            )

            # Random sampling: generate random offset for each unique source
            random_offsets = (torch.rand_like(counts.float()) * counts.float()).long()

            # Compute cumulative positions
            cumsum = torch.cat(
                [torch.tensor([0], device=self.device), counts.cumsum(0)[:-1]]
            )
            selected_edges = cumsum + random_offsets

            # Get positive samples
            positive_indices = torch.arange(
                batch_size, device=self.device
            )  # Default to self
            positive_indices[unique_src] = tgt_sorted[selected_edges]

            positive_emb = embeddings[positive_indices]

            # Compute loss with in-batch negatives
            loss = info_nce_loss(anchor_emb, positive_emb, self.temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self, gnn_model: nn.Module, cutoff_year: int | None = None
    ) -> torch.nn.Module:
        """Train GNN model using preprocessed data from graph builder."""
        os.makedirs(self.output_path, exist_ok=True)

        print("\n" + "=" * 80)
        print("Training GNN with Graph Builder")
        print("=" * 80)

        builder = HomogeneousGraphBuilder(self.preprocessed_dir)
        graph_data = builder.build_graph(
            train_cutoff_year=cutoff_year,
            include_only_citing=True,
        ).to(self.device)

        # Initialize model
        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Create data loader
        # Add 1 to num_neighbors to get all info for positive nodes
        loader = NeighborLoader(
            graph_data,
            num_neighbors=[-1] * (self.num_hops + 1) if self.num_hops > 0 else [-1],
            batch_size=self.batch_size,
            input_nodes=None,  # Sample all nodes
            shuffle=True,
            time_attr="time",
            subgraph_type="bidirectional",
        )

        print(f"\nStarting training for {self.epochs} epochs...")

        best_loss = float("inf")

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(model, loader, optimizer)

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            if train_loss < best_loss:

                torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                print(f"New best loss: {best_loss:.4f} -> {train_loss:.4f}")
                best_loss = train_loss

            scheduler.step()

        # Save final model
        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")
        print(f"\nModel saved to {self.output_path}/final_model.pt")

        return model

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.loader import NeighborLoader  # type: ignore
from torch_geometric.data import Data, HeteroData  # type: ignore
from tqdm import tqdm  # type: ignore
from preprocessing.graph_builder import (
    HomogeneousGraphBuilder,
    HeterogeneousGraphBuilder,
)


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
        patience: int = 3,  # Early stopping patience
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
            }

        else:
            # For homogeneous graphs
            batch_size = batch.batch_size
            x = batch.x.clone()

            # Use query embeddings for anchor nodes
            if hasattr(batch, "x_query"):
                x[:batch_size] = batch.x_query[:batch_size]

            edge_index = batch.edge_index

            # Mask edges to prevent leakage
            # Remove edges where BOTH src and tgt are in anchor batch
            src, tgt = edge_index
            leakage_mask = (src < batch_size) & (tgt < batch_size)
            masked_edge_index = edge_index[:, ~leakage_mask]

            return {
                "batch_size": batch_size,
                "modified_batch": None,
                "edge_index": edge_index,
                "x": x,
                "masked_edge_index": masked_edge_index,
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

        # Get embeddings
        if is_hetero:
            out = model(batch_data["modified_batch"])
            embeddings = out["paragraph"] if isinstance(out, dict) else out
        else:
            out = model(batch_data["x"], batch_data["masked_edge_index"])
            embeddings = out["paragraph"] if isinstance(out, dict) else out

        anchor_emb = embeddings[:batch_size]

        # Find edges where source is in the input batch (for positive sampling)
        src, tgt = edge_index
        input_mask = src < batch_size
        if input_mask.sum() == 0:
            # No edges for this batch, skip
            return None

        batch_src = src[input_mask]
        batch_tgt = tgt[input_mask]

        # Sort edges by source for efficient grouping
        sorted_idx = torch.argsort(batch_src)
        src_sorted = batch_src[sorted_idx]
        tgt_sorted = batch_tgt[sorted_idx]

        # Find unique sources and their edge counts
        unique_src, counts = torch.unique_consecutive(src_sorted, return_counts=True)

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

        return loss

    def train_epoch(
        self,
        model: nn.Module,
        loader: NeighborLoader,
        optimizer: torch.optim.Optimizer,
        is_hetero: bool,
    ) -> float:
        """Train for one epoch."""
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Training batches", leave=False):
            batch_data = self._process_batch(batch, is_hetero)
            if batch_data is None:
                continue

            loss = self._compute_loss(model, batch_data, is_hetero)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def validate(
        self,
        model: nn.Module,
        loader: NeighborLoader,
        is_hetero: bool,
    ) -> float:
        """Validate the model."""
        model.eval()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Validation batches", leave=False):
            batch_data = self._process_batch(batch, is_hetero)
            if batch_data is None:
                continue

            loss = self._compute_loss(model, batch_data, is_hetero)
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
        """
        Train GNN model using preprocessed data from graph builder.

        Args:
            gnn_model: The GNN model to train
            train_cutoff_year: Cutoff year for training data (e.g., 2018)
            val_cutoff_year: Cutoff year for validation data (e.g., 2019)
                            Should be after train_cutoff_year. If None, no validation.
        """
        os.makedirs(self.output_path, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Training GNN with {self.graph_type.capitalize()} Graph Builder")
        print("=" * 80)

        # Build graph based on type
        is_hetero = self.graph_type == "heterogeneous"

        # Type declarations
        train_graph_data: Data | HeteroData
        val_graph_data: Data | HeteroData | None
        input_nodes: tuple[str, None] | None

        if is_hetero:
            hetero_builder = HeterogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = hetero_builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
            ).to(self.device)

            # Build validation graph if cutoff provided
            if val_cutoff_year is not None:
                print(
                    f"\nBuilding validation graph (cutoff year: {val_cutoff_year})..."
                )
                val_graph_data = hetero_builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=True,
                ).to(self.device)
            else:
                val_graph_data = None
                print("\nNo validation set - training without validation")

            input_nodes = ("paragraph", None)
        else:
            homo_builder = HomogeneousGraphBuilder(self.preprocessed_dir)
            train_graph_data = homo_builder.build_graph(
                train_cutoff_year=train_cutoff_year,
                include_only_citing=True,
            ).to(self.device)

            # Build validation graph if cutoff provided
            if val_cutoff_year is not None:
                print(
                    f"\nBuilding validation graph (cutoff year: {val_cutoff_year})..."
                )
                val_graph_data = homo_builder.build_graph(
                    train_cutoff_year=val_cutoff_year,
                    include_only_citing=True,
                ).to(self.device)
            else:
                val_graph_data = None
                print("\nNo validation set - training without validation")

            input_nodes = None

        # Initialize model
        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

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
            )

        print(f"\nStarting training for {self.epochs} epochs...")

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer, is_hetero)

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            # Validate if validation set is available
            if val_loader is not None:
                val_loss = self.validate(model, val_loader, is_hetero)
                print(f"  Val Loss:   {val_loss:.4f}")

                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best validation loss! (improved by {improvement:.4f})"
                    )
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    print(
                        f"  ✗ No improvement ({epochs_without_improvement}/{self.patience})"
                    )

                # Early stopping
                if epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
            else:
                # No validation set - save based on training loss
                if train_loss < best_val_loss:
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(
                        f"  ✓ New best training loss: {best_val_loss:.4f} -> {train_loss:.4f}"
                    )
                    best_val_loss = train_loss

            scheduler.step()

        # Save final model
        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")

        # Load best model
        print(f"\nLoading best model (val loss: {best_val_loss:.4f})...")
        model.load_state_dict(torch.load(f"{self.output_path}/best_model.pt"))

        print(
            f"Training complete! Best model saved to {self.output_path}/best_model.pt"
        )

        return model

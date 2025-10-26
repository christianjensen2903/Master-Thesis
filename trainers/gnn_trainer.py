import os
import random
import pandas as pd  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data  # type: ignore
from torch_geometric.loader import NeighborLoader  # type: ignore
from tqdm import tqdm  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from retrievers.gnn_retriever import GNNEncoder
from utils.temporal_graph import (
    build_temporal_dag,
    validate_temporal_dag,
    print_temporal_graph_stats,
)
from .base_trainer import BaseTrainer


class GNNTrainer(BaseTrainer):
    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_negatives: int = 5,
        neighbor_batch_size: int = 32,
        num_neighbors: list[int] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_encoder_name = text_encoder_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.neighbor_batch_size = neighbor_batch_size
        # Fewer neighbors per layer to keep subgraphs small
        self.num_neighbors = num_neighbors or [10, 5]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

    def build_citation_graph_from_df(
        self, df: pd.DataFrame
    ) -> tuple[dict[int, list[int]], dict[str, int], np.ndarray, np.ndarray]:
        """
        Build citation graph using (CELEX, PARAGRAPH_NUMBER) as unique node IDs.

        This ensures:
        - Same text in different cases = different nodes
        - No date ambiguity (each case has one date)
        - Preserved provenance (know which case each paragraph comes from)
        """
        # Build unique paragraph IDs: CELEX::NUMBER
        if "FROM_ID" not in df.columns:
            df["FROM_ID"] = (
                df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
            )
        if "TO_ID" not in df.columns:
            df["TO_ID"] = (
                df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)
            )

        # Collect all unique paragraph IDs and their texts
        all_from_data = df[["FROM_ID", "TEXT_FROM", "DATE_FROM"]].drop_duplicates(
            subset=["FROM_ID"]
        )
        all_to_data = df[["TO_ID", "TEXT_TO", "DATE_TO"]].drop_duplicates(
            subset=["TO_ID"]
        )

        # Combine and create unified mapping
        combined_ids = pd.concat(
            [
                all_from_data.rename(
                    columns={
                        "FROM_ID": "PAR_ID",
                        "TEXT_FROM": "TEXT",
                        "DATE_FROM": "DATE",
                    }
                ),
                all_to_data.rename(
                    columns={"TO_ID": "PAR_ID", "TEXT_TO": "TEXT", "DATE_TO": "DATE"}
                ),
            ]
        ).drop_duplicates(subset=["PAR_ID"])

        # Create ID to integer mapping
        par_id_to_idx = {
            str(par_id): i for i, par_id in enumerate(combined_ids["PAR_ID"])
        }

        # Extract texts and dates arrays
        all_texts = combined_ids["TEXT"].fillna("").values
        paragraph_dates = pd.to_datetime(combined_ids["DATE"]).values

        # Build citation graph
        citation_graph: dict[int, list[int]] = {}
        for _, row in df.iterrows():
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])

            if from_id in par_id_to_idx and to_id in par_id_to_idx:
                src_idx = par_id_to_idx[from_id]
                tgt_idx = par_id_to_idx[to_id]

                if src_idx not in citation_graph:
                    citation_graph[src_idx] = []
                citation_graph[src_idx].append(tgt_idx)

        return citation_graph, par_id_to_idx, all_texts, paragraph_dates

    def build_graph_data(
        self,
        texts: np.ndarray,
        citation_graph: dict[int, list[int]],
        text_encoder: SentenceTransformer,
    ) -> Data:
        print("Encoding texts...")
        text_embeddings = text_encoder.encode(
            texts.tolist(),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        x = torch.tensor(text_embeddings, dtype=torch.float32)

        # Build edge index from citation graph
        edge_list = []
        for src, targets in citation_graph.items():
            for tgt in targets:
                if src < len(texts) and tgt < len(texts):
                    edge_list.append([src, tgt])
                    # Add reverse edge for undirected message passing
                    edge_list.append([tgt, src])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        print(f"Graph: {len(texts)} nodes, {edge_index.shape[1]} edges")

        return Data(x=x, edge_index=edge_index, num_nodes=len(texts))

    def contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        # Normalize embeddings
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negatives = F.normalize(negatives, p=2, dim=2)

        # Positive similarity
        pos_sim = (
            torch.sum(anchor * positive, dim=1) / self.temperature
        )  # (batch_size,)

        # Negative similarities
        neg_sim = (
            torch.bmm(negatives, anchor.unsqueeze(2)).squeeze(2) / self.temperature
        )  # (batch_size, num_negatives)

        # InfoNCE loss
        logits = torch.cat(
            [pos_sim.unsqueeze(1), neg_sim], dim=1
        )  # (batch_size, 1 + num_negatives)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)
        return loss

    def train_epoch(
        self,
        model: nn.Module,
        graph_data: Data,
        citation_graph: dict[int, list[int]],
        optimizer: torch.optim.Optimizer,
        epoch: int,
    ) -> float:
        model.train()
        total_loss = 0.0
        num_batches = 0

        # Mini-batch over subgraphs to avoid full-graph OOM
        # Use only nodes with citations as seeds to maximize useful pairs
        seed_nodes_with_citations = torch.tensor(
            list(citation_graph.keys()), dtype=torch.long
        )
        loader = NeighborLoader(
            graph_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.neighbor_batch_size,
            shuffle=True,
            input_nodes=seed_nodes_with_citations,
            num_workers=0,
        )

        for batch in tqdm(loader, desc=f"Epoch {epoch + 1}"):
            # Map from original ids to local indices in subgraph
            n_id: torch.Tensor = batch.n_id
            local_index_of = {int(orig): int(local) for local, orig in enumerate(n_id)}

            # Seed nodes are the first `batch_size` nodes
            seed_count = int(
                getattr(batch, "batch_size", min(len(n_id), self.neighbor_batch_size))
            )
            seed_orig_ids = [int(i) for i in n_id[:seed_count].tolist()]

            # Build anchor/positive pairs where positive exists inside the subgraph
            anchor_orig: list[int] = []
            positive_orig: list[int] = []
            for a in seed_orig_ids:
                cited = citation_graph.get(a, [])
                # Keep only positives present in this subgraph
                cited_in_sub = [p for p in cited if p in local_index_of]
                if cited_in_sub:
                    anchor_orig.append(a)
                    positive_orig.append(random.choice(cited_in_sub))

            if not anchor_orig:
                continue

            # Sample negatives from nodes present in this subgraph (excluding cited + anchor)
            all_local_nodes = [int(i) for i in range(len(n_id))]
            all_orig_nodes = [int(o) for o in n_id.tolist()]
            negative_local: list[list[int]] = []
            for a in anchor_orig:
                cited_set = set(citation_graph.get(a, []))
                excluded_orig = cited_set.union({a})
                candidates_orig = [o for o in all_orig_nodes if o not in excluded_orig]
                if not candidates_orig:
                    # Fallback to any local node except anchor
                    candidates_orig = [o for o in all_orig_nodes if o != a]
                # Map to local indices and sample
                candidates_local = [local_index_of[o] for o in candidates_orig]
                if len(candidates_local) >= self.num_negatives:
                    sampled = random.sample(candidates_local, self.num_negatives)
                else:
                    # Pad with random choices if needed
                    sampled = candidates_local.copy()
                    while len(sampled) < self.num_negatives and candidates_local:
                        sampled.append(random.choice(candidates_local))
                    while len(sampled) < self.num_negatives:
                        sampled.append(0)
                negative_local.append(sampled)

            # Forward on subgraph
            x = batch.x.to(self.device)
            edge_index = batch.edge_index.to(self.device)
            embeddings_local = model(x, edge_index)

            # Gather embeddings for anchors/positives/negatives
            anchor_local = torch.tensor(
                [local_index_of[a] for a in anchor_orig],
                dtype=torch.long,
                device=self.device,
            )
            positive_local = torch.tensor(
                [local_index_of[p] for p in positive_orig],
                dtype=torch.long,
                device=self.device,
            )
            negative_local_t = torch.tensor(
                negative_local, dtype=torch.long, device=self.device
            )

            anchor_emb = embeddings_local[anchor_local]
            positive_emb = embeddings_local[positive_local]
            negative_emb = embeddings_local[negative_local_t]

            # Compute loss
            loss = self.contrastive_loss(anchor_emb, positive_emb, negative_emb)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += float(loss.item())
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        graph_data: Data,
        val_citation_graph: dict[int, list[int]],
    ) -> dict[str, float]:
        model.eval()

        # Compute embeddings for all nodes via NeighborLoader in mini-batches
        embeddings = np.zeros((graph_data.num_nodes, self.output_dim), dtype=np.float32)
        loader = NeighborLoader(
            graph_data,
            num_neighbors=self.num_neighbors,
            batch_size=self.neighbor_batch_size,
            shuffle=False,
        )

        with torch.no_grad():
            for batch in tqdm(loader, desc="Evaluating"):
                if int(getattr(batch, "batch_size", 0)) == 0:
                    continue
                seed_cnt = int(getattr(batch, "batch_size", len(batch.n_id)))
                x = batch.x.to(self.device)
                edge_index = batch.edge_index.to(self.device)
                out = model(x, edge_index)
                out = F.normalize(out[:seed_cnt], p=2, dim=1).cpu().numpy()
                seed_orig = batch.n_id[:seed_cnt].cpu().numpy()
                embeddings[seed_orig] = out

        # Compute MRR and Recall@k
        ranks = []
        recall_at_5 = 0
        recall_at_10 = 0
        recall_at_50 = 0

        for anchor_id, cited_ids in val_citation_graph.items():
            if not cited_ids or anchor_id >= len(embeddings):
                continue

            anchor_emb = embeddings[anchor_id]
            similarities = embeddings @ anchor_emb

            # Rank all candidates
            ranked_indices = np.argsort(-similarities)

            # Find rank of positive samples
            for cited_id in cited_ids:
                if cited_id >= len(embeddings):
                    continue
                rank = np.where(ranked_indices == cited_id)[0]
                if len(rank) > 0:
                    rank = rank[0] + 1  # 1-indexed rank
                    ranks.append(rank)
                    if rank <= 5:
                        recall_at_5 += 1
                    if rank <= 10:
                        recall_at_10 += 1
                    if rank <= 50:
                        recall_at_50 += 1

        if ranks:
            mrr = float(np.mean([1.0 / r for r in ranks]))
            recall_5 = float(recall_at_5 / len(ranks))
            recall_10 = float(recall_at_10 / len(ranks))
            recall_50 = float(recall_at_50 / len(ranks))
        else:
            mrr = recall_5 = recall_10 = recall_50 = 0.0

        return {
            "mrr": mrr,
            "recall@5": recall_5,
            "recall@10": recall_10,
            "recall@50": recall_50,
        }

    def train(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
    ) -> torch.nn.Module:
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

        # Load and split data
        train_df, val_df = self.load_and_split_data(paragraph_file, cutoff_date)

        # Build citation graphs for train and validation
        print("\nBuilding training graph...")
        train_citation_graph, par_id_to_idx, all_texts, paragraph_dates = (
            self.build_citation_graph_from_df(train_df)
        )

        # Build temporal DAG (respects causality: older → newer only)
        print("\nBuilding temporal DAG...")
        temporal_dag = build_temporal_dag(train_citation_graph, paragraph_dates)

        # Validate temporal DAG
        print("\nValidating temporal DAG...")
        if not validate_temporal_dag(temporal_dag, paragraph_dates):
            raise ValueError("Invalid temporal DAG structure!")

        # Print statistics
        print_temporal_graph_stats(temporal_dag, paragraph_dates)

        print("\nBuilding validation graph...")
        # Build validation paragraph IDs if not present
        if "FROM_ID" not in val_df.columns:
            val_df["FROM_ID"] = (
                val_df["CELEX_FROM"].astype(str)
                + "::"
                + val_df["NUMBER_FROM"].astype(str)
            )
        if "TO_ID" not in val_df.columns:
            val_df["TO_ID"] = (
                val_df["CELEX_TO"].astype(str) + "::" + val_df["NUMBER_TO"].astype(str)
            )

        val_citation_graph: dict[int, list[int]] = {}
        for _, row in val_df.iterrows():
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])
            # Only include if both paragraphs are in training data
            if from_id in par_id_to_idx and to_id in par_id_to_idx:
                src_idx = par_id_to_idx[from_id]
                tgt_idx = par_id_to_idx[to_id]
                if src_idx not in val_citation_graph:
                    val_citation_graph[src_idx] = []
                val_citation_graph[src_idx].append(tgt_idx)

        # Initialize text encoder
        print(f"\nLoading text encoder: {self.text_encoder_name}")
        text_encoder = SentenceTransformer(self.text_encoder_name)
        input_dim = text_encoder.get_sentence_embedding_dimension()
        if input_dim is None:
            raise ValueError("Text encoder does not provide embedding dimension")

        # Build graph data using temporal DAG (causally masked)
        graph_data = self.build_graph_data(all_texts, temporal_dag, text_encoder)

        # Initialize GNN model
        print(f"\nInitializing GNN model...")
        model = GNNEncoder(
            input_dim=input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Setup wandb
        config = {
            "model_type": "GNN-GAT",
            "text_encoder": self.text_encoder_name,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "num_negatives": self.num_negatives,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "num_nodes": len(all_texts),
            "num_train_edges": sum(len(v) for k, v in temporal_dag.items()),
            "num_val_edges": sum(len(v) for k, v in val_citation_graph.items()),
        }
        self.setup_wandb(config)

        # Training loop
        print(f"\nStarting training for {self.epochs} epochs...")
        best_mrr = 0.0

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(
                model, graph_data, temporal_dag, optimizer, epoch
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            # Evaluate on validation set
            if (epoch + 1) % max(1, self.epochs // 5) == 0:
                val_metrics = self.evaluate(model, graph_data, val_citation_graph)
                print(f"  Validation MRR: {val_metrics['mrr']:.4f}")
                print(f"  Recall@5: {val_metrics['recall@5']:.4f}")
                print(f"  Recall@10: {val_metrics['recall@10']:.4f}")
                print(f"  Recall@50: {val_metrics['recall@50']:.4f}")

                if self.use_wandb:
                    import wandb

                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_mrr": val_metrics["mrr"],
                            "val_recall@5": val_metrics["recall@5"],
                            "val_recall@10": val_metrics["recall@10"],
                            "val_recall@50": val_metrics["recall@50"],
                        }
                    )

                # Save best model
                if val_metrics["mrr"] > best_mrr:
                    best_mrr = val_metrics["mrr"]
                    print(f"  New best MRR: {best_mrr:.4f}")
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_mrr": best_mrr,
                            "config": config,
                        },
                        f"{self.output_path}/best_model.pt",
                    )

            scheduler.step()

        # Save final model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
            },
            f"{self.output_path}/final_model.pt",
        )

        print(f"\nTraining complete! Best validation MRR: {best_mrr:.4f}")
        print(f"Model saved to {self.output_path}")

        self.cleanup_wandb()

        # Return the text encoder used for training
        return model

import os
import random
import pickle
import hashlib
import wandb
import pandas as pd  # type: ignore
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data  # type: ignore
from tqdm import tqdm  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from utils.temporal_graph import (
    build_temporal_dag,
    validate_temporal_dag,
    print_temporal_graph_stats,
)
from validation_utils import split_data_by_date


class GNNTrainer:
    def __init__(
        self,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_path: str = "output/gnn",
        batch_size: int = 16,
        epochs: int = 5,
        eval_every_n_epochs: int | None = None,
        validation_split: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        temperature: float = 0.07,
        num_negatives: int = 5,
        use_wandb: bool = True,
        project_name: str = "gnn-training",
        embeddings_cache_dir: str | None = None,
    ):
        self.text_encoder_name = text_encoder_name
        self.output_path = output_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.eval_every_n_epochs = eval_every_n_epochs
        self.validation_split = validation_split
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.num_negatives = num_negatives
        self.use_wandb = use_wandb
        self.project_name = project_name
        self.embeddings_cache_dir = embeddings_cache_dir

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Create cache directory if specified
        if self.embeddings_cache_dir:
            os.makedirs(self.embeddings_cache_dir, exist_ok=True)

    def load_and_split_data(
        self, paragraph_file: str, cutoff_date: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and split into train/validation sets."""
        df = pd.read_csv(paragraph_file)
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])

        # Add DATE_TO if it exists
        if "DATE_TO" in df.columns:
            df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])

        # Add ID columns if they don't exist
        if "FROM_ID" not in df.columns and "CELEX_FROM" in df.columns:
            df["FROM_ID"] = (
                df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
            )
        if "TO_ID" not in df.columns and "CELEX_TO" in df.columns:
            df["TO_ID"] = (
                df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)
            )

        return split_data_by_date(df, cutoff_date, self.validation_split)

    def setup_wandb(self, config: dict) -> None:
        """Initialize wandb with the given config."""
        if self.use_wandb:
            wandb.init(project=self.project_name, config=config)

    def cleanup_wandb(self) -> None:
        """Save model to wandb and finish the run."""
        if self.use_wandb:
            wandb.save(f"{self.output_path}/*")
            wandb.finish()

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

    def _compute_cache_key(self, texts: np.ndarray, encoder_name: str) -> str:
        """Compute a unique cache key based on texts and encoder name."""
        # Create hash from encoder name and texts
        hasher = hashlib.sha256()
        hasher.update(encoder_name.encode("utf-8"))
        for text in texts:
            hasher.update(str(text).encode("utf-8"))
        return hasher.hexdigest()

    def _load_cached_embeddings(self, cache_key: str) -> np.ndarray | None:
        """Load embeddings from cache if they exist."""
        if not self.embeddings_cache_dir:
            return None

        cache_path = os.path.join(self.embeddings_cache_dir, f"{cache_key}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cached_embeddings(self, embeddings: np.ndarray, cache_key: str) -> None:
        """Save embeddings to cache."""
        if not self.embeddings_cache_dir:
            return

        cache_path = os.path.join(self.embeddings_cache_dir, f"{cache_key}.pkl")
        print(f"Saving embeddings to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)

    def build_graph_data(
        self,
        texts: np.ndarray,
        citation_graph: dict[int, list[int]],
        text_encoder: SentenceTransformer,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> Data:
        # Use pre-computed embeddings if provided
        if precomputed_embeddings is not None:
            print("Using pre-computed embeddings...")
            text_embeddings = precomputed_embeddings
        else:
            # Try to load cached embeddings
            cache_key = self._compute_cache_key(texts, self.text_encoder_name)
            cached_embeddings = self._load_cached_embeddings(cache_key)

            if cached_embeddings is None:
                print("Encoding texts...")
                text_embeddings = text_encoder.encode(
                    texts.tolist(),
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                )
                # Save to cache for future use
                self._save_cached_embeddings(text_embeddings, cache_key)
            else:
                print("Using cached embeddings")
                text_embeddings = cached_embeddings

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

        # Move graph to device
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)

        # Forward pass on entire graph
        embeddings = model(x, edge_index)

        # Build anchor/positive pairs from citation graph
        anchor_ids: list[int] = []
        positive_ids: list[int] = []
        for anchor_id, cited_ids in citation_graph.items():
            if cited_ids and anchor_id < len(embeddings):
                # Sample one positive for each anchor
                valid_cited = [c for c in cited_ids if c < len(embeddings)]
                if valid_cited:
                    anchor_ids.append(anchor_id)
                    positive_ids.append(random.choice(valid_cited))

        if not anchor_ids:
            return 0.0

        # Sample negatives (excluding anchor and cited nodes)
        num_nodes = len(embeddings)
        negative_ids: list[list[int]] = []
        for i, anchor_id in enumerate(anchor_ids):
            cited_set = set(citation_graph.get(anchor_id, []))
            excluded = cited_set.union({anchor_id})
            candidates = [n for n in range(num_nodes) if n not in excluded]

            if len(candidates) >= self.num_negatives:
                sampled = random.sample(candidates, self.num_negatives)
            else:
                # Pad with random choices if needed
                sampled = candidates.copy()
                while len(sampled) < self.num_negatives and candidates:
                    sampled.append(random.choice(candidates))
                # Final fallback
                while len(sampled) < self.num_negatives:
                    sampled.append(0)
            negative_ids.append(sampled)

        # Gather embeddings
        anchor_tensor = torch.tensor(anchor_ids, dtype=torch.long, device=self.device)
        positive_tensor = torch.tensor(
            positive_ids, dtype=torch.long, device=self.device
        )
        negative_tensor = torch.tensor(
            negative_ids, dtype=torch.long, device=self.device
        )

        anchor_emb = embeddings[anchor_tensor]
        positive_emb = embeddings[positive_tensor]
        negative_emb = embeddings[negative_tensor]

        # Compute loss
        loss = self.contrastive_loss(anchor_emb, positive_emb, negative_emb)

        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        return float(loss.item())

    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        graph_data: Data,
        val_citation_graph: dict[int, list[int]],
        paragraph_dates: np.ndarray,
    ) -> dict[str, float]:
        """
        Evaluate with proper temporal filtering - only rank against older paragraphs.
        This matches the real evaluation in evaluator.py.
        """
        model.eval()

        # Compute embeddings for all nodes in one forward pass
        x = graph_data.x.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)

        embeddings = model(x, edge_index)
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()

        # Pre-sort by date for temporal filtering
        sort_idx = np.argsort(paragraph_dates)
        sorted_dates = paragraph_dates[sort_idx]

        # Compute MRR, MAP and Recall@k
        ranks = []
        recall_at_5 = 0
        recall_at_10 = 0
        recall_at_50 = 0
        recall_at_100 = 0
        average_precisions = []

        for anchor_id, cited_ids in tqdm(
            val_citation_graph.items(), desc="Evaluating", leave=False
        ):
            if not cited_ids or anchor_id >= len(embeddings):
                continue

            # ✅ CRITICAL: Only rank against temporally valid candidates (strictly older)
            anchor_date = paragraph_dates[anchor_id]
            cutoff = int(np.searchsorted(sorted_dates, anchor_date, side="left"))
            cand_pids = sort_idx[:cutoff]

            if len(cand_pids) == 0:
                continue

            # Filter cited_ids to only those that are temporally valid
            valid_cited_ids = [
                cid
                for cid in cited_ids
                if cid < len(embeddings) and cid in set(cand_pids)
            ]

            if not valid_cited_ids:
                continue

            # Compute similarities only for valid candidates
            anchor_emb = embeddings[anchor_id]
            candidate_embs = embeddings[cand_pids]
            similarities = candidate_embs @ anchor_emb

            # Rank candidates
            ranked_order = np.argsort(-similarities)
            ranked_pids = cand_pids[ranked_order]

            # Find rank of positive samples and compute average precision
            query_ranks = []
            for cited_id in valid_cited_ids:
                rank = np.where(ranked_pids == cited_id)[0]
                if len(rank) > 0:
                    rank_val = rank[0] + 1  # 1-indexed rank
                    ranks.append(rank_val)
                    query_ranks.append(rank_val)
                    if rank_val <= 5:
                        recall_at_5 += 1
                    if rank_val <= 10:
                        recall_at_10 += 1
                    if rank_val <= 50:
                        recall_at_50 += 1
                    if rank_val <= 100:
                        recall_at_100 += 1

            # Compute average precision for this query
            if query_ranks:
                query_ranks_sorted = sorted(query_ranks)
                precisions = []
                for i, rank in enumerate(query_ranks_sorted):
                    precision_at_rank = (i + 1) / rank
                    precisions.append(precision_at_rank)
                avg_precision = np.mean(precisions)
                average_precisions.append(avg_precision)

        if ranks:
            mrr = float(np.mean([1.0 / r for r in ranks]))
            recall_5 = float(recall_at_5 / len(ranks))
            recall_10 = float(recall_at_10 / len(ranks))
            recall_50 = float(recall_at_50 / len(ranks))
            recall_100 = float(recall_at_100 / len(ranks))
            map_score = (
                float(np.mean(average_precisions)) if average_precisions else 0.0
            )
        else:
            mrr = recall_5 = recall_10 = recall_50 = recall_100 = map_score = 0.0

        return {
            "mrr": mrr,
            "recall@5": recall_5,
            "recall@10": recall_10,
            "recall@50": recall_50,
            "recall@100": recall_100,
            "map": map_score,
        }

    def train(
        self,
        gnn_model: nn.Module,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
        precomputed_embeddings: np.ndarray | None = None,
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
        graph_data = self.build_graph_data(
            all_texts, temporal_dag, text_encoder, precomputed_embeddings
        )

        # Use provided GNN model (already initialized with text encoder)
        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        # Setup wandb
        config = {
            "model_type": type(gnn_model).__name__,
            "text_encoder": self.text_encoder_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "num_negatives": self.num_negatives,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "eval_every_n_epochs": self.eval_every_n_epochs,
            "num_nodes": len(all_texts),
            "num_train_edges": sum(len(v) for k, v in temporal_dag.items()),
            "num_val_edges": sum(len(v) for k, v in val_citation_graph.items()),
        }
        self.setup_wandb(config)

        # Training loop
        print(f"\nStarting training for {self.epochs} epochs...")
        best_loss = float("inf")
        best_val_map = 0.0
        patience_counter = 0
        patience = 10  # Early stopping patience

        for epoch in tqdm(range(self.epochs), desc="Training Progress"):
            train_loss = self.train_epoch(
                model, graph_data, temporal_dag, optimizer, epoch
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            # Evaluate on validation set
            should_eval = (
                self.eval_every_n_epochs is not None
                and (epoch + 1) % self.eval_every_n_epochs == 0
            )

            if should_eval or (epoch + 1) == self.epochs:
                print("  Evaluating on validation set...")
                val_metrics = self.evaluate(
                    model, graph_data, val_citation_graph, paragraph_dates
                )
                print(f"  Val MAP: {val_metrics['map']:.4f}")
                print(f"  Val MRR: {val_metrics['mrr']:.4f}")
                print(f"  Val Recall@10: {val_metrics['recall@10']:.4f}")

                if self.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                            "val_map": val_metrics["map"],
                            "val_mrr": val_metrics["mrr"],
                            "val_recall@5": val_metrics["recall@5"],
                            "val_recall@10": val_metrics["recall@10"],
                            "val_recall@50": val_metrics["recall@50"],
                            "val_recall@100": val_metrics["recall@100"],
                        }
                    )
                else:
                    if self.use_wandb:
                        wandb.log(
                            {
                                "epoch": epoch + 1,
                                "train_loss": train_loss,
                            }
                        )

                # Save best model based on validation MAP
                if val_metrics["map"] > best_val_map:
                    best_val_map = val_metrics["map"]
                    patience_counter = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_val_map": best_val_map,
                            "val_metrics": val_metrics,
                            "config": config,
                        },
                        f"{self.output_path}/best_model.pt",
                    )
                    print(f"  ✓ New best model saved (MAP: {best_val_map:.4f})")
                else:
                    patience_counter += 1
                    print(f"  No improvement ({patience_counter}/{patience})")

                # Early stopping
                if patience_counter >= patience:
                    print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                    break
            else:
                if self.use_wandb:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "train_loss": train_loss,
                        }
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

        print(f"\nTraining complete! Best validation MAP: {best_val_map:.4f}")
        print(f"Model saved to {self.output_path}")

        self.cleanup_wandb()

        # Return the text encoder used for training
        return model

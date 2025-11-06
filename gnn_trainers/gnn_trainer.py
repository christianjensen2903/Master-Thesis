import os
import pickle
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
from validation_utils import split_data_by_date


def info_nce_loss(anchor, positive, negatives, temperature=0.07):
    """
    anchor: [batch_size, dim]
    positive: [batch_size, dim]
    negatives: [batch_size, num_negatives, dim]
    """
    # Normalize embeddings
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    negatives = F.normalize(negatives, dim=-1)

    # Positive similarity
    pos_sim = torch.sum(anchor * positive, dim=-1) / temperature  # [batch_size]

    # Negative similarities
    neg_sim = (
        torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1) / temperature
    )  # [batch_size, num_negatives]

    # InfoNCE loss
    logits = torch.cat(
        [pos_sim.unsqueeze(1), neg_sim], dim=1
    )  # [batch_size, 1 + num_negatives]
    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss


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
        embeddings_cache_dir: str | None = None,
        num_negatives: int = 5,
        num_hops: int = -1,
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
        self.embeddings_cache_dir = embeddings_cache_dir
        self.num_negatives = num_negatives
        self.num_hops = num_hops

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        print(f"Using device: {self.device}")

        # Create cache directory if specified
        if self.embeddings_cache_dir:
            os.makedirs(self.embeddings_cache_dir, exist_ok=True)

    def load_and_split_data(
        self, paragraph_file: str, cutoff_year: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and split into train/validation sets."""
        df = pd.read_csv(paragraph_file)
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])

        df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])

        df["FROM_ID"] = df["CELEX_FROM"] + "::" + df["NUMBER_FROM"].astype(str)
        df["TO_ID"] = df["CELEX_TO"] + "::" + df["NUMBER_TO"].astype(str)

        return split_data_by_date(df, cutoff_year)

    def build_par_id_to_idx(
        self, df: pd.DataFrame
    ) -> tuple[dict[str, int], np.ndarray]:
        """Build paragraph ID to index mapping"""
        df["FROM_ID"] = df["CELEX_FROM"] + "::" + df["NUMBER_FROM"].astype(str)
        df["TO_ID"] = df["CELEX_TO"] + "::" + df["NUMBER_TO"].astype(str)

        # Get all unique paragraph IDs and their texts
        ids = pd.DataFrame(
            {
                "PAR_ID": pd.concat([df["FROM_ID"], df["TO_ID"]]),
                "TEXT": pd.concat([df["TEXT_FROM"], df["TEXT_TO"]]),
            }
        ).drop_duplicates(subset=["PAR_ID"])

        # Create ID to integer mapping
        par_id_to_idx = {par_id: i for i, par_id in enumerate(ids["PAR_ID"])}
        all_texts = ids["TEXT"].fillna("").values

        return par_id_to_idx, all_texts

    def build_citation_graph_from_df(
        self, df: pd.DataFrame, par_id_to_idx: dict[str, int]
    ) -> dict[int, list[int]]:
        """Build citation graph from dataframe using existing par_id_to_idx mapping."""
        df["src_idx"] = df["FROM_ID"].map(par_id_to_idx)
        df["tgt_idx"] = df["TO_ID"].map(par_id_to_idx)
        citation_graph = df.groupby("src_idx")["tgt_idx"].apply(list).to_dict()

        return citation_graph

    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name to be a valid filename by removing path separators."""
        # Extract basename and replace any remaining path separators with underscores
        basename = os.path.basename(model_name)
        return basename.replace("/", "_").replace("\\", "_")

    def _load_cached_embeddings(self, key: str) -> np.ndarray | None:
        """Load embeddings from cache if they exist."""
        if not self.embeddings_cache_dir:
            return None

        cache_path = os.path.join(self.embeddings_cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cached_embeddings(self, embeddings: np.ndarray, key: str) -> None:
        """Save embeddings to cache."""
        if not self.embeddings_cache_dir:
            return

        cache_path = os.path.join(self.embeddings_cache_dir, f"{key}.pkl")
        print(f"Saving embeddings to cache: {cache_path}")
        with open(cache_path, "wb") as f:
            pickle.dump(embeddings, f)

    def _encode_texts(
        self, texts: np.ndarray, text_encoder: SentenceTransformer
    ) -> np.ndarray:
        return text_encoder.encode(
            texts.tolist(),
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

    def build_graph_data(
        self,
        texts: np.ndarray,
        citation_graph: dict[int, list[int]],
        text_embeddings: np.ndarray,
    ) -> Data:

        x = torch.tensor(text_embeddings, dtype=torch.float32)

        # Build edge index from citation graph
        edge_list = []
        for src, targets in citation_graph.items():
            for tgt in targets:
                if src >= len(texts) or tgt >= len(texts):
                    continue

                edge_list.append([tgt, src])

        if edge_list:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        print(f"Graph: {len(texts)} nodes, {edge_index.shape[1]} edges")

        return Data(x=x, edge_index=edge_index, num_nodes=len(texts))

    def train_epoch(
        self, model: nn.Module, loader: NeighborLoader, optimizer: torch.optim.Optimizer
    ) -> float:
        model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(loader, desc="Training batches", leave=False):
            # Get batch data
            x = batch.x
            edge_index = batch.edge_index
            batch_size = batch.batch_size

            # Get embeddings for nodes in this batch
            embeddings = model(x, edge_index)

            anchor_emb = x[:batch_size]

            # Get positive samples from edges
            src, dst = edge_index

            # Find edges where source is in the input batch
            input_mask = src < batch_size
            if input_mask.sum() == 0:
                # No edges for this batch, skip
                continue

            batch_src = src[input_mask]
            batch_dst = dst[input_mask]

            # Sort edges by source for efficient grouping
            sorted_idx = torch.argsort(batch_src)
            src_sorted = batch_src[sorted_idx]
            dst_sorted = batch_dst[sorted_idx]

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
            positive_indices[unique_src] = dst_sorted[selected_edges]

            positive_emb = embeddings[positive_indices]

            # Sample negative pairs (random nodes from the batch)
            num_nodes_in_batch = embeddings.shape[0]
            negative_indices = torch.randint(
                0,
                num_nodes_in_batch,
                (batch_size, self.num_negatives),
                device=self.device,
            )
            negative_emb = embeddings[negative_indices]

            # Compute loss
            loss = info_nce_loss(anchor_emb, positive_emb, negative_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def train(
        self,
        gnn_model: nn.Module,
        paragraph_file: str,
        cutoff_year: int,
    ) -> torch.nn.Module:
        os.makedirs(self.output_path, exist_ok=True)

        train_df, val_df = self.load_and_split_data(paragraph_file, cutoff_year)

        # Build paragraph mapping only from training data
        train_id_to_idx, train_texts = self.build_par_id_to_idx(train_df)
        # _, val_texts = self.build_par_id_to_idx(val_df)
        qrels, docs, val_texts = self.create_validation_data(val_df, train_df)

        train_citation_graph = self.build_citation_graph_from_df(
            train_df, train_id_to_idx
        )

        print(f"\nLoading text encoder: {self.text_encoder_name}")
        text_encoder = SentenceTransformer(self.text_encoder_name)

        print("Loading embeddings...")
        model_name = self._sanitize_model_name(self.text_encoder_name)
        train_embeddings_key = f"{model_name}_train"
        train_embeddings = self._load_cached_embeddings(train_embeddings_key)
        if train_embeddings is None:
            train_embeddings = self._encode_texts(train_texts, text_encoder)
            self._save_cached_embeddings(train_embeddings, train_embeddings_key)

        val_embeddings_key = f"{model_name}_val"
        val_embeddings = self._load_cached_embeddings(val_embeddings_key)
        if val_embeddings is None:
            val_embeddings = self._encode_texts(val_texts, text_encoder)
            self._save_cached_embeddings(val_embeddings, val_embeddings_key)

        graph_data = self.build_graph_data(
            train_texts, train_citation_graph, train_embeddings
        )

        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        print(f"\nStarting training for {self.epochs} epochs...")
        best_map = 0.0

        graph_data = graph_data.to(self.device)

        loader = NeighborLoader(
            graph_data,
            num_neighbors=[-1] * self.num_hops,
            batch_size=self.batch_size,
            input_nodes=None,  # Sample all nodes
            shuffle=True,
        )

        for epoch in range(self.epochs):
            train_loss = self.train_epoch(model, loader, optimizer)

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            # Evaluate on validation set
            should_eval = (
                self.eval_every_n_epochs is None
                or (epoch + 1) % self.eval_every_n_epochs == 0
                or epoch == self.epochs - 1
            )

            if should_eval:
                val_metrics = self.evaluate_ir_metrics(
                    model,
                    graph_data,
                    val_embeddings,
                    qrels,
                    train_id_to_idx,
                )

                print(f"  Validation Metrics:")
                print(f"    MAP@1000: {val_metrics['map@1000']:.4f}")
                for k in [5, 10, 100]:
                    print(f"R@{k}: {val_metrics[f'recall@{k}']:.4f}")

                # Save best model based on MAP@1000
                if val_metrics["map@1000"] > best_map:
                    best_map = val_metrics["map@1000"]
                    torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                    print(f"  ✓ New best model saved (MAP@1000: {best_map:.4f})")

            scheduler.step()

        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")

        print(f"Model saved to {self.output_path}")
        return model

    def create_validation_data(
        self, val_df: pd.DataFrame, train_df: pd.DataFrame
    ) -> tuple[dict[str, set[str]], dict[str, str], np.ndarray]:
        """Create validation data structures for IR evaluation."""
        docs: dict[str, str] = {}
        qrels: dict[str, set[str]] = {}
        queries: list[str] = []

        # Build document corpus from training data
        for _, row in train_df.iterrows():
            text_from = str(row["TEXT_FROM"])
            text_to = str(row["TEXT_TO"])
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])

            docs[from_id] = text_from
            docs[to_id] = text_to

        # Build queries and relevant docs from validation data
        for _, row in val_df.iterrows():
            text_from = str(row["TEXT_FROM"])
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])

            # Only include if target document exists in training corpus
            if to_id not in docs:
                continue

            if from_id not in qrels:
                queries.append(text_from)
                qrels[from_id] = set()

            qrels[from_id].add(to_id)

        return qrels, docs, np.array(queries, dtype=object)

    def evaluate_ir_metrics(
        self,
        model: nn.Module,
        graph_data: Data,
        query_embeddings: np.ndarray,  # Raw query embeddings
        qrels: dict[str, set[str]],
        train_id_to_idx: dict[str, int],
        k_values: list[int] = [5, 10, 50, 100],
    ) -> dict[str, float]:
        """Evaluate IR metrics using projected queries and GNN documents."""
        model.eval()

        with torch.no_grad():
            # Get GNN embeddings for documents
            x = graph_data.x.to(self.device)
            edge_index = graph_data.edge_index.to(self.device)
            doc_embeddings = model(x, edge_index)
            doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)
            doc_embeddings = doc_embeddings.cpu().numpy()

            # Project queries through query encoder
            query_tensor = torch.tensor(
                query_embeddings, dtype=torch.float32, device=self.device
            )
            projected_queries = model.encode_query(query_tensor)
            projected_queries = F.normalize(projected_queries, p=2, dim=1)
            projected_queries = projected_queries.cpu().numpy()

        # Compute similarities with projected queries
        similarities = projected_queries @ doc_embeddings.T

        # Compute metrics
        metrics: dict[str, float] = {}
        map_scores: list[float] = []
        precision_scores: dict[int, list[float]] = {k: [] for k in k_values}
        recall_scores: dict[int, list[float]] = {k: [] for k in k_values}

        train_idx_to_id = {v: k for k, v in train_id_to_idx.items()}

        for i, rel_docs in enumerate(qrels.values()):
            # Get top-k documents
            top_k = max(k_values)
            top_indices = np.argsort(-similarities[i])[:top_k]
            top_doc_ids = [train_idx_to_id[idx] for idx in top_indices]

            # Compute MAP@1000 (using all top-k)
            map_score = 0.0
            num_rel = 0
            for rank, doc_id in enumerate(top_doc_ids, 1):
                if doc_id in rel_docs:
                    num_rel += 1
                    map_score += num_rel / rank
            if len(rel_docs) > 0:
                map_score /= len(rel_docs)
            map_scores.append(map_score)

            # Compute Precision@k and Recall@k
            for k in k_values:
                top_k_docs = set(top_doc_ids[:k])
                num_relevant_retrieved = len(top_k_docs.intersection(rel_docs))

                precision = num_relevant_retrieved / k if k > 0 else 0.0
                recall = (
                    num_relevant_retrieved / len(rel_docs) if len(rel_docs) > 0 else 0.0
                )

                precision_scores[k].append(precision)
                recall_scores[k].append(recall)

        # Average metrics
        if map_scores:
            metrics["map@1000"] = float(np.mean(map_scores))
        else:
            metrics["map@1000"] = 0.0
        for k in k_values:
            if precision_scores[k]:
                metrics[f"precision@{k}"] = float(np.mean(precision_scores[k]))
            else:
                metrics[f"precision@{k}"] = 0.0
            if recall_scores[k]:
                metrics[f"recall@{k}"] = float(np.mean(recall_scores[k]))
            else:
                metrics[f"recall@{k}"] = 0.0

        model.train()
        return metrics

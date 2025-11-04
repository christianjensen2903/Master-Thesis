import os
import random
import pickle
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
        use_temporal_validation: bool = True,
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
        self.use_temporal_validation = use_temporal_validation

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        return split_data_by_date(df, cutoff_year)

    def setup_wandb(self) -> None:
        """Initialize wandb with the given config."""

        config = {
            "text_encoder": self.text_encoder_name,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "temperature": self.temperature,
            "num_negatives": self.num_negatives,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "eval_every_n_epochs": self.eval_every_n_epochs,
        }
        if self.use_wandb:
            wandb.init(project=self.project_name, config=config)

    def cleanup_wandb(self) -> None:
        """Save model to wandb and finish the run."""
        if self.use_wandb:
            wandb.save(f"{self.output_path}/*")
            wandb.finish()

    def build_citation_graph_from_df(
        self, df: pd.DataFrame
    ) -> tuple[dict[int, list[int]], dict[str, int], np.ndarray]:
        """Build citation graph using (CELEX, NUMBER) as unique node IDs."""

        df["FROM_ID"] = df["CELEX_FROM"] + "::" + df["NUMBER_FROM"].astype(str)
        df["TO_ID"] = df["CELEX_TO"] + "::" + df["NUMBER_TO"].astype(str)

        combined_ids = pd.DataFrame(
            {
                "PAR_ID": pd.concat([df["FROM_ID"], df["TO_ID"]]),
                "TEXT": pd.concat([df["TEXT_FROM"], df["TEXT_TO"]]),
            }
        ).drop_duplicates(subset=["PAR_ID"])

        # Create ID to integer mapping
        par_id_to_idx = {par_id: i for i, par_id in enumerate(combined_ids["PAR_ID"])}

        all_texts = combined_ids["TEXT"].fillna("").values

        # Build citation graph
        df["src_idx"] = df["FROM_ID"].map(par_id_to_idx)
        df["tgt_idx"] = df["TO_ID"].map(par_id_to_idx)
        citation_graph = df.groupby("src_idx")["tgt_idx"].apply(list).to_dict()

        return citation_graph, par_id_to_idx, all_texts

    def _sanitize_model_name(self, model_name: str) -> str:
        """Sanitize model name to be a valid filename by removing path separators."""
        # Extract basename and replace any remaining path separators with underscores
        basename = os.path.basename(model_name)
        return basename.replace("/", "_").replace("\\", "_")

    def _load_cached_embeddings(self, model_name: str) -> np.ndarray | None:
        """Load embeddings from cache if they exist."""
        if not self.embeddings_cache_dir:
            return None

        sanitized_name = self._sanitize_model_name(model_name)
        cache_path = os.path.join(self.embeddings_cache_dir, f"{sanitized_name}.pkl")
        if os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            with open(cache_path, "rb") as f:
                return pickle.load(f)
        return None

    def _save_cached_embeddings(self, embeddings: np.ndarray, model_name: str) -> None:
        """Save embeddings to cache."""
        if not self.embeddings_cache_dir:
            return

        sanitized_name = self._sanitize_model_name(model_name)
        cache_path = os.path.join(self.embeddings_cache_dir, f"{sanitized_name}.pkl")
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

    def train(
        self,
        gnn_model: nn.Module,
        paragraph_file: str,
        cutoff_year: int,
    ) -> torch.nn.Module:
        os.makedirs(self.output_path, exist_ok=True)

        # Load and split data
        train_df, _ = self.load_and_split_data(paragraph_file, cutoff_year)

        train_citation_graph, par_id_to_idx, all_texts = (
            self.build_citation_graph_from_df(train_df)
        )

        print(f"\nLoading text encoder: {self.text_encoder_name}")
        text_encoder = SentenceTransformer(self.text_encoder_name)

        print("Loading embeddings...")
        text_embeddings = self._load_cached_embeddings(self.text_encoder_name)
        if text_embeddings is None:
            text_embeddings = self._encode_texts(all_texts, text_encoder)
            self._save_cached_embeddings(text_embeddings, self.text_encoder_name)

        graph_data = self.build_graph_data(
            all_texts, train_citation_graph, text_embeddings
        )

        print("\nInitializing GNN model...")
        model = gnn_model.to(self.device)

        # Initialize optimizer and scheduler
        optimizer = AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        self.setup_wandb()

        print(f"\nStarting training for {self.epochs} epochs...")
        best_loss = float("inf")

        for epoch in tqdm(
            range(self.epochs), desc="Training Progress", position=0, ncols=80
        ):
            train_loss = self.train_epoch(
                model, graph_data, train_citation_graph, optimizer
            )

            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_loss:.4f}")

            if train_loss < best_loss:
                best_loss = train_loss
                torch.save(model.state_dict(), f"{self.output_path}/best_model.pt")
                print(f"  ✓ New best model saved (Loss: {best_loss:.4f})")

            if self.use_wandb:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "train_loss": train_loss,
                    }
                )

            scheduler.step()

        torch.save(model.state_dict(), f"{self.output_path}/final_model.pt")

        print(f"Model saved to {self.output_path}")

        self.cleanup_wandb()
        return model

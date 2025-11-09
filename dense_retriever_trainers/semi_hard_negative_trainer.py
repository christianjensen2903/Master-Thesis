import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from datasets import Dataset  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from .base_trainer import BaseDenseRetrieverTrainer


class SemiHardNegativeDenseRetrieverTrainer(BaseDenseRetrieverTrainer):
    """Trainer that uses semi-hard negative sampling (FaceNet style).

    Semi-hard negatives are those where:
    distance(anchor, negative) > distance(anchor, positive)
    but distance(anchor, negative) < distance(anchor, hardest_negative)
    """

    def __init__(
        self,
        model_name: str,
        training_args: SentenceTransformerTrainingArguments,
        validation_split: float = 0.1,
        use_wandb: bool = True,
        max_seq_length: int | None = None,
        loss_scale: float = 20.0,
        num_semi_hard_negatives: int = 1,
        margin: float = 0.2,
    ):
        super().__init__(
            model_name, training_args, validation_split, use_wandb, max_seq_length
        )
        self.loss_scale = loss_scale
        self.num_semi_hard_negatives = num_semi_hard_negatives
        self.margin = margin

    def _compute_distances(
        self, anchor_emb: np.ndarray, candidate_embs: np.ndarray
    ) -> np.ndarray:
        """Compute cosine distances between anchor and candidates."""
        # Normalize embeddings
        anchor_norm = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-8)
        candidate_norms = candidate_embs / (
            np.linalg.norm(candidate_embs, axis=1, keepdims=True) + 1e-8
        )

        # Cosine distance = 1 - cosine similarity
        similarities = np.dot(candidate_norms, anchor_norm)
        distances = 1 - similarities
        return distances

    def _select_semi_hard_negatives(
        self,
        distances: np.ndarray,
        positive_idx: int,
        positive_distance: float,
    ) -> list[int]:
        """Select semi-hard negatives based on FaceNet criteria."""
        # Semi-hard: distance > positive_distance but < (positive_distance + margin)
        # Also exclude the hardest negative (max distance) to avoid too-hard examples
        max_distance = np.max(distances)

        # Find candidates that are semi-hard
        semi_hard_mask = (distances > positive_distance) & (
            distances < min(positive_distance + self.margin, max_distance)
        )

        # Exclude the positive itself
        semi_hard_mask[positive_idx] = False

        semi_hard_indices = np.where(semi_hard_mask)[0]

        if len(semi_hard_indices) == 0:
            # Fallback: select negatives that are harder than positive but not hardest
            hard_mask = (distances > positive_distance) & (distances < max_distance)
            hard_mask[positive_idx] = False
            hard_indices = np.where(hard_mask)[0]

            if len(hard_indices) == 0:
                return []

            # Select from hard negatives
            num_samples = min(self.num_semi_hard_negatives, len(hard_indices))
            selected = np.random.choice(hard_indices, size=num_samples, replace=False)
            return selected.tolist()

        # Select random semi-hard negatives
        num_samples = min(self.num_semi_hard_negatives, len(semi_hard_indices))
        selected = np.random.choice(semi_hard_indices, size=num_samples, replace=False)
        return selected.tolist()

    def get_training_data(
        self, paragraph_file: str, cutoff_year: int
    ) -> tuple[Dataset, pd.DataFrame, pd.DataFrame]:
        """Get training data with semi-hard negatives."""
        train_df, val_df = self.load_and_split_data(paragraph_file, cutoff_year)

        # Get unique candidate texts (all TO texts)
        candidate_texts = train_df["TEXT_TO"].astype(str).unique().tolist()

        # Initialize model for computing embeddings
        print("Computing embeddings for semi-hard negative mining...")
        temp_model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            temp_model.max_seq_length = self.max_seq_length

        # Compute embeddings for all candidates (with passage prefix)
        candidate_texts_prefixed = [f"passage: {text}" for text in candidate_texts]
        candidate_embs = temp_model.encode(
            candidate_texts_prefixed,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        train_data = []
        for _, row in tqdm(
            train_df.iterrows(),
            total=len(train_df),
            desc="Creating training dataset with semi-hard negatives",
        ):
            text_from = str(row["TEXT_FROM"])
            text_to = str(row["TEXT_TO"])

            # Compute anchor embedding (with query prefix)
            anchor_emb = temp_model.encode(
                [f"query: {text_from}"], convert_to_numpy=True
            )[0]

            # Find positive index
            positive_idx = None
            for i, candidate_text in enumerate(candidate_texts):
                if candidate_text == text_to:
                    positive_idx = i
                    break

            if positive_idx is None:
                # Fallback: use positive pair only
                train_data.append(
                    {
                        "sentence1": f"query: {text_from}",
                        "sentence2": f"passage: {text_to}",
                    }
                )
                continue

            # Compute distances to all candidates
            distances = self._compute_distances(anchor_emb, candidate_embs)
            positive_distance = distances[positive_idx]

            # Select semi-hard negatives
            semi_hard_indices = self._select_semi_hard_negatives(
                distances, positive_idx, positive_distance
            )

            # Add positive pair
            train_data.append(
                {"sentence1": f"query: {text_from}", "sentence2": f"passage: {text_to}"}
            )

            # Add semi-hard negative pairs
            for neg_idx in semi_hard_indices:
                neg_text = candidate_texts[neg_idx]
                train_data.append(
                    {
                        "sentence1": f"query: {text_from}",
                        "sentence2": f"passage: {neg_text}",
                    }
                )

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, val_df, train_df

    def train(self, paragraph_file: str, cutoff_year: int) -> SentenceTransformer:
        """Train with semi-hard negatives (FaceNet style)."""
        train_dataset, val_df, train_df = self.get_training_data(
            paragraph_file, cutoff_year
        )

        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length

        train_loss = losses.MultipleNegativesRankingLoss(
            model=model, scale=self.loss_scale
        )
        evaluator = self.create_ir_evaluator(val_df, train_df)

        print(
            f"\nTraining {self.model_name} with Semi-Hard Negatives (FaceNet style, margin={self.margin})..."
        )
        print(f"Total training examples: {len(train_dataset)}")

        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
            args=self.training_args,
        )

        trainer.train()
        output_dir = self.training_args.output_dir
        if output_dir is None:
            raise ValueError("output_dir must be set in training_args")
        model.save(output_dir)

        print(f"Training finished. Model saved to {output_dir}")
        return model

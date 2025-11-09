import numpy as np
from tqdm import tqdm  # type: ignore
from datasets import Dataset  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from dense_retriever_trainers.base_trainer import BaseDenseRetrieverTrainer


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
        use_wandb: bool = True,
        max_seq_length: int | None = None,
        loss_scale: float = 20.0,
        num_semi_hard_negatives: int = 1,
        margin: float = 0.2,
    ):
        super().__init__(model_name, training_args, use_wandb, max_seq_length)
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
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> tuple[Dataset, dict[str, str], dict[str, str], dict[str, set[str]]]:
        """Get training data with semi-hard negatives."""
        train_pairs, corpus, queries, relevant_docs = self.load_and_split_data(
            judgments_path, queries_path, qrel_path, cutoff_year
        )

        # Get candidate texts from corpus
        candidate_texts = list(corpus.values())
        candidate_ids = list(corpus.keys())

        # Initialize model for computing embeddings
        print("Computing embeddings for semi-hard negative mining...")
        temp_model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            temp_model.max_seq_length = self.max_seq_length

        # Compute embeddings for all candidates
        candidate_embs = temp_model.encode(
            candidate_texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
        )

        train_data = []
        for query_id, query_text, doc_id, doc_text in tqdm(
            train_pairs,
            desc="Creating training dataset with semi-hard negatives",
        ):
            # Compute anchor embedding
            anchor_emb = temp_model.encode([query_text], convert_to_numpy=True)[0]

            # Find positive index
            positive_idx = None
            for i, cand_id in enumerate(candidate_ids):
                if cand_id == doc_id:
                    positive_idx = i
                    break

            if positive_idx is None:
                # Fallback: use positive pair only
                train_data.append(
                    {
                        "sentence1": query_text,
                        "sentence2": doc_text,
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
            train_data.append({"sentence1": query_text, "sentence2": doc_text})

            # Add semi-hard negative pairs (ensure positive is not included)
            for neg_idx in semi_hard_indices:
                if neg_idx == positive_idx:
                    continue
                neg_text = candidate_texts[neg_idx]
                train_data.append(
                    {
                        "sentence1": query_text,
                        "sentence2": neg_text,
                    }
                )

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, corpus, queries, relevant_docs

    def train(
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> SentenceTransformer:
        """Train with semi-hard negatives (FaceNet style)."""
        train_dataset, corpus, queries, relevant_docs = self.get_training_data(
            judgments_path, queries_path, qrel_path, cutoff_year
        )

        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length

        train_loss = losses.MultipleNegativesRankingLoss(
            model=model, scale=self.loss_scale
        )
        evaluator = self.create_ir_evaluator(corpus, queries, relevant_docs)

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

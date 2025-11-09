import random
import numpy as np
from tqdm import tqdm  # type: ignore
from datasets import Dataset  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    losses,
)
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

from dense_retriever_trainers.base_trainer import BaseDenseRetrieverTrainer


class HardNegativeDenseRetrieverTrainer(BaseDenseRetrieverTrainer):
    """Trainer that uses TF-IDF to select hard negatives from a rank range."""

    def __init__(
        self,
        model_name: str,
        training_args: SentenceTransformerTrainingArguments,
        use_wandb: bool = True,
        max_seq_length: int | None = None,
        loss_scale: float = 20.0,
        hard_negative_min_rank: int = 100,
        hard_negative_max_rank: int = 300,
        num_hard_negatives: int = 1,
        tfidf_kwargs: dict | None = None,
    ):
        super().__init__(model_name, training_args, use_wandb, max_seq_length)
        self.loss_scale = loss_scale
        self.hard_negative_min_rank = hard_negative_min_rank
        self.hard_negative_max_rank = hard_negative_max_rank
        self.num_hard_negatives = num_hard_negatives

        if tfidf_kwargs is None:
            tfidf_kwargs = {
                "stop_words": "english",
                "strip_accents": "ascii",
                "norm": "l2",
            }
        self.tfidf_kwargs = tfidf_kwargs

    def _select_hard_negatives(
        self, ranked_indices: np.ndarray, positive_idx: int
    ) -> list[int]:
        """Select random hard negatives from the specified rank range."""
        # ranked_indices contains candidate indices sorted by similarity (highest first)
        # Exclude the positive from ranked list
        valid_ranked = ranked_indices[ranked_indices != positive_idx]

        # Get rank range (0-indexed positions in ranked list)
        min_rank = min(self.hard_negative_min_rank, len(valid_ranked))
        max_rank = min(self.hard_negative_max_rank, len(valid_ranked))

        if min_rank >= max_rank:
            # Fallback: use all available candidates after positive
            candidates = valid_ranked.tolist()
        else:
            candidates = valid_ranked[min_rank:max_rank].tolist()

        if len(candidates) == 0:
            return []

        # Ensure positive is not in candidates (defensive check)
        candidates = [idx for idx in candidates if idx != positive_idx]

        if len(candidates) == 0:
            return []

        # Sample random negatives
        num_samples = min(self.num_hard_negatives, len(candidates))
        return random.sample(candidates, num_samples)

    def get_training_data(
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> tuple[Dataset, dict[str, str], dict[str, str], dict[str, set[str]]]:
        """Get training data with hard negatives."""
        train_pairs, corpus, queries, relevant_docs = self.load_and_split_data(
            judgments_path, queries_path, qrel_path, cutoff_year
        )

        # Get unique candidate texts from corpus
        candidate_texts = list(corpus.values())
        candidate_ids = list(corpus.keys())

        print("Fitting TF-IDF vectorizer on training data...")
        self.tfidf_vectorizer = TfidfVectorizer(**self.tfidf_kwargs)
        candidate_tfidf_matrix = self.tfidf_vectorizer.fit_transform(candidate_texts)

        train_data = []
        for query_id, query_text, doc_id, doc_text in tqdm(
            train_pairs,
            desc="Creating training dataset with hard negatives",
        ):
            # Rank all candidates using TF-IDF
            query_vec = self.tfidf_vectorizer.transform([query_text])
            similarities = candidate_tfidf_matrix.dot(query_vec.T).toarray().ravel()
            ranked_indices = np.argsort(-similarities)

            # Find positive index in candidate list
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

            # Select hard negatives
            hard_negative_indices = self._select_hard_negatives(
                ranked_indices, positive_idx
            )

            # Add positive pair
            train_data.append({"sentence1": query_text, "sentence2": doc_text})

            # Add hard negative pairs (ensure positive is not included)
            for neg_idx in hard_negative_indices:
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
        """Train with hard negatives selected via TF-IDF ranking."""
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
            f"\nTraining {self.model_name} with Hard Negatives (TF-IDF ranks {self.hard_negative_min_rank}-{self.hard_negative_max_rank})..."
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

"""
Random Negative Trainer using random negative sampling.
"""

import pandas as pd  # type: ignore
from .base_triplet_trainer import BaseTripletTrainer
from sentence_transformers import SentenceTransformer


class RandomNegativeTrainer(BaseTripletTrainer):
    """Trainer for triplet models using random negative sampling."""

    def _random_negative_sampler(
        self, anchor: str, positive: str, positives: set[str], context: dict
    ) -> str | None:
        """
        Sample random negatives from the pool.
        Returns a randomly selected text that is not the anchor and not a true positive.
        """
        pool_texts = context["pool_texts"]
        rng = context["rng"]
        max_attempts = context["max_attempts"]

        for _ in range(max_attempts):
            neg = pool_texts[rng.randrange(len(pool_texts))]
            if neg != anchor and neg not in positives:
                return neg

        return None

    def train(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
    ) -> SentenceTransformer:
        """Train a model using random negative sampling."""
        return self.train_triplet_model(
            paragraph_file, cutoff_date, self._random_negative_sampler, "random"
        )

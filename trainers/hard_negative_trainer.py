"""
Hard Negative Trainer using TF-IDF similarity for negative sampling.
"""

import pandas as pd  # type: ignore
from .base_triplet_trainer import BaseTripletTrainer
from sentence_transformers import SentenceTransformer


class HardNegativeTrainer(BaseTripletTrainer):
    """Trainer for triplet models using hard negative sampling."""

    def _hard_negative_sampler(
        self, anchor: str, positive: str, positives: set[str], context: dict
    ) -> str | None:
        """
        Sample hard negatives using TF-IDF similarity.
        Returns the first nearest neighbor that is not the anchor and not a true positive.
        """
        pool_texts = context["pool_texts"]
        text2idx = context["text2idx"]
        get_neighbors = context["get_neighbors"]

        anchor_idx = text2idx.get(anchor)
        if anchor_idx is None:
            return None

        neighbors = get_neighbors(anchor_idx)

        for neg_idx, _dist in neighbors:
            neg_text = pool_texts[neg_idx]
            if neg_text == anchor:
                continue
            if neg_text in positives:
                continue
            return neg_text

        return None

    def train(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
    ) -> SentenceTransformer:
        """Train a model using hard negative sampling."""
        return self.train_triplet_model(
            paragraph_file, cutoff_date, self._hard_negative_sampler, "hard"
        )

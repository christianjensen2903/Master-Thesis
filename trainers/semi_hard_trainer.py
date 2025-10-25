"""
Semi-Hard Trainer using margin-based negative sampling.
"""

import pandas as pd  # type: ignore
from sklearn.metrics.pairwise import cosine_distances  # type: ignore
from .base_triplet_trainer import BaseTripletTrainer
from sentence_transformers import SentenceTransformer


class SemiHardTrainer(BaseTripletTrainer):
    """Trainer for triplet models using semi-hard negative sampling."""

    def _semi_hard_negative_sampler(
        self, anchor: str, positive: str, positives: set[str], context: dict
    ) -> str | None:
        """
        Sample semi-hard negatives that satisfy:
        d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin

        This ensures the negative is harder than the positive but not too hard.
        """
        pool_texts = context["pool_texts"]
        text2idx = context["text2idx"]
        get_neighbors = context["get_neighbors"]
        margin = context.get("margin", 0.2)

        anchor_idx = text2idx.get(anchor)
        positive_idx = text2idx.get(positive)

        if anchor_idx is None or positive_idx is None:
            return None

        # Get pre-computed neighbors with distances
        neighbors = get_neighbors(anchor_idx)

        # Get positive distance from neighbors list (if positive is in neighbors)
        # Otherwise compute it once
        anchor_positive_dist = None
        for idx, dist in neighbors:
            if idx == positive_idx:
                anchor_positive_dist = dist
                break

        # If positive not in neighbors, compute distance once
        if anchor_positive_dist is None:
            from sklearn.metrics.pairwise import cosine_distances

            X = context["X"]
            anchor_positive_dist = cosine_distances(X[anchor_idx], X[positive_idx])[0][
                0
            ]

        # Find semi-hard negatives using pre-computed distances
        semi_hard_candidates = []

        for neg_idx, neg_dist in neighbors:
            neg_text = pool_texts[neg_idx]

            # Skip if it's the anchor itself or a positive
            if neg_text == anchor or neg_text in positives:
                continue

            # Check if it's semi-hard: positive_dist < neg_dist < positive_dist + margin
            if anchor_positive_dist < neg_dist < anchor_positive_dist + margin:
                semi_hard_candidates.append((neg_text, neg_dist))

        if not semi_hard_candidates:
            # If no semi-hard negatives found, try to find any negative that's not too easy
            for neg_idx, neg_dist in neighbors:
                neg_text = pool_texts[neg_idx]

                if neg_text == anchor or neg_text in positives:
                    continue

                # Accept any negative that's at least as hard as the positive
                if neg_dist > anchor_positive_dist:
                    semi_hard_candidates.append((neg_text, neg_dist))

        if not semi_hard_candidates:
            return None

        # Select the semi-hard negative with distance closest to (positive_dist + margin/2)
        # This gives us a "medium-hard" negative
        target_dist = anchor_positive_dist + margin / 2
        best_neg = min(semi_hard_candidates, key=lambda x: abs(x[1] - target_dist))

        return best_neg[0]

    def train(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
    ) -> SentenceTransformer:
        """Train a model using semi-hard negative sampling."""
        return self.train_triplet_model(
            paragraph_file, cutoff_date, self._semi_hard_negative_sampler, "semi_hard"
        )

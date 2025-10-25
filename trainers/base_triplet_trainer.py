import random
from typing import Any, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.losses import TripletDistanceMetric
from sklearn.feature_extraction.text import (
    TfidfVectorizer,  # type: ignore
    HashingVectorizer,  # type: ignore
)
from sklearn.neighbors import NearestNeighbors  # type: ignore
from .base_trainer import BaseTrainer


class BaseTripletTrainer(BaseTrainer):
    """Base class for triplet-based training with common functionality."""

    def __init__(
        self,
        margin: float = 0.2,
        distance_metric: TripletDistanceMetric = TripletDistanceMetric.COSINE,
        n_neighbors: int = 50,
        max_attempts: int = 100,
        # TF-IDF/indexing controls for performance
        tfidf_max_features: int | None = 100_000,
        tfidf_min_df: int | float = 3,
        tfidf_max_df: float = 0.9,
        tfidf_ngram_range: tuple[int, int] = (1, 1),
        tfidf_use_hashing: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize BaseTripletTrainer.

        Args:
            margin: Margin for triplet loss
            distance_metric: Distance metric for triplet loss
            n_neighbors: Number of neighbors for KNN search
            max_attempts: Maximum attempts to find a valid negative
        """
        super().__init__(**kwargs)
        self.margin = margin
        self.distance_metric = distance_metric
        self.n_neighbors = n_neighbors
        self.max_attempts = max_attempts
        # TF-IDF/indexing settings
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_max_df = tfidf_max_df
        self.tfidf_ngram_range = tfidf_ngram_range
        self.tfidf_use_hashing = tfidf_use_hashing

    def create_triplet_loss(self, model: SentenceTransformer) -> losses.TripletLoss:
        """Create triplet loss with configured parameters."""
        return losses.TripletLoss(
            model=model,
            distance_metric=self.distance_metric,
            triplet_margin=self.margin,
        )

    def _unique_texts(self, series: pd.Series) -> list[str]:
        """Preserve order while deduping (pd.unique keeps first occurrence)."""
        return pd.unique(series.fillna("").astype(str))

    def _build_tfidf_index(self, texts: list[str]) -> Tuple[Any, Any, dict[str, int]]:
        """Build a fast, memory-efficient TF-IDF (or Hashing) index over texts.

        Returns (vectorizer_like, X_matrix, text_to_index).
        """
        # Map texts to row indices
        text2idx: dict[str, int] = {t: i for i, t in enumerate(texts)}

        if self.tfidf_use_hashing:
            n_features = self.tfidf_max_features or (1 << 18)
            vec = HashingVectorizer(
                n_features=int(n_features),
                alternate_sign=False,
                norm="l2",
                analyzer="word",
                ngram_range=self.tfidf_ngram_range,
                dtype=np.float32,
            )
            X = vec.transform(texts)
            return vec, X, text2idx

        vec = TfidfVectorizer(
            min_df=self.tfidf_min_df,
            max_df=self.tfidf_max_df,
            ngram_range=self.tfidf_ngram_range,
            max_features=self.tfidf_max_features,
            dtype=np.float32,
        )
        X = vec.fit_transform(texts)
        return vec, X, text2idx

    def _build_knn(self, X, n_neighbors: int = 50):
        """
        Cosine KNN over TF-IDF space. Returns a fitted NearestNeighbors.
        Note: cosine distance = 1 - cosine similarity; nearest = most similar.
        """
        knn = NearestNeighbors(
            n_neighbors=n_neighbors, metric="cosine", algorithm="brute"
        )
        knn.fit(X)
        return knn

    def _get_triplet_data_base(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
        negative_sampler,
        random_state: int = 13,
    ) -> tuple[list[InputExample], pd.DataFrame]:
        """
        Base method for constructing triplets using the provided negative sampling strategy.

        Args:
            paragraph_file: Path to paragraph pairs CSV
            cutoff_date: Date cutoff for training data
            negative_sampler: Function that takes (anchor, positive, positives_set, context) and returns negative
            random_state: Random seed
        """
        rng = random.Random(random_state)

        train_df, val_df = self.load_and_split_data(paragraph_file, cutoff_date)
        df = train_df.copy()

        # Keep only rows with both texts present
        df = df.dropna(subset=["TEXT_FROM", "TEXT_TO"])
        df["TEXT_FROM"] = df["TEXT_FROM"].astype(str)
        df["TEXT_TO"] = df["TEXT_TO"].astype(str)

        # Negatives pool: all unique training paragraphs (both sides)
        pool_texts = self._unique_texts(
            pd.concat([df["TEXT_FROM"], df["TEXT_TO"]], ignore_index=True)
        )
        if len(pool_texts) == 0:
            raise ValueError("No training texts found for negative sampling pool.")

        # Map each anchor text to ALL of its legitimate positives
        positives_by_anchor: dict[str, set[str]] = {}
        for a, p in zip(df["TEXT_FROM"], df["TEXT_TO"]):
            positives_by_anchor.setdefault(a, set()).add(p)

        # Build TF-IDF index over pool
        vec, X, text2idx = self._build_tfidf_index(pool_texts)
        knn = self._build_knn(X, n_neighbors=min(self.n_neighbors, len(pool_texts)))

        # Cache for neighbor queries
        neighbor_cache = {}

        def get_neighbors(anchor_idx: int) -> list[tuple[int, float]]:
            """Get cached neighbors for an anchor."""
            if anchor_idx not in neighbor_cache:
                distances, indices = knn.kneighbors(X[anchor_idx], return_distance=True)
                neighbor_cache[anchor_idx] = list(
                    zip(indices[0].tolist(), distances[0].tolist())
                )
            return neighbor_cache[anchor_idx]

        triplets: list[InputExample] = []

        # Build triplets
        for a, p in tqdm(
            zip(df["TEXT_FROM"], df["TEXT_TO"]),
            total=len(df),
            desc="Creating triplets",
        ):
            pos_set = positives_by_anchor.get(a, set())

            # Create context for negative sampler
            context = {
                "pool_texts": pool_texts,
                "text2idx": text2idx,
                "X": X,
                "get_neighbors": get_neighbors,
                "rng": rng,
                "max_attempts": self.max_attempts,
                "margin": self.margin,
            }

            neg = negative_sampler(a, p, pos_set, context)

            if neg is None:
                # Fallback to random negative
                for _ in range(self.max_attempts):
                    cand = pool_texts[rng.randrange(len(pool_texts))]
                    if cand != a and cand not in pos_set:
                        neg = cand
                        break
                if neg is None:
                    continue

            triplets.append(InputExample(texts=[a, p, neg]))

        if not triplets:
            raise ValueError(
                "Failed to construct any triplets. Check your data coverage and cutoff."
            )

        return triplets, val_df

    def train_triplet_model(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
        negative_sampler,
        sampling_strategy_name: str,
    ) -> SentenceTransformer:
        """Train a triplet model with the provided negative sampling strategy."""
        # Get training data
        train_dataset, val_df = self._get_triplet_data_base(
            paragraph_file, cutoff_date, negative_sampler
        )
        train_dataloader: DataLoader = DataLoader(
            train_dataset, shuffle=True, batch_size=self.batch_size  # type: ignore
        )

        # Setup wandb
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "margin": self.margin,
            "sampling_strategy": sampling_strategy_name,
            "n_neighbors": self.n_neighbors,
            "validation_split": self.validation_split,
            "cutoff_date": str(cutoff_date),
        }
        self.setup_wandb(config)

        # Create model and loss
        model = SentenceTransformer(self.model_name)
        train_loss = self.create_triplet_loss(model)

        # Create evaluator
        evaluator = self.create_ir_evaluator(val_df)

        print(
            f"\nTraining {self.model_name} with TripletLoss + {sampling_strategy_name} sampling..."
        )
        print(f"Total triplets: {len(train_dataset)}")
        print(f"Total validation examples: {len(val_df)}")
        print(f"Total batches: {len(train_dataloader)}")

        # Train the model
        trained_model = self._train_model(
            train_dataloader,
            train_loss,
            evaluator,
            f"Training with {sampling_strategy_name} sampling",
        )

        self.cleanup_wandb()
        return trained_model

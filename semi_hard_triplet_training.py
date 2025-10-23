import random
import pandas as pd
import numpy as np
import wandb
from tqdm import tqdm
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.losses import TripletDistanceMetric
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_distances
from typing import Optional
from validation_utils import split_data_by_date, create_ir_evaluator


def _unique_texts(series: pd.Series) -> list[str]:
    """Preserve order while deduping (pd.unique keeps first occurrence)."""
    return pd.unique(series.fillna("").astype(str))


def build_tfidf_index(
    texts: list[str],
    *,
    min_df: int = 3,
    max_df: float = 0.9,
    ngram_range: tuple = (1, 2),
) -> tuple:
    """
    Fit a TF-IDF vectorizer on `texts` and return (vectorizer, X, index_of_text).
    `texts` should be a list of unique strings.
    """
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X = vec.fit_transform(texts)  # csr_matrix [N, V]
    text2idx = {t: i for i, t in enumerate(texts)}
    return vec, X, text2idx


def build_knn(X, n_neighbors: int = 50):
    """
    Cosine KNN over TF-IDF space. Returns a fitted NearestNeighbors.
    Note: cosine distance = 1 - cosine similarity; nearest = most similar.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    knn.fit(X)
    return knn


def get_semi_hard_triplet_data(
    paragraph_file: str,
    cutoff_date: pd.Timestamp,
    n_neighbors: int = 50,
    random_state: int = 13,
    margin: float = 0.2,
    max_attempts: int = 100,
    validation_split: float = 0.1,
) -> tuple[list[InputExample], pd.DataFrame]:
    """
    Construct (anchor, positive, semi-hard-negative) triplets using semi-hard sampling.

    Semi-hard sampling selects negatives that are:
    - Harder than the positive (d(anchor, negative) < d(anchor, positive) + margin)
    - But not the hardest possible (to avoid overfitting to hardest examples)

    This is inspired by FaceNet's semi-hard triplet mining strategy.
    """
    rng = random.Random(random_state)

    df = pd.read_csv(paragraph_file)
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])

    # Split into train and validation
    train_df, val_df = split_data_by_date(df, cutoff_date, validation_split)
    df = train_df.copy()

    # Keep only rows with both texts present
    df = df.dropna(subset=["TEXT_FROM", "TEXT_TO"])
    df["TEXT_FROM"] = df["TEXT_FROM"].astype(str)
    df["TEXT_TO"] = df["TEXT_TO"].astype(str)

    # Negatives pool: all unique training paragraphs (both sides)
    pool_texts = _unique_texts(
        pd.concat([df["TEXT_FROM"], df["TEXT_TO"]], ignore_index=True)
    )
    if len(pool_texts) == 0:
        raise ValueError("No training texts found for TF-IDF pool.")

    # Map each anchor text to ALL of its legitimate positives
    positives_by_anchor: dict[str, set[str]] = {}
    for a, p in zip(df["TEXT_FROM"], df["TEXT_TO"]):
        positives_by_anchor.setdefault(a, set()).add(p)

    # Build TF-IDF index over pool
    vec, X, text2idx = build_tfidf_index(pool_texts)

    # KNN over TF-IDF vectors
    knn = build_knn(X, n_neighbors=min(n_neighbors, len(pool_texts)))

    triplets: list[InputExample] = []

    # Query KNN for each anchor only once; cache neighbors by anchor index
    neighbor_cache = {}

    def pick_semi_hard_negative(
        anchor: str, positive: str, positives: set[str]
    ) -> Optional[str]:
        """
        Return a semi-hard negative that satisfies:
        d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin

        This ensures the negative is harder than the positive but not too hard.
        """
        anchor_idx = text2idx.get(anchor)
        positive_idx = text2idx.get(positive)

        if anchor_idx is None or positive_idx is None:
            return None

        # Cache neighbors for this anchor
        if anchor_idx not in neighbor_cache:
            distances, indices = knn.kneighbors(X[anchor_idx], return_distance=True)
            neighbor_cache[anchor_idx] = list(
                zip(indices[0].tolist(), distances[0].tolist())
            )

        # Calculate distance from anchor to positive
        anchor_positive_dist = cosine_distances(X[anchor_idx], X[positive_idx])[0][0]

        # Find semi-hard negatives
        semi_hard_candidates = []

        for neg_idx, neg_dist in neighbor_cache[anchor_idx]:
            neg_text = pool_texts[neg_idx]

            # Skip if it's the anchor itself or a positive
            if neg_text == anchor or neg_text in positives:
                continue

            # Check if it's semi-hard: positive_dist < neg_dist < positive_dist + margin
            if anchor_positive_dist < neg_dist < anchor_positive_dist + margin:
                semi_hard_candidates.append((neg_text, neg_dist))

        if not semi_hard_candidates:
            # If no semi-hard negatives found, try to find any negative that's not too easy
            for neg_idx, neg_dist in neighbor_cache[anchor_idx]:
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

    # Build triplets
    for a, p in tqdm(
        zip(df["TEXT_FROM"], df["TEXT_TO"]),
        total=len(df),
        desc="Creating semi-hard triplets",
    ):
        pos_set = positives_by_anchor.get(a, set())
        neg = pick_semi_hard_negative(a, p, pos_set)

        if neg is None:
            # Fallback to a random negative not equal to a or any positive
            for _ in range(max_attempts):
                cand = pool_texts[rng.randrange(len(pool_texts))]
                if cand != a and cand not in pos_set:
                    neg = cand
                    break
            if neg is None:
                # As a last resort, skip this example
                continue

        triplets.append(InputExample(texts=[a, p, neg]))

    if not triplets:
        raise ValueError(
            "Failed to construct any triplets. Check your data coverage and cutoff."
        )

    return triplets, val_df


def train_semi_hard_triplet(
    paragraph_file: str,
    cutoff_date: pd.Timestamp,
    model_name: str = "all-MiniLM-L6-v2",
    output_path: str = "output/simcse_semi_hard_triplet",
    batch_size: int = 16,
    epochs: int = 5,
    warmup_steps: int = 100,
    margin: float = 0.2,
    distance_metric: TripletDistanceMetric = TripletDistanceMetric.COSINE,
    show_progress_bar: bool = True,
    n_neighbors: int = 50,
    max_attempts: int = 100,
    validation_split: float = 0.1,
    use_wandb: bool = True,
    project_name: str = "semi-hard-triplet-model",
) -> SentenceTransformer:
    """
    Train a model using semi-hard triplet sampling.

    Args:
        paragraph_file: Path to the paragraph pairs CSV file
        cutoff_date: Date cutoff for training data
        model_name: Name of the base model to use
        output_path: Path to save the trained model
        batch_size: Training batch size
        epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        margin: Margin for triplet loss and semi-hard selection
        distance_metric: Distance metric for triplet loss
        show_progress_bar: Whether to show progress bar
        n_neighbors: Number of neighbors for KNN search
        max_attempts: Maximum attempts to find a valid negative
    """
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project=project_name,
            config={
                "model_name": model_name,
                "batch_size": batch_size,
                "epochs": epochs,
                "warmup_steps": warmup_steps,
                "margin": margin,
                "n_neighbors": n_neighbors,
                "validation_split": validation_split,
                "cutoff_date": str(cutoff_date),
            },
        )

    train_dataset, val_df = get_semi_hard_triplet_data(
        paragraph_file,
        cutoff_date,
        n_neighbors=n_neighbors,
        margin=margin,
        max_attempts=max_attempts,
        validation_split=validation_split,
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)

    model = SentenceTransformer(model_name)
    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=distance_metric,
        triplet_margin=margin,
    )

    # Create validation evaluator if validation data exists
    evaluator = None
    if len(val_df) > 0:
        evaluator = create_ir_evaluator(val_df, model_name)

    print(f"\nTraining {model_name} with TripletLoss + Semi-Hard Sampling...")
    print(f"Total triplets: {len(train_dataset)}")
    print(f"Total validation examples: {len(val_df)}")
    print(f"Total batches: {len(train_dataloader)}")

    # Training with validation
    if evaluator is not None:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            scheduler="WarmupLinear",
            show_progress_bar=show_progress_bar,
            evaluator=evaluator,
            evaluation_steps=500,  # Evaluate every 500 steps
        )
    else:
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            scheduler="WarmupLinear",
            show_progress_bar=show_progress_bar,
        )

    # Save model to wandb
    if use_wandb:
        wandb.save(f"{output_path}/*")
        wandb.finish()

    print(f"Training finished. Model saved to {output_path}")
    return model


if __name__ == "__main__":
    train_semi_hard_triplet(
        paragraph_file="data/par-to-par.csv",
        cutoff_date=pd.Timestamp("2018-01-01"),
        model_name="all-mpnet-base-v2",
        output_path="artifacts/simcse_semi_hard_triplet",
        margin=0.2,
        n_neighbors=100,
        validation_split=0.1,
        use_wandb=True,
        project_name="semi-hard-triplet-model",
    )

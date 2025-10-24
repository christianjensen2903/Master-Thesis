import random
import pandas as pd  # type: ignore
import wandb
from tqdm import tqdm  # type: ignore
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.losses import TripletDistanceMetric

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.neighbors import NearestNeighbors  # type: ignore
from validation_utils import split_data_by_date, create_ir_evaluator


def _unique_texts(series):
    # Preserve order while deduping (pd.unique keeps first occurrence)
    return pd.unique(series.fillna("").astype(str))


def build_tfidf_index(texts, *, min_df=3, max_df=0.9, ngram_range=(1, 2)):
    """
    Fit a TF-IDF vectorizer on `texts` and return (vectorizer, X, index_of_text).
    `texts` should be a list of unique strings.
    """
    vec = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X = vec.fit_transform(texts)  # csr_matrix [N, V]
    text2idx = {t: i for i, t in enumerate(texts)}
    return vec, X, text2idx


def build_knn(X, n_neighbors=50):
    """
    Cosine KNN over TF-IDF space. Returns a fitted NearestNeighbors.
    Note: cosine distance = 1 - cosine similarity; nearest = most similar.
    """
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute")
    knn.fit(X)
    return knn


def get_triplet_data_tfidf(
    paragraph_file: str,
    cutoff_date: pd.Timestamp,
    n_neighbors: int = 50,
    random_state: int = 13,
    validation_split: float = 0.1,
) -> tuple[list[InputExample], pd.DataFrame]:
    """
    Construct (anchor, positive, hard-negative) triplets using TF-IDF hard negatives.
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

    # Map each anchor text to ALL of its legitimate positives (some anchors can have many).
    # If your task is directional, this is fine. If it's symmetric, also add reverse edges.
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

    def pick_hard_negative(anchor: str, positives: set[str]) -> str | None:
        """Return first nearest neighbor that is not the anchor and not a true positive."""
        idx = text2idx.get(anchor)
        if idx is None:
            return None  # anchor not in pool (shouldn't happen)

        if idx not in neighbor_cache:
            # Query neighbors for this anchor
            distances, indices = knn.kneighbors(X[idx], return_distance=True)
            neighbor_cache[idx] = list(zip(indices[0].tolist(), distances[0].tolist()))

        for j, _dist in neighbor_cache[idx]:
            cand = pool_texts[j]
            if cand == anchor:
                continue
            if cand in positives:
                continue
            return cand
        return None  # all neighbors are positives or self

    # Build triplets
    for a, p in tqdm(
        zip(df["TEXT_FROM"], df["TEXT_TO"]),
        total=len(df),
        desc="Creating TF-IDF hard-negative triplets",
    ):
        pos_set = positives_by_anchor.get(a, set())
        neg = pick_hard_negative(a, pos_set)

        if neg is None:
            # Fallback to a random negative not equal to a or any positive
            for _ in range(10):
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


def train_triplet_tfidf(
    paragraph_file: str,
    cutoff_date: pd.Timestamp,
    model_name: str = "all-MiniLM-L6-v2",
    output_path: str = "output/simcse_triplet_tfidf",
    batch_size: int = 16,
    epochs: int = 5,
    warmup_steps: int = 100,
    margin: float = 0.2,
    distance_metric: TripletDistanceMetric = TripletDistanceMetric.COSINE,
    show_progress_bar: bool = True,
    validation_split: float = 0.1,
    use_wandb: bool = True,
    project_name: str = "triplet-tfidf-model",
):
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
                "validation_split": validation_split,
                "cutoff_date": str(cutoff_date),
            },
        )

    train_dataset, val_df = get_triplet_data_tfidf(
        paragraph_file, cutoff_date, validation_split=validation_split
    )
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)  # type: ignore

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

    print(f"\nTraining {model_name} with TripletLoss + TF-IDF hard negatives...")
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
    train_triplet_tfidf(
        paragraph_file="data/par-to-par.csv",
        cutoff_date=pd.Timestamp("2018-01-01"),
        model_name="all-mpnet-base-v2",
        output_path="artifacts/simcse_triplet_tfidf",
        validation_split=0.1,
        use_wandb=True,
        project_name="triplet-tfidf-model",
    )

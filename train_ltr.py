import argparse
import os
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray
import lightgbm as lgb  # type: ignore
from tqdm import tqdm

from data_loader import (
    load_citation_data,
    split_train_test,
    build_paragraph_index,
    build_citation_graph,
)
from retrievers.base_retriever import BaseRetriever
from ltr_feature_extractor import LTRFeatureExtractor


def generate_training_data(
    retriever: BaseRetriever,
    embeddings: NDArray,
    feature_extractor: LTRFeatureExtractor,
    paragraph_set: NDArray[np.object_],
    paragraph_celex: NDArray[np.object_],
    paragraph_number: NDArray[np.object_],
    paragraph_dates: NDArray,
    cited_by_pid: dict[int, list[int]],
    num_negatives: int = 5,
    max_queries: int | None = None,
) -> tuple[list[list[float]], list[int], list[int]]:
    """
    Generate training data for LTR model.

    For each query paragraph in the training set:
    - Retrieve top candidates using base retriever
    - Label positives (actual citations) as 1, negatives as 0
    - Extract features for all query-candidate pairs

    Returns:
        features: List of feature vectors
        labels: List of labels (1=relevant, 0=not relevant)
        groups: List of group sizes (number of candidates per query)
    """
    # Pre-sort by date for temporal filtering
    sort_idx = np.argsort(paragraph_dates)
    sorted_dates = paragraph_dates[sort_idx]

    # Get training source PIDs (paragraphs that cite others)
    train_mask = paragraph_set == "train"
    train_pids = np.where(train_mask)[0]

    train_source_pids = [
        pid for pid in train_pids if len(cited_by_pid.get(int(pid), [])) > 0
    ]

    if max_queries:
        train_source_pids = train_source_pids[:max_queries]

    print(f"Generating training data from {len(train_source_pids)} query paragraphs...")

    all_features = []
    all_labels = []
    all_groups = []

    skipped_no_candidates = 0
    skipped_no_negatives = 0

    for src_pid in tqdm(train_source_pids, desc="Generating training data"):
        src_date = paragraph_dates[src_pid]

        # Get all paragraphs strictly older than source
        cutoff = int(np.searchsorted(sorted_dates, src_date, side="left"))
        if cutoff == 0:
            skipped_no_candidates += 1
            continue

        cand_pids = sort_idx[:cutoff]

        # Ground truth: cited paragraphs
        relevant_pids = set(cited_by_pid[int(src_pid)])
        relevant_pids = {pid for pid in relevant_pids if pid in set(cand_pids)}

        if len(relevant_pids) == 0:
            skipped_no_candidates += 1
            continue

        # Retrieve top candidates using base retriever
        retrieve_k = max(100, len(relevant_pids) * (num_negatives + 1))
        ranked_pids = retriever.retrieve(
            int(src_pid), embeddings, cand_pids, top_k=retrieve_k
        )

        if len(ranked_pids) == 0:
            skipped_no_candidates += 1
            continue

        # Collect positives and negatives
        positives = [pid for pid in ranked_pids if pid in relevant_pids]
        negatives = [pid for pid in ranked_pids if pid not in relevant_pids]

        # Sample negatives
        if len(negatives) > num_negatives * len(positives):
            # Sample more negatives from top ranks (hard negatives)
            hard_negative_ratio = 0.7
            num_hard = int(num_negatives * len(positives) * hard_negative_ratio)
            num_random = num_negatives * len(positives) - num_hard

            hard_negatives = negatives[: min(num_hard, len(negatives))]
            random_negatives = []
            if num_random > 0 and len(negatives) > len(hard_negatives):
                remaining = negatives[len(hard_negatives) :]
                random_indices = np.random.choice(
                    len(remaining), size=min(num_random, len(remaining)), replace=False
                )
                random_negatives = [remaining[i] for i in random_indices]

            negatives = hard_negatives + random_negatives

        if len(negatives) == 0:
            skipped_no_negatives += 1
            continue

        # Combine positives and negatives
        training_pairs = positives + negatives

        # Get query metadata
        query_celex = str(paragraph_celex[src_pid])
        query_par_num = int(paragraph_number[src_pid])
        query_date = paragraph_dates[src_pid]

        # Compute dense similarities
        query_emb = embeddings[src_pid]
        cand_embs = embeddings[training_pairs]

        query_norm = np.linalg.norm(query_emb)
        cand_norms = np.linalg.norm(cand_embs, axis=1)

        if query_norm > 0 and np.all(cand_norms > 0):
            similarities = (query_emb @ cand_embs.T) / (query_norm * cand_norms)
        else:
            similarities = np.zeros(len(training_pairs))

        # Extract features for each pair
        group_features = []
        group_labels = []

        for idx, cand_pid in enumerate(training_pairs):
            try:
                cand_celex = str(paragraph_celex[cand_pid])
                cand_par_num = int(paragraph_number[cand_pid])
                cand_date = paragraph_dates[cand_pid]

                # Extract features
                feature_dict = feature_extractor.extract_features(
                    query_celex=query_celex,
                    query_par_num=query_par_num,
                    cand_celex=cand_celex,
                    cand_par_num=cand_par_num,
                    dense_similarity=float(similarities[idx]),
                    query_date=query_date,
                    cand_date=cand_date,
                )

                # Convert to feature vector
                feature_names = feature_extractor.get_feature_names()
                feature_vec = [feature_dict.get(name, 0.0) for name in feature_names]

                label = 1 if cand_pid in relevant_pids else 0

                group_features.append(feature_vec)
                group_labels.append(label)
            except Exception as e:
                # Skip pairs with errors
                continue

        if group_features:
            all_features.extend(group_features)
            all_labels.extend(group_labels)
            all_groups.append(len(group_features))

    print(f"\nSkipped {skipped_no_candidates} queries with no candidates")
    print(f"Skipped {skipped_no_negatives} queries with no negatives")
    print(
        f"Generated {len(all_groups)} query groups with {len(all_features)} total pairs"
    )
    print(f"Positive samples: {sum(all_labels)}")
    print(f"Negative samples: {len(all_labels) - sum(all_labels)}")

    return all_features, all_labels, all_groups


def train_ltr_model(
    features: list[list[float]],
    labels: list[int],
    groups: list[int],
    feature_names: list[str],
    output_path: str,
) -> None:
    """Train LightGBM ranker model."""
    print("\nTraining LightGBM ranker...")

    # Convert to numpy arrays
    X = np.array(features)
    y = np.array(labels)

    # Split into train/val (80/20)
    # Keep groups together
    np.random.seed(42)
    num_groups = len(groups)
    perm = np.random.permutation(num_groups)
    train_size = int(0.8 * num_groups)

    train_groups_idx = perm[:train_size]
    val_groups_idx = perm[train_size:]

    # Split data by groups
    train_groups = [groups[i] for i in train_groups_idx]
    val_groups = [groups[i] for i in val_groups_idx]

    # Compute indices
    group_starts = np.cumsum([0] + groups)

    train_indices = []
    for idx in train_groups_idx:
        start = group_starts[idx]
        end = group_starts[idx + 1]
        train_indices.extend(range(start, end))

    val_indices = []
    for idx in val_groups_idx:
        start = group_starts[idx]
        end = group_starts[idx + 1]
        val_indices.extend(range(start, end))

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]

    print(f"Train: {len(X_train)} pairs, {len(train_groups)} groups")
    print(f"Val: {len(X_val)} pairs, {len(val_groups)} groups")

    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_train, label=y_train, group=train_groups, feature_name=feature_names
    )
    val_data = lgb.Dataset(
        X_val,
        label=y_val,
        group=val_groups,
        reference=train_data,
        feature_name=feature_names,
    )

    # Training parameters
    params = {
        "objective": "lambdarank",
        "metric": "map",
        "map_eval_at": [5, 10, 100],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
    }

    # Train
    print("\nTraining model...")
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=["train", "val"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
            lgb.log_evaluation(period=10),
            lgb.record_evaluation(evals_result),
        ],
    )

    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save_model(output_path)
    print(f"\nModel saved to {output_path}")

    # Print feature importance
    print("\nTop 20 most important features:")
    importance = model.feature_importance(importance_type="gain")
    feature_importance = sorted(
        zip(feature_names, importance), key=lambda x: x[1], reverse=True
    )
    for name, imp in feature_importance[:20]:
        print(f"  {name}: {imp:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LTR model")
    parser.add_argument(
        "--base-retriever",
        type=str,
        default="dense",
        choices=["dense", "tfidf", "bow"],
        help="Base retriever for initial ranking",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="checkpoints/simcse_citation_model",
        help="Model name for dense retriever",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/par-to-par-cleaned.csv",
        help="Path to citation CSV",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        default="data/par-to-par.json",
        help="Path to metadata JSON",
    )
    parser.add_argument(
        "--judgments-path",
        type=str,
        default="data/judgments_cleaned.json",
        help="Path to judgments JSON",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="checkpoints/ltr/ltr_model.txt",
        help="Output path for trained model",
    )
    parser.add_argument(
        "--embeddings-path",
        type=str,
        default=None,
        help="Path to pre-computed embeddings (optional)",
    )
    parser.add_argument(
        "--train-cutoff-year",
        type=int,
        default=2018,
        help="Year cutoff for train/test split",
    )
    parser.add_argument(
        "--num-negatives",
        type=int,
        default=5,
        help="Number of negative samples per positive",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum number of training queries (for debugging)",
    )

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    df, metadata = load_citation_data(args.csv_path, args.metadata_path)
    train_meta, test_meta = split_train_test(metadata, args.train_cutoff_year)

    (
        pid_to_text,
        celex_number_to_pid,
        paragraph_dates,
        paragraph_celex,
        paragraph_number,
        paragraph_set,
    ) = build_paragraph_index(df, train_meta, test_meta)

    cited_by_pid = build_citation_graph(df, celex_number_to_pid)

    print(f"Total paragraphs: {len(pid_to_text)}")
    print(f"Train paragraphs: {np.sum(paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(paragraph_set == 'test')}")

    # Initialize base retriever
    print(f"\nInitializing {args.base_retriever} retriever...")
    if args.base_retriever == "dense":
        from retrievers import DenseRetriever

        retriever = DenseRetriever(
            model_name=args.model_name,
            max_seq_length=256,
        )
    elif args.base_retriever == "tfidf":
        from retrievers import TfidfRetriever

        retriever = TfidfRetriever()
    else:  # bow
        from retrievers import BOWRetriever

        retriever = BOWRetriever()

    # Load or generate embeddings
    embeddings = None
    if args.embeddings_path and os.path.exists(args.embeddings_path):
        print(f"\nLoading embeddings from {args.embeddings_path}...")
        embeddings = np.load(args.embeddings_path)
        print(f"Loaded embeddings shape: {embeddings.shape}")

    if embeddings is None:
        print("\nGenerating embeddings...")
        train_mask = paragraph_set == "train"
        retriever.fit(pid_to_text, mask=train_mask)
        embeddings = retriever.transform(pid_to_text)
        print(f"Embeddings shape: {embeddings.shape}")

        # Save embeddings if path provided
        if args.embeddings_path:
            os.makedirs(os.path.dirname(args.embeddings_path), exist_ok=True)
            np.save(args.embeddings_path, embeddings)
            print(f"Saved embeddings to {args.embeddings_path}")

    # Initialize feature extractor
    feature_extractor = LTRFeatureExtractor(args.judgments_path)
    feature_extractor.load()

    # Generate training data
    features, labels, groups = generate_training_data(
        retriever=retriever,
        embeddings=embeddings,
        feature_extractor=feature_extractor,
        paragraph_set=paragraph_set,
        paragraph_celex=paragraph_celex,
        paragraph_number=paragraph_number,
        paragraph_dates=paragraph_dates,
        cited_by_pid=cited_by_pid,
        num_negatives=args.num_negatives,
        max_queries=args.max_queries,
    )

    if len(features) == 0:
        print("No training data generated!")
        return

    # Get feature names
    feature_names = feature_extractor.get_feature_names()

    # Train model
    train_ltr_model(features, labels, groups, feature_names, args.output_path)


if __name__ == "__main__":
    main()

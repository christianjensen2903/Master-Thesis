import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluator import Evaluator
from retrievers import DenseRetriever


def evaluate_recall_vs_k(
    retriever: DenseRetriever | None = None,
    preprocessed_dir: str | None = None,
    model_path: str | None = None,
    k_values: list[int] | None = None,
    output_path: str = "artifacts/recall_vs_k.png",
    mode: str = "citation_pairs",
) -> dict[int, float]:
    """
    Evaluate recall at different k values and plot the results.

    Args:
        retriever: Pre-initialized retriever. If None, will create one.
        preprocessed_dir: Path to preprocessed embeddings directory
        model_path: Path to the model (used if retriever is None and preprocessed_dir is None)
        k_values: List of k values to evaluate. If None, uses a default range.
        output_path: Path to save the plot
        mode: Evaluation mode ("citation_pairs" or "all_paragraphs")

    Returns:
        Dictionary mapping k values to recall scores
    """
    if k_values is None:
        # Generate a range of k values: 1, 5, 10, 20, 30, ..., 100, 200, 300, ..., 1000
        k_values = (
            [1, 5, 10]
            + list(range(20, 101, 10))
            + list(range(150, 501, 50))
            + list(range(600, 1001, 100))
        )

    if retriever is None:
        if preprocessed_dir:
            print(
                f"Initializing DenseRetriever with preprocessed_dir: {preprocessed_dir}"
            )
            retriever = DenseRetriever(preprocessed_dir=preprocessed_dir)
        elif model_path:
            print(f"Initializing DenseRetriever with model: {model_path}")
            retriever = DenseRetriever(
                model_name=model_path,
                batch_size=32,
                show_progress_bar=True,
                normalize_embeddings=True,
            )
        else:
            raise ValueError(
                "Must provide either retriever, preprocessed_dir, or model_path"
            )

    # Set top_k to at least the maximum k value to ensure we retrieve enough candidates
    max_k = max(k_values) if k_values else 1000
    print(f"Initializing Evaluator (mode: {mode})...")
    evaluator = Evaluator(
        retriever=retriever,
        mode=mode,  # type: ignore
        judgments_path="data/judgments_cleaned.json",
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,
        top_k=max_k
        + 100,  # Retrieve more than max k to ensure we have enough candidates
    )

    print("Loading and preparing data...")
    evaluator.load_and_prepare()

    assert evaluator.pid_to_text is not None
    assert evaluator.paragraph_set is not None

    print(f"Unique paragraphs: {len(evaluator.pid_to_text)}")
    print(f"Train paragraphs: {np.sum(evaluator.paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(evaluator.paragraph_set == 'test')}")

    # Generate embeddings if not already loaded
    if evaluator.embeddings is None:
        print("\nGenerating embeddings from retriever...")
        train_mask = evaluator.paragraph_set == "train"
        paragraph_ids = [
            (evaluator.paragraph_celex[pid], int(evaluator.paragraph_number[pid]))
            for pid in range(len(evaluator.pid_to_text))
        ]
        evaluator.retriever.fit(evaluator.pid_to_text, mask=train_mask)
        evaluator.embeddings = evaluator.retriever.transform(
            evaluator.pid_to_text, paragraph_ids=paragraph_ids
        )
        print(f"Embeddings shape: {evaluator.embeddings.shape}")

    # Embed queries
    evaluator._embed_queries()

    # Evaluate recall at different k values
    print(f"\nEvaluating recall at k values: {k_values}")
    _, recall_scores = evaluator.evaluate_iterative(k_values=k_values)

    # Print results
    print("\nRecall@k results:")
    for k in sorted(recall_scores.keys()):
        recall = recall_scores[k]
        ci = evaluator.recall_cis[k] if evaluator.recall_cis else (recall, recall)
        print(f"Recall@{k}: {recall:.4f} (95% CI [{ci[0]:.4f}, {ci[1]:.4f}])")

    # Plot results
    print(f"\nPlotting results to {output_path}...")
    k_sorted = sorted(recall_scores.keys())
    recall_sorted = [recall_scores[k] for k in k_sorted]

    plt.figure(figsize=(10, 6))
    plt.plot(
        k_sorted, recall_sorted, marker="o", linestyle="-", linewidth=2, markersize=6
    )
    plt.xlabel("k", fontsize=12)
    plt.ylabel("Recall", fontsize=12)

    retriever_name = "DenseRetriever"
    if preprocessed_dir:
        retriever_name = Path(preprocessed_dir).name
    elif model_path:
        retriever_name = Path(model_path).name

    plt.title(f"Recall vs k ({retriever_name})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.xlim(left=1)
    plt.ylim(bottom=0, top=1.05)

    # Add value annotations for key points
    for i, (k, recall) in enumerate(zip(k_sorted, recall_sorted)):
        if k in [1, 5, 10, 50, 100, 500, 1000] or i % 5 == 0:
            plt.annotate(
                f"{recall:.3f}",
                (k, recall),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=8,
            )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    return recall_scores


if __name__ == "__main__":
    # Example usage with preprocessed embeddings
    recall_scores = evaluate_recall_vs_k(
        preprocessed_dir="data/preprocessed_new",
        output_path="artifacts/recall_vs_k.png",
        mode="citation_pairs",
    )

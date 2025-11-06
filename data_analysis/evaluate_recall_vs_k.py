import matplotlib.pyplot as plt  # type: ignore
import numpy as np

from evaluator import Evaluator
from retrievers import DenseRetriever


def evaluate_recall_vs_k(
    model_path: str = "checkpoints/simcse_citation_model",
    k_values: list[int] | None = None,
    output_path: str = "artifacts/recall_vs_k.png",
    mode: str = "citation_pairs",
) -> dict[int, float]:
    """
    Evaluate recall at different k values and plot the results.

    Args:
        model_path: Path to the simcse_citation_model
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

    print(f"Initializing DenseRetriever with model: {model_path}")
    retriever = DenseRetriever(
        model_name=model_path,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Set top_k to at least the maximum k value to ensure we retrieve enough candidates
    max_k = max(k_values) if k_values else 1000
    print(f"Initializing Evaluator (mode: {mode})...")
    evaluator = Evaluator(
        retriever=retriever,
        mode=mode,  # type: ignore
        csv_path="data/par-to-par-cleaned.csv",
        metadata_path="data/par-to-par.json",
        judgments_path="data/judgments_cleaned.json",
        train_cutoff_year=2018,
        top_k=max_k + 100,  # Retrieve more than max k to ensure we have enough candidates
    )

    print("Loading and preparing data...")
    evaluator.load_and_prepare()

    assert evaluator.pid_to_text is not None
    assert evaluator.paragraph_set is not None

    print(f"Unique paragraphs: {len(evaluator.pid_to_text)}")
    print(f"Train paragraphs: {np.sum(evaluator.paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(evaluator.paragraph_set == 'test')}")

    # Generate embeddings
    print("\nGenerating embeddings from retriever...")
    evaluator.embeddings = evaluator.retriever.transform(evaluator.pid_to_text)
    print(f"Embeddings shape: {evaluator.embeddings.shape}")

    # Evaluate recall at different k values
    print(f"\nEvaluating recall at k values: {k_values}")
    recall_scores = evaluator.evaluate_recall(k_values=k_values)

    # Print results
    print("\nRecall@k results:")
    for k in sorted(recall_scores.keys()):
        print(f"Recall@{k}: {recall_scores[k]:.4f}")

    # Plot results
    print(f"\nPlotting results to {output_path}...")
    k_sorted = sorted(recall_scores.keys())
    recall_sorted = [recall_scores[k] for k in k_sorted]

    plt.figure(figsize=(10, 6))
    plt.plot(k_sorted, recall_sorted, marker="o", linestyle="-", linewidth=2, markersize=6)
    plt.xlabel("k", fontsize=12)
    plt.ylabel("Recall", fontsize=12)
    plt.title(f"Recall vs k using {model_path.split('/')[-1]}", fontsize=14)
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
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {output_path}")

    return recall_scores


if __name__ == "__main__":
    recall_scores = evaluate_recall_vs_k(
        model_path="checkpoints/simcse_citation_model",
        output_path="artifacts/recall_vs_k_simcse.png",
        mode="citation_pairs",
    )


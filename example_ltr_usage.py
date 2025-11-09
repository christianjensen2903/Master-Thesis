"""
Example usage of LTR retriever with the evaluator.

This script demonstrates:
1. How to train an LTR model (or use a pre-trained one)
2. How to evaluate the LTR model using the evaluator

Usage:
    # Train LTR model first:
    python train_ltr.py --base-retriever dense --output-path checkpoints/ltr/ltr_model.txt

    # Then evaluate:
    python example_ltr_usage.py
"""

from retrievers import DenseRetriever, LTRRetriever
from evaluator import Evaluator


def main() -> None:
    # Initialize base retriever (used for initial ranking)
    base_retriever = DenseRetriever(
        model_name="checkpoints/simcse_citation_model",
        max_seq_length=256,
    )

    # Initialize LTR retriever with base retriever
    ltr_retriever = LTRRetriever(
        base_retriever=base_retriever,
        model_path="checkpoints/ltr/ltr_model.txt",  # Path to trained model
        judgments_path="data/judgments_cleaned.json",
        rerank_top_k=1000,  # Only rerank top 1000 from base retriever
    )

    # Initialize evaluator
    evaluator = Evaluator(
        retriever=ltr_retriever,
        mode="citation_pairs",
        csv_path="data/par-to-par-cleaned.csv",
        metadata_path="data/par-to-par.json",
        judgments_path="data/judgments_cleaned.json",
        train_cutoff_year=2018,
        top_k=10000,
        save_embeddings_path="artifacts/ltr_embeddings.npy",
    )

    # Run evaluation (evaluator automatically sets metadata arrays for LTR retriever)
    score = evaluator.run()
    print(f"\nFinal MAP: {score:.3f}")


if __name__ == "__main__":
    main()

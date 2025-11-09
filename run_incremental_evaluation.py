"""
Script to run incremental GNN evaluation.

This script demonstrates how to use the IncrementalGNNEvaluator to evaluate
a GNN model in a temporally-aware manner that prevents data leakage.
"""

import torch
from sentence_transformers import SentenceTransformer  # type: ignore

from example_gnn_usage import CitationGNN
from incremental_gnn_evaluator import IncrementalGNNEvaluator, EvaluatorMode


def main() -> None:
    # Configuration
    encoding_model = "checkpoints/simcse_citation_model"
    model_path = "checkpoints/gnn/best_model.pt"
    mode: EvaluatorMode = "citation_pairs"  # "citation_pairs" or "all_paragraphs"
    csv_path = "data/par-to-par-cleaned.csv"
    metadata_path = "data/par-to-par.json"
    judgments_path = "data/judgments_cleaned.json"
    train_cutoff_year = 2018
    k_hops = 2  # How many hops to re-embed when adding a node
    top_k = 10000  # Limit retrieval to top-k candidates
    embeddings_cache_dir = (
        "artifacts/embeddings_cache"  # Cache directory for text embeddings
    )

    # Load text encoder to get embedding dimension
    print("Loading text encoder...")
    text_encoder = SentenceTransformer(encoding_model)
    in_channels = text_encoder.get_sentence_embedding_dimension()

    # Initialize GNN model
    print("Initializing GNN model...")
    model = CitationGNN(
        in_channels, hidden_dim=512, output_dim=in_channels, num_layers=2
    )

    # Load trained weights
    print(f"Loading model from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # Create incremental evaluator
    evaluator = IncrementalGNNEvaluator(
        gnn_model=model,
        text_encoder_name=encoding_model,
        mode=mode,
        csv_path=csv_path,
        metadata_path=metadata_path,
        judgments_path=judgments_path,
        train_cutoff_year=train_cutoff_year,
        k_hops=k_hops,
        top_k=top_k,
        device="cuda" if torch.cuda.is_available() else "cpu",
        normalize_embeddings=True,
        embeddings_cache_dir=embeddings_cache_dir,
    )

    # Run evaluation
    print("\nStarting incremental evaluation...")
    print("=" * 80)
    metrics = evaluator.run()

    # Print summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Queries evaluated: {len(evaluator.query_results)}")

    if evaluator.query_results:
        avg_candidates = sum(
            r["num_candidates"] for r in evaluator.query_results
        ) / len(evaluator.query_results)
        avg_relevant = sum(r["num_relevant"] for r in evaluator.query_results) / len(
            evaluator.query_results
        )
        print(f"Avg. candidate pool size: {avg_candidates:.1f}")
        print(f"Avg. relevant docs per query: {avg_relevant:.1f}")


if __name__ == "__main__":
    main()

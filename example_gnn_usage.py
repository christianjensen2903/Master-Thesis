import pandas as pd  # type: ignore
import numpy as np
from trainers.gnn_trainer import GNNTrainer
from retrievers import GNNRetriever, DenseRetriever, TfidfRetriever
from data_loader import (
    load_citation_data,
    split_train_test,
    build_paragraph_index,
    build_citation_graph,
)
from utils import build_temporal_dag, validate_temporal_dag
from evaluator import Evaluator


def train_example() -> None:
    """Example: Train a GNN model."""
    print("\n" + "=" * 80)
    print("Training GNN Model")
    print("=" * 80 + "\n")

    # Initialize trainer
    trainer = GNNTrainer(
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=256,
        output_dim=192,
        num_layers=3,
        num_heads=2,
        dropout=0.2,
        output_path="checkpoints/gnn",
        batch_size=8,
        neighbor_batch_size=32,
        epochs=10,
        learning_rate=1e-4,
        temperature=0.07,
        num_negatives=2,
        validation_split=0.1,
        use_wandb=True,
    )

    # Train on paragraph pairs
    cutoff_date = pd.Timestamp("2018-01-01")
    trainer.train("data/par-to-par-og.csv", cutoff_date)

    print("\nTraining complete!")


def inference_example() -> None:
    """Example: Use a pretrained GNN model for inference."""
    print("\n" + "=" * 80)
    print("Inference with Pretrained GNN")
    print("=" * 80 + "\n")

    # Load data

    df, metadata = load_citation_data("data/par-to-par-og.csv", "data/par-to-par.json")
    train_meta, test_meta = split_train_test(metadata, cutoff_year=2018)
    pid_to_text, text_to_pid, paragraph_dates, _, paragraph_set = build_paragraph_index(
        df, train_meta, test_meta
    )
    citation_graph = build_citation_graph(df, text_to_pid)

    # Build temporal DAG (causally masked: only older → newer edges)
    temporal_dag = build_temporal_dag(citation_graph, paragraph_dates)
    validate_temporal_dag(temporal_dag, paragraph_dates)

    print(f"Loaded {len(pid_to_text)} paragraphs")
    print(f"Raw citation graph: {sum(len(v) for v in citation_graph.values())} edges")
    print(
        f"Temporal DAG: {sum(len(v) for v in temporal_dag.values())} edges (causally valid)"
    )

    # Initialize retriever
    retriever = GNNRetriever(
        model_path="checkpoints/gnn/best_model.pt",
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=96,
        output_dim=192,
        num_layers=2,
        num_heads=2,
    )

    # Fit on corpus with temporal DAG (causality built-in)
    print("\nFitting retriever on corpus (temporal DAG - causality enforced)...")
    retriever.fit(
        texts=pid_to_text, citation_graph=temporal_dag, paragraph_dates=paragraph_dates
    )

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = retriever.transform(pid_to_text)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Example retrieval
    query_idx = 100
    # Get all candidates before the query date (realistic retrieval scenario)
    candidate_indices = np.arange(0, query_idx)

    print(f"\nRetrieving for query {query_idx}:")
    print(f"Query text: {pid_to_text[query_idx][:100]}...")

    # Retrieve top-10
    top_k = 10
    ranked_candidates = retriever.retrieve(
        query_idx=query_idx,
        embeddings=embeddings,
        candidate_indices=candidate_indices,
        top_k=top_k,
    )

    print(f"\nTop-{top_k} retrieved paragraphs:")
    for rank, pid in enumerate(ranked_candidates[:top_k], 1):
        similarity = embeddings[pid] @ embeddings[query_idx]
        print(f"{rank}. PID={pid}, Similarity={similarity:.4f}")
        print(f"   Text: {pid_to_text[pid][:80]}...")


def evaluate_gnn_map(
    model_path: str = "checkpoints/gnn/best_model.pt",
    cutoff_year: int = 2018,
    top_k: int | None = 1000,
) -> float:
    """Evaluate MAP of the GNN model."""
    print("\n" + "=" * 80)
    print("Evaluating GNN Model MAP")
    print("=" * 80 + "\n")

    # Load data
    print("Loading data...")
    df, metadata = load_citation_data("data/par-to-par-og.csv", "data/par-to-par.json")
    train_meta, test_meta = split_train_test(metadata, cutoff_year=cutoff_year)
    pid_to_text, text_to_pid, paragraph_dates, _, paragraph_set = build_paragraph_index(
        df, train_meta, test_meta
    )
    citation_graph = build_citation_graph(df, text_to_pid)

    # Build temporal DAG (causally masked: only older → newer edges)
    print("\nBuilding temporal DAG...")
    temporal_dag = build_temporal_dag(citation_graph, paragraph_dates)
    validate_temporal_dag(temporal_dag, paragraph_dates)

    print(f"Loaded {len(pid_to_text)} paragraphs")
    print(f"Train paragraphs: {np.sum(paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(paragraph_set == 'test')}")
    print(f"Raw citation graph: {sum(len(v) for v in citation_graph.values())} edges")
    print(
        f"Temporal DAG: {sum(len(v) for v in temporal_dag.values())} edges (causally valid)"
    )

    # Initialize retriever
    print("\nInitializing GNN retriever...")
    retriever = GNNRetriever(
        model_path=model_path,
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=96,
        output_dim=192,
        num_layers=2,
        num_heads=2,
    )

    # Fit on corpus with temporal DAG
    print("\nFitting retriever on corpus (temporal DAG - causality enforced)...")
    retriever.fit(
        texts=pid_to_text, citation_graph=temporal_dag, paragraph_dates=paragraph_dates
    )

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = retriever.transform(pid_to_text)
    print(f"Generated embeddings with shape: {embeddings.shape}")

    # Evaluate MAP
    evaluator = Evaluator(
        retriever=retriever,
        embeddings=embeddings,
        csv_path="data/par-to-par-og.csv",
        metadata_path="data/par-to-par.json",
        train_cutoff_year=cutoff_year,
        top_k=top_k,
    )

    map_score = evaluator.run()

    return map_score


if __name__ == "__main__":
    train_example()
    # inference_example()
    evaluate_gnn_map()

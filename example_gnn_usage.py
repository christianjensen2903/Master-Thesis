import pandas as pd  # type: ignore
import numpy as np
from gnn_trainers import GNNTrainer
from retrievers import GNNRetriever, DenseRetriever, TfidfRetriever
from data_loader import (
    load_citation_data,
    split_train_test,
    build_paragraph_index,
    build_citation_graph,
)
from utils import (
    build_temporal_dag,
    validate_temporal_dag,
    filter_graph_to_train,
    count_edges,
    validate_no_test_edges,
)
from evaluator import Evaluator


def train_example() -> None:
    """Example: Train a GNN model with embeddings caching."""
    print("\n" + "=" * 80)
    print("Training GNN Model")
    print("=" * 80 + "\n")

    # Initialize trainer with embeddings caching
    # Set embeddings_cache_dir to cache text embeddings for faster subsequent runs
    trainer = GNNTrainer(
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=64,
        output_dim=192,
        num_layers=2,
        num_heads=2,
        dropout=0.2,
        output_path="checkpoints/gnn",
        batch_size=8,
        epochs=10,
        learning_rate=1e-4,
        temperature=0.07,
        num_negatives=2,
        validation_split=0.1,
        use_wandb=True,
        embeddings_cache_dir="artifacts/embeddings_cache",  # Cache embeddings here
    )

    # Train on paragraph pairs
    cutoff_date = pd.Timestamp("2018-01-01")
    trainer.train("data/par-to-par-og.csv", cutoff_date)

    print("\nTraining complete!")


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

    train_temporal_dag = filter_graph_to_train(temporal_dag, paragraph_set)
    validate_no_test_edges(train_temporal_dag, paragraph_set)

    print(f"Loaded {len(pid_to_text)} paragraphs")
    print(f"Train paragraphs: {np.sum(paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(paragraph_set == 'test')}")
    print(f"Raw citation graph: {count_edges(citation_graph)} edges")
    print(f"Temporal DAG (all): {count_edges(temporal_dag)} edges")
    print(
        f"Temporal DAG (train only): {count_edges(train_temporal_dag)} edges (no leakage)"
    )

    # Initialize retriever
    print("\nInitializing GNN retriever...")
    retriever = GNNRetriever(
        model_path=model_path,
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim=64,
        output_dim=192,
        num_layers=2,
        num_heads=2,
    )

    # Fit on corpus with TRAINING temporal DAG only (no leakage)
    print("\nFitting retriever on corpus (TRAIN temporal DAG only - no leakage)...")
    retriever.fit(
        texts=pid_to_text,
        citation_graph=train_temporal_dag,
        paragraph_dates=paragraph_dates,
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
    evaluate_gnn_map()

import pandas as pd  # type: ignore
import numpy as np
import pickle
import os
from sentence_transformers import SentenceTransformer  # type: ignore
from gnn_trainers import GNNTrainer
from torch_geometric.nn import GraphSAGE, GAT
from retrievers import GNNRetriever, DenseRetriever
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

    # Initialize text encoder
    text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Option 1: Optimized GraphSAGE
    model = GraphSAGE(
        in_channels=text_encoder.get_sentence_embedding_dimension(),  # 384 for MiniLM
        hidden_channels=256,  # Increased from 32 for better expressiveness
        out_channels=512,  # Increased from 192 for richer embeddings
        num_layers=3,  # Increased from 2 for better graph structure learning
        dropout=0.3,  # Increased for better regularization
        aggr="mean",
    )

    # Option 2: GAT with attention (uncomment to use)
    # model = GAT(
    #     in_channels=384,  # MiniLM embedding dimension
    #     hidden_channels=256,  # Memory-efficient size
    #     out_channels=512,  # Good representation capacity
    #     num_layers=3,  # Optimal depth
    #     dropout=0.2,  # GAT regularization
    #     v2=True,  # Use GATv2 (better than GAT)
    #     heads=4,  # 4 heads for memory efficiency
    # )

    # Initialize trainer with memory-efficient hyperparameters
    trainer = GNNTrainer(
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
        output_path="checkpoints/gnn",
        batch_size=64,  # Increased slightly for faster batching
        epochs=30,  # Reduced - GNNs converge faster than you think!
        eval_every_n_epochs=5,  # Evaluate every 5 epochs (not every epoch)
        learning_rate=5e-4,  # Slightly higher LR for faster convergence
        weight_decay=1e-4,  # Good regularization
        temperature=0.05,  # Lower for harder negatives
        num_negatives=16,  # Good balance of quality vs speed
        gradient_accumulation_steps=4,  # Effective batch_size = 64*4 = 256
        validation_split=0.1,
        use_wandb=False,
        embeddings_cache_dir="artifacts/embeddings_cache",
    )

    # Train on paragraph pairs (pass model to train method)
    cutoff_date = pd.Timestamp("2018-01-01")
    trainer.train(model, "data/par-to-par-og.csv", cutoff_date)

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
    text_encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Use same architecture as training
    # Option 1: GraphSAGE
    model = GraphSAGE(
        in_channels=text_encoder.get_sentence_embedding_dimension(),  # 384
        hidden_channels=256,
        out_channels=512,
        num_layers=3,
        dropout=0.3,
        aggr="mean",
    )

    # Option 2: GAT (uncomment if you trained with GAT)
    # model = GAT(
    #     in_channels=text_encoder.get_sentence_embedding_dimension(),  # 384
    #     hidden_channels=256,
    #     out_channels=512,
    #     num_layers=3,
    #     dropout=0.2,
    #     v2=True,
    #     heads=4,
    # )
    retriever = GNNRetriever(
        gnn_model=model,
        model_path=model_path,
        text_encoder_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    print("\nFitting retriever on corpus...")

    embeddings_cache_dir = "artifacts/embeddings_cache"
    cache_files = [f for f in os.listdir(embeddings_cache_dir) if f.endswith(".pkl")]

    text_embeddings = None
    if cache_files:
        cache_path = os.path.join(embeddings_cache_dir, cache_files[0])
        print(f"Loading pre-computed embeddings from {cache_path}...")
        with open(cache_path, "rb") as f:
            text_embeddings = pickle.load(f)
        print(f"Loaded embeddings shape: {text_embeddings.shape}")
    else:
        print("No cached embeddings found, computing with DenseRetriever...")
        dense_retriever = DenseRetriever(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=32,
        )
        text_embeddings = dense_retriever.transform(pid_to_text)
        print(f"Computed embeddings shape: {text_embeddings.shape}")

    retriever.fit(
        texts=pid_to_text,
        citation_graph=train_temporal_dag,
        paragraph_dates=paragraph_dates,
        precomputed_embeddings=text_embeddings,
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

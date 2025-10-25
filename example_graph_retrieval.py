"""
Example script demonstrating graph-based and hybrid retrieval methods.

This implements the approach from combining:
- Case-level network analysis (Mones et al., 2021)
- Paragraph-level semantic similarity (Palmer Olsen et al., 2023)

Two methods are demonstrated:
1. Two-stage filtering: Filter top-k cases, then apply paragraph retrieval
2. Re-ranking: Multiply paragraph similarity by case probability
"""

import numpy as np
from data_loader import (
    load_citation_data,
    split_train_test,
    build_paragraph_index,
    build_citation_graph,
)
from retrievers import TfidfRetriever, DenseRetriever, GraphRetriever, HybridRetriever
from evaluator import Evaluator


def main() -> None:
    print("=" * 80)
    print("Graph-Based Retrieval Example")
    print("=" * 80)
    
    # Load data
    print("\n1. Loading citation data...")
    df, metadata = load_citation_data()
    train_meta, test_meta = split_train_test(metadata, cutoff_year=2018)
    
    print("Building paragraph index...")
    pid_to_text, text_to_pid, paragraph_dates, paragraph_celex, paragraph_set = (
        build_paragraph_index(df, train_meta, test_meta)
    )
    
    print(f"Total paragraphs: {len(pid_to_text)}")
    print(f"Train paragraphs: {np.sum(paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(paragraph_set == 'test')}")
    
    # Build citation graph
    print("\nBuilding citation graph...")
    citation_graph = build_citation_graph(df, text_to_pid)
    print(f"Paragraphs with citations: {len(citation_graph)}")
    
    # Option 1: Graph-based retriever (network features only)
    print("\n" + "=" * 80)
    print("Option 1: Pure Graph-Based Retrieval (Network Features)")
    print("=" * 80)
    
    # First compute TF-IDF embeddings for the TF-IDF feature
    print("\nFitting TF-IDF for network features...")
    tfidf_retriever = TfidfRetriever(
        stop_words="english",
        strip_accents="ascii",
        norm="l2",
    )
    train_mask = paragraph_set == "train"
    tfidf_embeddings = tfidf_retriever.fit_transform(pid_to_text, train_mask)
    print(f"TF-IDF embeddings shape: {tfidf_embeddings.shape}")
    
    print("\nInitializing Graph Retriever...")
    graph_retriever = GraphRetriever(
        citation_graph=citation_graph,
        paragraph_dates=paragraph_dates,
        paragraph_celex=paragraph_celex,
        tfidf_embeddings=tfidf_embeddings,
        n_estimators=100,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    
    print("\nFitting Graph Retriever (Random Forest on network features)...")
    # Note: transform not needed for graph retriever (uses identity)
    graph_embeddings = graph_retriever.fit_transform(pid_to_text, train_mask)
    
    print("\nEvaluating Graph Retriever...")
    evaluator = Evaluator(
        retriever=graph_retriever,
        embeddings=graph_embeddings,
        top_k=1000,
    )
    map_score = evaluator.run()
    print(f"\nGraph Retriever MAP@1000: {map_score:.4f}")
    
    # Option 2: Hybrid retrieval - Two-stage method
    print("\n" + "=" * 80)
    print("Option 2: Hybrid Retrieval (Two-Stage Method)")
    print("=" * 80)
    
    # Use dense retriever for paragraph-level semantics
    print("\nInitializing Dense Retriever for paragraph-level semantics...")
    dense_retriever = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=32,
        show_progress_bar=True,
    )
    
    print("Computing dense embeddings...")
    dense_embeddings = dense_retriever.fit_transform(pid_to_text, train_mask)
    print(f"Dense embeddings shape: {dense_embeddings.shape}")
    
    print("\nInitializing Hybrid Retriever (two-stage)...")
    hybrid_two_stage = HybridRetriever(
        paragraph_retriever=dense_retriever,
        case_retriever=graph_retriever,
        paragraph_celex=paragraph_celex,
        method="two_stage",
        top_k_cases=100,
    )
    
    # Fit is already done for sub-retrievers
    hybrid_two_stage.fit(pid_to_text, train_mask)
    
    print("\nEvaluating Hybrid Retriever (two-stage)...")
    evaluator_hybrid = Evaluator(
        retriever=hybrid_two_stage,
        embeddings=dense_embeddings,
        top_k=1000,
    )
    map_score_hybrid = evaluator_hybrid.run()
    print(f"\nHybrid Two-Stage MAP@1000: {map_score_hybrid:.4f}")
    
    # Option 3: Hybrid retrieval - Re-ranking method
    print("\n" + "=" * 80)
    print("Option 3: Hybrid Retrieval (Re-ranking Method)")
    print("=" * 80)
    
    print("\nInitializing Hybrid Retriever (rerank)...")
    hybrid_rerank = HybridRetriever(
        paragraph_retriever=dense_retriever,
        case_retriever=graph_retriever,
        paragraph_celex=paragraph_celex,
        method="rerank",
        top_k_cases=None,  # Not used in rerank method
    )
    
    # Fit is already done for sub-retrievers
    hybrid_rerank.fit(pid_to_text, train_mask)
    
    print("\nEvaluating Hybrid Retriever (rerank)...")
    evaluator_rerank = Evaluator(
        retriever=hybrid_rerank,
        embeddings=dense_embeddings,
        top_k=1000,
    )
    map_score_rerank = evaluator_rerank.run()
    print(f"\nHybrid Re-ranking MAP@1000: {map_score_rerank:.4f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("Summary of Results")
    print("=" * 80)
    print(f"Graph-based (network features only): {map_score:.4f}")
    print(f"Hybrid two-stage (k=100):            {map_score_hybrid:.4f}")
    print(f"Hybrid re-ranking:                    {map_score_rerank:.4f}")
    
    # Compare with different k values for two-stage
    print("\n" + "=" * 80)
    print("Comparing Different k Values for Two-Stage Method")
    print("=" * 80)
    
    for k in [100, 500, 1000]:
        print(f"\nEvaluating with top_k_cases={k}...")
        hybrid_k = HybridRetriever(
            paragraph_retriever=dense_retriever,
            case_retriever=graph_retriever,
            paragraph_celex=paragraph_celex,
            method="two_stage",
            top_k_cases=k,
        )
        hybrid_k.fit(pid_to_text, train_mask)
        
        evaluator_k = Evaluator(
            retriever=hybrid_k,
            embeddings=dense_embeddings,
            top_k=1000,
        )
        map_k = evaluator_k.run()
        print(f"Two-stage (k={k}) MAP@1000: {map_k:.4f}")


if __name__ == "__main__":
    main()


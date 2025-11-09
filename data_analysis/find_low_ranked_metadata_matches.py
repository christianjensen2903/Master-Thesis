import json
import os
from typing import Any

import numpy as np
from tqdm import tqdm

from data_loader import load_citation_data, split_train_test, build_paragraph_index
from retrievers import DenseRetriever


def load_metadata_dict(metadata_path: str) -> dict[str, dict[str, Any]]:
    """Load metadata and create a mapping from CELEX to metadata."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    return metadata


def get_metadata_for_paragraph(
    celex: str, metadata_dict: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Get metadata for a paragraph given its CELEX ID."""
    case_data = metadata_dict.get(celex, {})
    return case_data.get("meta", {})


def share_metadata(
    celex1: str,
    celex2: str,
    metadata_dict: dict[str, dict[str, Any]],
    fields: list[str] = [
        "rapporteur",
        "advocate_general",
        "authentic_language",
        "defendant",
        "applicant",
    ],
) -> tuple[bool, list[str]]:
    """
    Check if two paragraphs share any metadata fields.

    Returns:
        Tuple of (shares_metadata, list of shared fields)
    """
    meta1 = get_metadata_for_paragraph(celex1, metadata_dict)
    meta2 = get_metadata_for_paragraph(celex2, metadata_dict)

    shared_fields = []
    for field in fields:
        val1 = meta1.get(field)
        val2 = meta2.get(field)

        if val1 and val2:
            # Handle both list and string values
            if isinstance(val1, list) and isinstance(val2, list):
                if set(val1) & set(val2):  # Check for intersection
                    shared_fields.append(field)
            elif isinstance(val1, list):
                if val2 in val1:
                    shared_fields.append(field)
            elif isinstance(val2, list):
                if val1 in val2:
                    shared_fields.append(field)
            elif val1 == val2:
                shared_fields.append(field)

    return len(shared_fields) > 0, shared_fields


def find_low_ranked_metadata_matches(
    retriever: DenseRetriever,
    csv_path: str = "data/par-to-par-cleaned.csv",
    metadata_path: str = "data/par-to-par.json",
    train_cutoff_year: int = 2018,
    max_rank_threshold: int = 100,
    top_k: int | None = 10000,
    min_rank: int = 10,
    embeddings_cache_path: str | None = None,
) -> list[dict[str, Any]]:
    """
    Find paragraphs where:
    1. The correct/relevant paragraph is ranked low by dense retriever
    2. The correct paragraph shares metadata (rapporteur/advocate_general) with query

    Args:
        retriever: Dense retriever instance
        csv_path: Path to citation CSV
        metadata_path: Path to metadata JSON
        train_cutoff_year: Year cutoff for train/test split
        max_rank_threshold: Maximum rank to consider as "low ranked"
        top_k: Top k to retrieve
        min_rank: Minimum rank to consider (to avoid very top results)

    Returns:
        List of examples with query info, relevant paragraph info, rank, and shared metadata
    """
    print("Loading data...")
    df, metadata = load_citation_data(csv_path, metadata_path)
    train_meta, test_meta = split_train_test(metadata, train_cutoff_year)

    (
        pid_to_text,
        celex_number_to_pid,
        paragraph_dates,
        paragraph_celex,
        paragraph_number,
        paragraph_set,
    ) = build_paragraph_index(df, train_meta, test_meta)

    # Build citation graph
    from data_loader import build_citation_graph

    cited_by_pid = build_citation_graph(df, celex_number_to_pid)

    # Load metadata dict
    metadata_dict = load_metadata_dict(metadata_path)

    # Try to load cached embeddings
    embeddings: np.ndarray | None = None
    if embeddings_cache_path and os.path.exists(embeddings_cache_path):
        print(f"Attempting to load cached embeddings from {embeddings_cache_path}...")
        try:
            cached_embeddings = np.load(embeddings_cache_path)
            if len(cached_embeddings) == len(pid_to_text):
                embeddings = cached_embeddings
                print(f"Loaded cached embeddings (shape: {cached_embeddings.shape})")  # type: ignore[union-attr]
            else:
                print(
                    f"Cache size mismatch: {len(cached_embeddings)} vs {len(pid_to_text)}. "
                    "Regenerating embeddings..."
                )
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}. Regenerating...")

    # Generate embeddings if not loaded from cache
    if embeddings is None:
        print("Generating embeddings...")
        train_mask = paragraph_set == "train"
        retriever.fit(pid_to_text, mask=train_mask)
        embeddings = retriever.transform(pid_to_text)

        # Save to cache if path is provided
        if embeddings_cache_path:
            cache_dir = os.path.dirname(embeddings_cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            print(f"Saving embeddings to cache: {embeddings_cache_path}")
            np.save(embeddings_cache_path, embeddings)

    # At this point, embeddings should always be set
    assert embeddings is not None, "Embeddings must be generated or loaded"

    # Prepare temporal index (sort by date)
    sort_idx = np.argsort(paragraph_dates)
    sorted_dates = paragraph_dates[sort_idx]

    # Find examples
    print(f"Searching for low-ranked paragraphs with metadata matches...")
    examples = []

    test_mask = paragraph_set == "test"
    test_pids = np.where(test_mask)[0]

    # Filter to test paragraphs with citations
    test_source_pids = [
        pid for pid in test_pids if len(cited_by_pid.get(int(pid), [])) > 0
    ]

    for src_pid in tqdm(test_source_pids, desc="Processing queries"):
        src_date = paragraph_dates[src_pid]
        src_celex = paragraph_celex[src_pid]

        # Get candidate paragraphs (strictly older)
        cutoff = int(np.searchsorted(sorted_dates, src_date, side="left"))
        if cutoff == 0:
            continue

        cand_pids = sort_idx[:cutoff]

        # Get relevant paragraphs
        relevant_list = cited_by_pid[int(src_pid)]
        relevant_array = np.array(relevant_list, dtype=np.int64)

        # Filter to only candidates
        relevant_mask = np.isin(relevant_array, cand_pids)
        relevant_pids = relevant_array[relevant_mask]

        if len(relevant_pids) == 0:
            continue

        # Retrieve and rank
        ranked_pids = retriever.retrieve(
            int(src_pid), embeddings, cand_pids, top_k=top_k
        )

        # Find ranks of relevant paragraphs
        rank_map = {pid: rank for rank, pid in enumerate(ranked_pids, 1)}

        # Check each relevant paragraph
        for rel_pid in relevant_pids:
            rank = rank_map.get(int(rel_pid))
            if rank is None:
                continue  # Not in top_k results

            # Only consider low-ranked results (but not too low)
            if rank < min_rank or rank > max_rank_threshold:
                continue

            rel_celex = paragraph_celex[rel_pid]

            # Check if they share metadata
            shares, shared_fields = share_metadata(src_celex, rel_celex, metadata_dict)

            if shares:
                # Get metadata values for display
                src_meta = get_metadata_for_paragraph(src_celex, metadata_dict)
                rel_meta = get_metadata_for_paragraph(rel_celex, metadata_dict)

                examples.append(
                    {
                        "query_pid": int(src_pid),
                        "query_celex": src_celex,
                        "query_paragraph": int(paragraph_number[src_pid]),
                        "query_text": pid_to_text[src_pid][:200] + "...",
                        "relevant_pid": int(rel_pid),
                        "relevant_celex": rel_celex,
                        "relevant_paragraph": int(paragraph_number[rel_pid]),
                        "relevant_text": pid_to_text[rel_pid][:200] + "...",
                        "rank": rank,
                        "shared_fields": shared_fields,
                        "query_metadata": {
                            "rapporteur": src_meta.get("rapporteur"),
                            "advocate_general": src_meta.get("advocate_general"),
                            "authentic_language": src_meta.get("authentic_language"),
                        },
                        "relevant_metadata": {
                            "rapporteur": rel_meta.get("rapporteur"),
                            "advocate_general": rel_meta.get("advocate_general"),
                            "authentic_language": rel_meta.get("authentic_language"),
                        },
                    }
                )

    return examples


if __name__ == "__main__":
    # Initialize dense retriever
    retriever = DenseRetriever(
        model_name="checkpoints/simcse_citation_model",
        max_seq_length=256,
    )

    # Cache path for embeddings (using same format as evaluator)
    embeddings_cache_path = "artifacts/simcse_embeddings.npy"

    # Find examples
    examples = find_low_ranked_metadata_matches(
        retriever=retriever,
        csv_path="data/par-to-par-cleaned.csv",
        metadata_path="data/par-to-par.json",
        train_cutoff_year=2018,
        max_rank_threshold=100,
        top_k=10000,
        min_rank=10,
        embeddings_cache_path=embeddings_cache_path,
    )

    print(f"\nFound {len(examples)} examples")

    # Save results
    output_path = "artifacts/low_ranked_metadata_matches.json"
    import os

    os.makedirs("artifacts", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(examples, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")

    # Print some examples
    print("\n" + "=" * 80)
    print("Sample Examples:")
    print("=" * 80)

    for i, ex in enumerate(examples[:5], 1):
        print(f"\nExample {i}:")
        print(f"  Query: {ex['query_celex']} paragraph {ex['query_paragraph']}")
        print(
            f"  Relevant: {ex['relevant_celex']} paragraph {ex['relevant_paragraph']}"
        )
        print(f"  Rank: {ex['rank']}")
        print(f"  Shared fields: {', '.join(ex['shared_fields'])}")
        if ex["query_metadata"]["rapporteur"]:
            print(f"  Query rapporteur: {ex['query_metadata']['rapporteur']}")
        if ex["query_metadata"]["advocate_general"]:
            print(
                f"  Query advocate general: {ex['query_metadata']['advocate_general']}"
            )
        if ex["query_metadata"]["authentic_language"]:
            print(
                f"  Query authentic language: {ex['query_metadata']['authentic_language']}"
            )
        if ex["query_metadata"]["defendant"]:
            print(f"  Query defendant: {ex['query_metadata']['defendant']}")
        if ex["query_metadata"]["applicant"]:
            print(f"  Query applicant: {ex['query_metadata']['applicant']}")
        print(f"  Query text: {ex['query_text']}")
        print(f"  Relevant text: {ex['relevant_text']}")

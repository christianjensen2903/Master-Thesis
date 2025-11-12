import json
import os
import pickle
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

    # Load preprocessed embeddings and metadata
    print("Loading preprocessed embeddings...")
    embeddings_doc = np.load("data/preprocessed/paragraph_embeddings_doc.npy")
    embeddings_query = np.load("data/preprocessed/paragraph_embeddings_query.npy")

    with open("data/preprocessed/paragraph_metadata.pkl", "rb") as f:
        preprocessed_metadata = pickle.load(f)

    print(f"Loaded document embeddings (shape: {embeddings_doc.shape})")
    print(f"Loaded query embeddings (shape: {embeddings_query.shape})")
    print(f"Loaded preprocessed metadata ({len(preprocessed_metadata)} paragraphs)")

    # Create mapping from (celex, paragraph_number) to embedding index
    celex_para_to_emb_idx: dict[tuple[str, int], int] = {}
    for idx, meta in enumerate(preprocessed_metadata):
        celex = meta["celex"]
        para_num = meta["paragraph_number"]
        celex_para_to_emb_idx[(celex, para_num)] = idx

    print(f"Created mapping for {len(celex_para_to_emb_idx)} paragraph-embedding pairs")

    # Create mapping from pid to embedding index
    pid_to_emb_idx: dict[int, int] = {}
    missing_embeddings = 0
    for pid in range(len(pid_to_text)):
        celex = paragraph_celex[pid]
        para_num = int(paragraph_number[pid])
        key = (celex, para_num)
        if key in celex_para_to_emb_idx:
            pid_to_emb_idx[pid] = celex_para_to_emb_idx[key]
        else:
            missing_embeddings += 1

    print(
        f"Mapped {len(pid_to_emb_idx)}/{len(pid_to_text)} pids to embeddings "
        f"({missing_embeddings} missing)"
    )

    # Create pid-indexed embedding arrays
    print("Creating pid-indexed embeddings...")
    embedding_dim = embeddings_doc.shape[1]
    pid_embeddings_doc = np.zeros((len(pid_to_text), embedding_dim), dtype=np.float32)
    pid_embeddings_query = np.zeros((len(pid_to_text), embedding_dim), dtype=np.float32)

    for pid, emb_idx in pid_to_emb_idx.items():
        pid_embeddings_doc[pid] = embeddings_doc[emb_idx]
        pid_embeddings_query[pid] = embeddings_query[emb_idx]

    print(f"Created pid-indexed embeddings (shape: {pid_embeddings_doc.shape})")

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

        # Skip if query embedding is not available
        if src_pid not in pid_to_emb_idx:
            continue

        # Retrieve and rank
        query_embedding = pid_embeddings_query[src_pid]
        ranked_pids = retriever.retrieve(
            query_embedding, pid_embeddings_doc, cand_pids, top_k=top_k
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
                            "defendant": src_meta.get("defendant"),
                            "applicant": src_meta.get("applicant"),
                        },
                        "relevant_metadata": {
                            "rapporteur": rel_meta.get("rapporteur"),
                            "advocate_general": rel_meta.get("advocate_general"),
                            "authentic_language": rel_meta.get("authentic_language"),
                            "defendant": rel_meta.get("defendant"),
                            "applicant": rel_meta.get("applicant"),
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

    # Find examples
    examples = find_low_ranked_metadata_matches(
        retriever=retriever,
        csv_path="data/par-to-par-cleaned.csv",
        metadata_path="data/par-to-par.json",
        train_cutoff_year=2018,
        max_rank_threshold=100,
        top_k=10000,
        min_rank=10,
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

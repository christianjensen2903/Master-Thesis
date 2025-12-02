"""
Analyze whether k closest semantic neighbors and their citations can predict correct citations.

This analysis checks:
1. For a given citing paragraph, find k most similar paragraphs that came before it
2. Check if correct citations are in:
   - The k semantic neighbors themselves
   - The citations made by those k neighbors
3. Test both query-passage and query-query similarity
"""

import os
import pickle
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Fix OpenMP conflict on macOS
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import faiss  # type: ignore

# Set FAISS to single-threaded mode to avoid segmentation faults
faiss.omp_set_num_threads(1)


def load_preprocessed_data(preprocessed_dir: str = "data/preprocessed") -> dict:
    """Load all preprocessed embeddings and metadata."""
    path = Path(preprocessed_dir)

    print("Loading preprocessed data...")

    # Load embeddings
    doc_embeddings = np.load(path / "paragraph_embeddings_doc.npy")
    query_embeddings = np.load(path / "paragraph_embeddings_query.npy")

    # Load metadata
    with open(path / "paragraph_metadata.pkl", "rb") as f:
        paragraph_metadata = pickle.load(f)

    # Load citations
    with open(path / "citations.pkl", "rb") as f:
        citations = pickle.load(f)

    print(f"  Loaded {len(paragraph_metadata)} paragraphs")
    print(f"  Doc embeddings shape: {doc_embeddings.shape}")
    print(f"  Query embeddings shape: {query_embeddings.shape}")
    print(f"  Total citations: {len(citations)}")

    return {
        "doc_embeddings": doc_embeddings,
        "query_embeddings": query_embeddings,
        "metadata": paragraph_metadata,
        "citations": citations,
    }


def build_indices(data: dict) -> dict:
    """Build indices for fast lookups."""
    print("Building indices...")

    metadata = data["metadata"]
    citations = data["citations"]

    # Map paragraph ID to index
    id_to_idx = {p["id"]: i for i, p in enumerate(metadata)}
    idx_to_id = {i: p["id"] for i, p in enumerate(metadata)}

    # Map paragraph ID to timestamp (for temporal ordering)
    idx_to_timestamp = {}
    for i, p in enumerate(metadata):
        date_str = p.get("date")
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                idx_to_timestamp[i] = int(dt.timestamp())
            except:
                pass

    # Build citation graph: who cites whom
    outgoing_citations: dict[str, set[str]] = defaultdict(set)

    # Only keep paragraph-to-paragraph citations
    par_citations = [
        (src, tgt)
        for src, tgt in citations
        if src.startswith("par:") and tgt.startswith("par:")
    ]

    for source, target in par_citations:
        outgoing_citations[source].add(target)

    # Get paragraphs that cite (and have timestamps and exist in metadata)
    citing_paragraphs = {
        src
        for src in outgoing_citations.keys()
        if src in id_to_idx and id_to_idx[src] in idx_to_timestamp
    }

    print(f"  Paragraphs with timestamps: {len(idx_to_timestamp)}")
    print(f"  Citing paragraphs (with dates): {len(citing_paragraphs)}")
    print(f"  Par-to-par citations: {len(par_citations)}")

    return {
        "id_to_idx": id_to_idx,
        "idx_to_id": idx_to_id,
        "idx_to_timestamp": idx_to_timestamp,
        "outgoing_citations": outgoing_citations,
        "citing_paragraphs": citing_paragraphs,
    }


def analyze_neighbors_faiss(
    data: dict,
    indices: dict,
    k_values: list[int],
    sample_citing_ids: set[str] | None = None,
    use_query_query: bool = False,
    only_citing_neighbors: bool = False,
) -> dict:
    """
    Analyze citation prediction using FAISS with temporal constraints.

    Processes paragraphs chronologically, building FAISS index incrementally.
    Each paragraph can only find neighbors among earlier paragraphs.

    Args:
        data: Preprocessed data dict
        indices: Index mappings
        k_values: List of k values to test
        sample_citing_ids: Set of citing paragraph IDs to analyze (None = all)
        use_query_query: If True, use query-query similarity; if False, use query-passage
        only_citing_neighbors: If True, only include neighbors that have made citations

    Returns:
        Dict with results for each k
    """
    doc_embeddings = data["doc_embeddings"]
    query_embeddings = data["query_embeddings"]

    id_to_idx = indices["id_to_idx"]
    idx_to_id = indices["idx_to_id"]
    idx_to_timestamp = indices["idx_to_timestamp"]
    outgoing_citations = indices["outgoing_citations"]
    citing_paragraphs = indices["citing_paragraphs"]

    # Build set of citing paragraph indices for filtering
    citing_indices = {id_to_idx[pid] for pid in citing_paragraphs}

    # Filter citing paragraphs if sample provided
    if sample_citing_ids:
        citing_to_analyze = citing_paragraphs & sample_citing_ids
    else:
        citing_to_analyze = citing_paragraphs

    print(f"\nWill analyze {len(citing_to_analyze)} citing paragraphs")
    print(f"Similarity mode: {'query-query' if use_query_query else 'query-passage'}")
    print(f"Only citing neighbors: {only_citing_neighbors}")

    max_k = max(k_values)

    # Normalize embeddings for cosine similarity
    doc_norm = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_norm = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )
    doc_norm = doc_norm.astype(np.float32)
    query_norm = query_norm.astype(np.float32)

    # Group paragraphs by timestamp
    time_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, ts in idx_to_timestamp.items():
        time_to_indices[ts].append(idx)

    unique_times = sorted(time_to_indices.keys())
    print(f"Processing {len(unique_times)} unique time groups chronologically...")

    # Initialize FAISS index
    d = doc_norm.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss_to_orig: list[int] = []  # Maps FAISS index position to original idx

    # Results storage
    results = {
        k: {
            "citations_in_neighbors": [],
            "citations_in_neighbor_citations_only": [],  # Deduplicated: not in neighbors
            "citations_in_either": [],
            "total_citations": [],
            "total_candidate_nodes": [],  # Total nodes in candidate pool
            "citation_occurrences": [],  # How many times each citation appears
        }
        for k in k_values
    }

    analyzed_count = 0

    for time_idx, t in enumerate(tqdm(unique_times, desc="Processing time groups")):
        group_indices = time_to_indices[t]

        # Process citing paragraphs in this time group
        if index.ntotal > 0:
            # Find which paragraphs in this group we need to analyze
            citing_in_group = [
                idx for idx in group_indices if idx_to_id[idx] in citing_to_analyze
            ]

            if citing_in_group:
                # Get query embeddings for citing paragraphs
                citing_queries = query_norm[citing_in_group]
                if not citing_queries.flags["C_CONTIGUOUS"]:
                    citing_queries = np.ascontiguousarray(citing_queries)

                # Search for k nearest neighbors
                k_search = min(max_k, index.ntotal)
                similarities, faiss_neighbors = index.search(citing_queries, k_search)

                # Process each citing paragraph
                for i, citing_idx in enumerate(citing_in_group):
                    citing_id = idx_to_id[citing_idx]
                    true_citations = outgoing_citations[citing_id]

                    if not true_citations:
                        continue

                    # Get neighbor IDs (convert FAISS indices to original indices)
                    neighbor_orig_indices = [
                        faiss_to_orig[faiss_neighbors[i, j]]
                        for j in range(k_search)
                        if faiss_neighbors[i, j] >= 0
                    ]
                    neighbor_ids = [idx_to_id[idx] for idx in neighbor_orig_indices]

                    # Analyze for each k value
                    for k in k_values:
                        neighbors = set(neighbor_ids[:k])

                        # Collect citations from neighbors
                        neighbor_citations = set()
                        for neighbor_id in neighbors:
                            neighbor_citations.update(
                                outgoing_citations.get(neighbor_id, set())
                            )

                        # Total candidate pool = neighbors + their citations
                        candidate_pool = neighbors | neighbor_citations

                        # Count how many true citations are found
                        citations_in_neighbors = true_citations & neighbors
                        citations_in_nbr_cites = true_citations & neighbor_citations
                        citations_in_either = true_citations & candidate_pool

                        # Deduplicated: citations found ONLY in neighbor citations
                        citations_only_in_nbr_cites = citations_in_nbr_cites - neighbors

                        # Count occurrences of each true citation in candidate pool
                        occurrence_counts = []
                        for tc in true_citations:
                            count = 0
                            if tc in neighbors:
                                count += 1
                            # Count how many neighbors cite this
                            for neighbor_id in neighbors:
                                if tc in outgoing_citations.get(neighbor_id, set()):
                                    count += 1
                            occurrence_counts.append(count)

                        results[k]["citations_in_neighbors"].append(
                            len(citations_in_neighbors)
                        )
                        results[k]["citations_in_neighbor_citations_only"].append(
                            len(citations_only_in_nbr_cites)
                        )
                        results[k]["citations_in_either"].append(
                            len(citations_in_either)
                        )
                        results[k]["total_citations"].append(len(true_citations))
                        results[k]["total_candidate_nodes"].append(len(candidate_pool))
                        results[k]["citation_occurrences"].extend(occurrence_counts)

                    analyzed_count += 1

        # Add this time group's paragraphs to the index
        if group_indices:
            # Filter to only citing paragraphs if requested
            if only_citing_neighbors:
                indices_to_add = [idx for idx in group_indices if idx in citing_indices]
            else:
                indices_to_add = group_indices

            if indices_to_add:
                if use_query_query:
                    # Use query embeddings for index
                    group_embs = query_norm[indices_to_add]
                else:
                    # Use doc embeddings for index
                    group_embs = doc_norm[indices_to_add]

                if not group_embs.flags["C_CONTIGUOUS"]:
                    group_embs = np.ascontiguousarray(group_embs)

                index.add(group_embs)
                faiss_to_orig.extend(indices_to_add)

    print(f"Analyzed {analyzed_count} citing paragraphs")
    return results


def summarize_results(results: dict, mode: str) -> None:
    """Print summary statistics for the analysis."""
    from collections import Counter

    print(f"\n{'='*80}")
    print(f"Results for {mode} similarity")
    print("=" * 80)

    for k, data in sorted(results.items()):
        total_citations = sum(data["total_citations"])
        total_in_neighbors = sum(data["citations_in_neighbors"])
        total_in_nbr_cites_only = sum(data["citations_in_neighbor_citations_only"])
        total_in_either = sum(data["citations_in_either"])

        n = len(data["total_citations"])

        # Calculate recall metrics
        recall_neighbors = (
            total_in_neighbors / total_citations if total_citations > 0 else 0
        )
        recall_nbr_cites_only = (
            total_in_nbr_cites_only / total_citations if total_citations > 0 else 0
        )
        recall_either = total_in_either / total_citations if total_citations > 0 else 0

        # Average per paragraph
        avg_total = np.mean(data["total_citations"]) if data["total_citations"] else 0
        avg_candidates = (
            np.mean(data["total_candidate_nodes"])
            if data["total_candidate_nodes"]
            else 0
        )

        # Occurrence distribution
        occurrences = data["citation_occurrences"]
        occ_counter = Counter(occurrences)
        total_occ = len(occurrences)

        print(f"\nk = {k}:")
        print(f"  Analyzed paragraphs: {n}")
        print(
            f"  Total citations: {total_citations} (avg {avg_total:.2f} per paragraph)"
        )
        print(f"  Avg candidate pool size: {avg_candidates:.1f} nodes")
        print()
        print(
            f"  Citations in k neighbors:              {total_in_neighbors:6d} ({recall_neighbors*100:5.2f}%)"
        )
        print(
            f"  Citations ONLY in neighbor citations:  {total_in_nbr_cites_only:6d} ({recall_nbr_cites_only*100:5.2f}%)"
        )
        print(
            f"  Citations in either (total recall):    {total_in_either:6d} ({recall_either*100:5.2f}%)"
        )

        # Occurrence distribution
        print()
        print("  Occurrence distribution (how many times correct citations appear):")
        for occ in sorted(occ_counter.keys())[:8]:  # Show up to 8 occurrence levels
            count = occ_counter[occ]
            pct = count / total_occ * 100 if total_occ > 0 else 0
            print(f"    {occ:2d} times: {count:6d} ({pct:5.2f}%)")
        if max(occ_counter.keys()) > 7:
            high_count = sum(c for o, c in occ_counter.items() if o > 7)
            high_pct = high_count / total_occ * 100 if total_occ > 0 else 0
            print(f"    8+ times: {high_count:6d} ({high_pct:5.2f}%)")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze citation prediction from semantic neighbors"
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=str,
        default="data/preprocessed",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of citing paragraphs to sample (0 = all)",
    )
    parser.add_argument(
        "--k-values",
        type=str,
        default="5,10,25,50,100,200",
        help="Comma-separated list of k values to test",
    )
    args = parser.parse_args()

    k_values = [int(k) for k in args.k_values.split(",")]

    # Load data
    data = load_preprocessed_data(args.preprocessed_dir)
    indices = build_indices(data)

    # Sample citing paragraphs if requested
    sample_citing_ids = None
    if args.sample_size > 0:
        all_citing = list(indices["citing_paragraphs"])
        np.random.seed(42)
        sample_citing_ids = set(
            np.random.choice(
                all_citing, min(args.sample_size, len(all_citing)), replace=False
            )
        )
        print(f"\nSampled {len(sample_citing_ids)} citing paragraphs for analysis")

    # Analyze with query-passage similarity (all neighbors)
    print("\n" + "=" * 80)
    print("ANALYSIS 1: Query-Passage Similarity (all neighbors)")
    print("=" * 80)
    results_qp = analyze_neighbors_faiss(
        data, indices, k_values, sample_citing_ids, use_query_query=False
    )
    summarize_results(results_qp, "query-passage (all)")

    # Analyze with query-passage similarity (only citing neighbors)
    print("\n" + "=" * 80)
    print("ANALYSIS 2: Query-Passage Similarity (only citing neighbors)")
    print("=" * 80)
    results_qp_citing = analyze_neighbors_faiss(
        data,
        indices,
        k_values,
        sample_citing_ids,
        use_query_query=False,
        only_citing_neighbors=True,
    )
    summarize_results(results_qp_citing, "query-passage (citing only)")

    # Analyze with query-query similarity
    print("\n" + "=" * 80)
    print("ANALYSIS 3: Query-Query Similarity (all neighbors)")
    print("=" * 80)
    results_qq = analyze_neighbors_faiss(
        data, indices, k_values, sample_citing_ids, use_query_query=True
    )
    summarize_results(results_qq, "query-query (all)")

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (deduplicated)")
    print("=" * 80)
    print(f"\n{'k':>6} | {'Q-P All':>12} | {'Q-P Citing':>12} | {'Q-Q All':>12}")
    print(
        f"{'':>6} | {'Nodes':>5} {'Total':>6} | {'Nodes':>5} {'Total':>6} | {'Nodes':>5} {'Total':>6}"
    )
    print("-" * 55)

    for k in k_values:
        qp_total = sum(results_qp[k]["total_citations"])
        qpc_total = sum(results_qp_citing[k]["total_citations"])
        qq_total = sum(results_qq[k]["total_citations"])

        if qp_total == 0 or qq_total == 0:
            continue

        qp_avg_nodes = np.mean(results_qp[k]["total_candidate_nodes"])
        qp_either = sum(results_qp[k]["citations_in_either"]) / qp_total * 100

        qpc_avg_nodes = np.mean(results_qp_citing[k]["total_candidate_nodes"])
        qpc_either = sum(results_qp_citing[k]["citations_in_either"]) / qpc_total * 100

        qq_avg_nodes = np.mean(results_qq[k]["total_candidate_nodes"])
        qq_either = sum(results_qq[k]["citations_in_either"]) / qq_total * 100

        print(
            f"{k:>6} | {qp_avg_nodes:>5.0f} {qp_either:>5.1f}% | {qpc_avg_nodes:>5.0f} {qpc_either:>5.1f}% | {qq_avg_nodes:>5.0f} {qq_either:>5.1f}%"
        )

    # Detailed comparison table
    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN (Query-Passage: All vs Citing-Only Neighbors)")
    print("=" * 80)
    print(f"\n{'k':>6} | {'--- All Neighbors ---':>30} | {'--- Citing Only ---':>30}")
    print(
        f"{'':>6} | {'Nodes':>6} {'InNbrs':>8} {'NbrCites':>8} {'Total':>7} | {'Nodes':>6} {'InNbrs':>8} {'NbrCites':>8} {'Total':>7}"
    )
    print("-" * 80)

    for k in k_values:
        qp_total = sum(results_qp[k]["total_citations"])
        qpc_total = sum(results_qp_citing[k]["total_citations"])

        if qp_total == 0:
            continue

        qp_avg_nodes = np.mean(results_qp[k]["total_candidate_nodes"])
        qp_nbrs = sum(results_qp[k]["citations_in_neighbors"]) / qp_total * 100
        qp_only = (
            sum(results_qp[k]["citations_in_neighbor_citations_only"]) / qp_total * 100
        )
        qp_either = sum(results_qp[k]["citations_in_either"]) / qp_total * 100

        qpc_avg_nodes = np.mean(results_qp_citing[k]["total_candidate_nodes"])
        qpc_nbrs = sum(results_qp_citing[k]["citations_in_neighbors"]) / qpc_total * 100
        qpc_only = (
            sum(results_qp_citing[k]["citations_in_neighbor_citations_only"])
            / qpc_total
            * 100
        )
        qpc_either = sum(results_qp_citing[k]["citations_in_either"]) / qpc_total * 100

        print(
            f"{k:>6} | {qp_avg_nodes:>6.0f} {qp_nbrs:>7.1f}% {qp_only:>7.1f}% {qp_either:>6.1f}% | {qpc_avg_nodes:>6.0f} {qpc_nbrs:>7.1f}% {qpc_only:>7.1f}% {qpc_either:>6.1f}%"
        )


if __name__ == "__main__":
    main()

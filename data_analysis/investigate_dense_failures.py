"""
Investigate Dense vs TF-IDF ranking behavior on high overlap citations.
Analyzes what both retrievers rank higher than the correct document.
"""

import csv
import json
import sys
from datetime import datetime as dt
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_processing.mask_verbatim_passages import find_verbatim_passages
from retrievers import DenseRetriever, TfidfRetriever


def calculate_verbatim_percentage(
    text_from: str, text_to: str, min_length: int = 50
) -> float:
    if not text_from or not text_to:
        return 0.0
    passages = find_verbatim_passages(text_from, [text_to], min_length=min_length)
    if not passages:
        return 0.0
    total_verbatim_chars = sum(end - start for start, end in passages)
    return total_verbatim_chars / len(text_from)


def main():
    print("Loading data...")

    with open("data/judgments_cleaned.json") as f:
        judgments = json.load(f)

    paragraphs: list[dict] = []
    for celex, judgment in judgments.items():
        meta = judgment.get("meta", {})
        date_str = meta.get("date")
        try:
            date = dt.strptime(date_str, "%Y-%m-%d")
        except:
            continue
        year = date.year
        set_type = "train" if year < 2018 else "test"
        for par_num, text in judgment["paragraphs"].items():
            paragraphs.append(
                {
                    "celex": celex,
                    "number": int(par_num),
                    "text": text,
                    "set_type": set_type,
                }
            )

    paragraphs.sort(key=lambda p: (p["celex"], p["number"]))
    celex_number_to_pid = {
        (p["celex"], p["number"]): i for i, p in enumerate(paragraphs)
    }
    pid_to_text = np.array([p["text"] for p in paragraphs], dtype=object)
    paragraph_set = np.array([p["set_type"] for p in paragraphs], dtype=object)
    paragraph_celex = np.array([p["celex"] for p in paragraphs], dtype=object)

    # Load cached verbatim data as lookup
    print("Loading cached verbatim data...")
    with open("artifacts/citation_pairs_with_verbatim.json") as f:
        all_cached_pairs = json.load(f)

    # Build lookup: (query_key, doc_key) -> verbatim_pct
    verbatim_lookup: dict[tuple[tuple[str, int], tuple[str, int]], float] = {}
    for pair in all_cached_pairs:
        query_key = tuple(pair["query_key"])
        doc_key = tuple(pair["doc_key"])
        verbatim_lookup[(query_key, doc_key)] = pair["verbatim_pct"]

    # Load citation pairs from par-to-par-cleaned.csv (like evaluator mode=citation_pairs)
    print("Loading citation pairs from par-to-par-cleaned.csv...")
    citation_pairs: list[dict] = []
    with open("data/par-to-par-cleaned.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query_key = (str(row["CELEX_FROM"]), int(row["NUMBER_FROM"]))
            doc_key = (str(row["CELEX_TO"]), int(row["NUMBER_TO"]))

            if (
                query_key not in celex_number_to_pid
                or doc_key not in celex_number_to_pid
            ):
                continue

            query_pid = celex_number_to_pid[query_key]
            if paragraph_set[query_pid] != "test":
                continue

            # Get verbatim from cache
            verbatim_pct = verbatim_lookup.get((query_key, doc_key), 0.0)

            citation_pairs.append(
                {
                    "query_key": query_key,
                    "doc_key": doc_key,
                    "query_pid": query_pid,
                    "doc_pid": celex_number_to_pid[doc_key],
                    "text_from": str(row["TEXT_FROM"]),
                    "text_to": str(row["TEXT_TO"]),
                    "verbatim_pct": verbatim_pct,
                }
            )

    print(f"Loaded {len(citation_pairs)} test citation pairs")
    high_overlap = [p for p in citation_pairs if p["verbatim_pct"] >= 0.9]
    print(f"Found {len(high_overlap)} high overlap pairs (>=90% verbatim)")

    # Initialize retrievers
    print("\nInitializing retrievers...")
    dense_retriever = DenseRetriever(preprocessed_dir="data/preprocessed_new")
    tfidf_retriever = TfidfRetriever()
    tfidf_retriever.fit(pid_to_text)

    # Get all embeddings
    paragraph_ids = list(celex_number_to_pid.keys())
    paragraph_ids.sort(key=lambda k: celex_number_to_pid[k])

    print("Computing dense embeddings...")
    dense_doc_embeddings = dense_retriever.transform(
        pid_to_text, paragraph_ids=[(celex, num) for celex, num in paragraph_ids]
    )
    tfidf_doc_matrix = tfidf_retriever.transform(pid_to_text)

    # Analyze all high overlap pairs
    print("\n" + "=" * 80)
    print("ANALYZING ALL HIGH OVERLAP PAIRS")
    print("=" * 80)

    results = []
    for pair in tqdm(high_overlap, desc="Analyzing pairs"):
        query_pid = pair["query_pid"]
        correct_doc_pid = pair["doc_pid"]
        query_text = pair["text_from"]

        # Dense ranking
        query_emb = dense_retriever.transform_queries(
            np.array([query_text]), paragraph_ids=[pair["query_key"]]
        )
        dense_sims = np.dot(query_emb, dense_doc_embeddings.T).flatten()
        dense_ranking = (-dense_sims).argsort()
        dense_rank = int(np.where(dense_ranking == correct_doc_pid)[0][0])

        # TF-IDF ranking (use transform, not transform_queries)
        tfidf_query = tfidf_retriever.transform(np.array([query_text]))
        tfidf_sims = (tfidf_query @ tfidf_doc_matrix.T).toarray().flatten()
        tfidf_ranking = (-tfidf_sims).argsort()
        tfidf_rank = int(np.where(tfidf_ranking == correct_doc_pid)[0][0])

        results.append(
            {
                "query_key": pair["query_key"],
                "doc_key": pair["doc_key"],
                "query_text": query_text,
                "doc_text": pair["text_to"],
                "verbatim_pct": pair["verbatim_pct"],
                "dense_rank": dense_rank,
                "tfidf_rank": tfidf_rank,
                "dense_sim": float(dense_sims[correct_doc_pid]),
                "tfidf_sim": float(tfidf_sims[correct_doc_pid]),
                "dense_top_sim": float(dense_sims[dense_ranking[0]]),
                "tfidf_top_sim": float(tfidf_sims[tfidf_ranking[0]]),
                "dense_ranking": dense_ranking,
                "tfidf_ranking": tfidf_ranking,
                "dense_sims": dense_sims,
                "tfidf_sims": tfidf_sims,
            }
        )

    # Summary statistics
    dense_ranks = np.array([r["dense_rank"] for r in results])
    tfidf_ranks = np.array([r["tfidf_rank"] for r in results])

    print("\n" + "-" * 60)
    print("SUMMARY STATISTICS")
    print("-" * 60)

    for k in [1, 5, 10, 50, 100]:
        dense_recall = np.mean(dense_ranks < k) * 100
        tfidf_recall = np.mean(tfidf_ranks < k) * 100
        print(f"Recall@{k:3d}: Dense={dense_recall:5.1f}%  TF-IDF={tfidf_recall:5.1f}%")

    print(
        f"\nMean rank:   Dense={np.mean(dense_ranks):7.1f}  TF-IDF={np.mean(tfidf_ranks):7.1f}"
    )
    print(
        f"Median rank: Dense={np.median(dense_ranks):7.1f}  TF-IDF={np.median(tfidf_ranks):7.1f}"
    )

    # Cases where Dense beats TF-IDF and vice versa
    dense_wins = sum(1 for r in results if r["dense_rank"] < r["tfidf_rank"])
    tfidf_wins = sum(1 for r in results if r["tfidf_rank"] < r["dense_rank"])
    ties = sum(1 for r in results if r["tfidf_rank"] == r["dense_rank"])

    print(
        f"\nHead-to-head: Dense wins={dense_wins}, TF-IDF wins={tfidf_wins}, Ties={ties}"
    )

    # Analyze what gets ranked higher
    print("\n" + "-" * 60)
    print("WHAT GETS RANKED ABOVE CORRECT DOCUMENT")
    print("-" * 60)

    def analyze_higher_ranked(
        results: list[dict], retriever_name: str, use_dense: bool
    ) -> None:
        print(f"\n{retriever_name} Analysis:")

        rank_key = "dense_rank" if use_dense else "tfidf_rank"
        ranking_key = "dense_ranking" if use_dense else "tfidf_ranking"

        # Aggregate: what types of docs get ranked higher?
        same_case_count = 0  # From same case as query
        cited_case_count = 0  # From the cited case (but different paragraph)
        other_case_count = 0

        total_above = 0

        for r in results:
            rank = r[rank_key]
            if rank == 0:
                continue  # Correct doc is top-1

            ranking = r[ranking_key]
            query_celex = r["query_key"][0]
            doc_celex = r["doc_key"][0]

            for pid in ranking[:rank]:  # All docs ranked above correct
                total_above += 1
                pid_celex = paragraph_celex[pid]

                if pid_celex == query_celex:
                    same_case_count += 1
                elif pid_celex == doc_celex:
                    cited_case_count += 1
                else:
                    other_case_count += 1

        if total_above > 0:
            print(f"  Total docs ranked above correct: {total_above}")
            print(
                f"  From same case as query: {same_case_count} ({same_case_count/total_above*100:.1f}%)"
            )
            print(
                f"  From cited case (diff para): {cited_case_count} ({cited_case_count/total_above*100:.1f}%)"
            )
            print(
                f"  From other cases: {other_case_count} ({other_case_count/total_above*100:.1f}%)"
            )
        else:
            print("  All correct documents ranked at position 0!")

    analyze_higher_ranked(results, "Dense", use_dense=True)
    analyze_higher_ranked(results, "TF-IDF", use_dense=False)

    # Detailed breakdown by rank ranges
    print("\n" + "-" * 60)
    print("BREAKDOWN BY RANK RANGES")
    print("-" * 60)

    ranges = [
        (0, 1),
        (1, 5),
        (5, 10),
        (10, 50),
        (50, 100),
        (100, 500),
        (500, float("inf")),
    ]

    print(f"\n{'Range':<15} {'Dense':>8} {'TF-IDF':>8}")
    print("-" * 35)
    for low, high in ranges:
        dense_in_range = sum(1 for r in results if low <= r["dense_rank"] < high)
        tfidf_in_range = sum(1 for r in results if low <= r["tfidf_rank"] < high)
        range_str = f"[{low}, {high})" if high != float("inf") else f"[{low}, inf)"
        print(f"{range_str:<15} {dense_in_range:>8} {tfidf_in_range:>8}")

    # Look at worst cases for each retriever
    print("\n" + "-" * 60)
    print("WORST CASES FOR EACH RETRIEVER (Top 10)")
    print("-" * 60)

    print("\nWorst Dense rankings:")
    worst_dense = sorted(results, key=lambda r: -r["dense_rank"])[:10]
    for r in worst_dense:
        print(
            f"  {r['query_key']} -> {r['doc_key']}: "
            f"Dense rank={r['dense_rank']}, TF-IDF rank={r['tfidf_rank']}, "
            f"verbatim={r['verbatim_pct']*100:.0f}%"
        )

    print("\nWorst TF-IDF rankings:")
    worst_tfidf = sorted(results, key=lambda r: -r["tfidf_rank"])[:10]
    for r in worst_tfidf:
        print(
            f"  {r['query_key']} -> {r['doc_key']}: "
            f"Dense rank={r['dense_rank']}, TF-IDF rank={r['tfidf_rank']}, "
            f"verbatim={r['verbatim_pct']*100:.0f}%"
        )

    # Cases where retrievers strongly disagree
    print("\n" + "-" * 60)
    print("CASES WHERE RETRIEVERS STRONGLY DISAGREE")
    print("-" * 60)

    print("\nDense much better (TF-IDF rank - Dense rank > 50):")
    dense_better = [r for r in results if r["tfidf_rank"] - r["dense_rank"] > 50]
    dense_better.sort(key=lambda r: r["tfidf_rank"] - r["dense_rank"], reverse=True)
    for r in dense_better[:5]:
        print(
            f"  {r['query_key']} -> {r['doc_key']}: "
            f"Dense={r['dense_rank']}, TF-IDF={r['tfidf_rank']}, diff={r['tfidf_rank']-r['dense_rank']}"
        )

    print("\nTF-IDF much better (Dense rank - TF-IDF rank > 50):")
    tfidf_better = [r for r in results if r["dense_rank"] - r["tfidf_rank"] > 50]
    tfidf_better.sort(key=lambda r: r["dense_rank"] - r["tfidf_rank"], reverse=True)
    for r in tfidf_better[:5]:
        print(
            f"  {r['query_key']} -> {r['doc_key']}: "
            f"Dense={r['dense_rank']}, TF-IDF={r['tfidf_rank']}, diff={r['dense_rank']-r['tfidf_rank']}"
        )

    # Similarity score analysis
    print("\n" + "-" * 60)
    print("SIMILARITY SCORE ANALYSIS")
    print("-" * 60)

    dense_correct_sims = [r["dense_sim"] for r in results]
    tfidf_correct_sims = [r["tfidf_sim"] for r in results]
    dense_top_sims = [r["dense_top_sim"] for r in results]
    tfidf_top_sims = [r["tfidf_top_sim"] for r in results]

    print("\nSimilarity to correct document:")
    print(
        f"  Dense:  mean={np.mean(dense_correct_sims):.4f}, std={np.std(dense_correct_sims):.4f}"
    )
    print(
        f"  TF-IDF: mean={np.mean(tfidf_correct_sims):.4f}, std={np.std(tfidf_correct_sims):.4f}"
    )

    print("\nSimilarity to top-1 document:")
    print(
        f"  Dense:  mean={np.mean(dense_top_sims):.4f}, std={np.std(dense_top_sims):.4f}"
    )
    print(
        f"  TF-IDF: mean={np.mean(tfidf_top_sims):.4f}, std={np.std(tfidf_top_sims):.4f}"
    )

    print("\nGap (top-1 sim - correct sim):")
    dense_gaps = [r["dense_top_sim"] - r["dense_sim"] for r in results]
    tfidf_gaps = [r["tfidf_top_sim"] - r["tfidf_sim"] for r in results]
    print(f"  Dense:  mean={np.mean(dense_gaps):.4f}, std={np.std(dense_gaps):.4f}")
    print(f"  TF-IDF: mean={np.mean(tfidf_gaps):.4f}, std={np.std(tfidf_gaps):.4f}")

    # Generate interactive HTML report for cases where TF-IDF beats Dense
    print("\n" + "-" * 60)
    print("GENERATING INTERACTIVE HTML REPORT")
    print("-" * 60)

    # Get cases where TF-IDF is better (sorted by difference)
    tfidf_better_cases = [r for r in results if r["dense_rank"] > r["tfidf_rank"]]
    tfidf_better_cases.sort(
        key=lambda r: r["dense_rank"] - r["tfidf_rank"], reverse=True
    )

    # Build case details with both Dense and TF-IDF rankings
    case_details = []
    for r in tfidf_better_cases:
        query_key = r["query_key"]
        doc_key = r["doc_key"]
        dense_ranking = r["dense_ranking"]
        tfidf_ranking = r["tfidf_ranking"]
        dense_sims = r["dense_sims"]
        tfidf_sims = r["tfidf_sims"]

        query_text = r["query_text"]
        correct_doc_text = r["doc_text"]
        correct_doc_pid = celex_number_to_pid[doc_key]

        # Get top 20 Dense ranked docs with verbatim and similarity
        top_dense_docs = []
        for rank, pid in enumerate(dense_ranking[:20]):
            doc_text = str(pid_to_text[pid])
            verbatim_with_query = calculate_verbatim_percentage(query_text, doc_text)
            top_dense_docs.append(
                {
                    "rank": int(rank),
                    "celex": str(paragraph_celex[pid]),
                    "number": int(paragraphs[pid]["number"]),
                    "text": doc_text,
                    "is_correct": bool(pid == correct_doc_pid),
                    "verbatim_pct": float(verbatim_with_query),
                    "similarity": float(dense_sims[pid]),
                }
            )

        # Get top 20 TF-IDF ranked docs with verbatim and similarity
        top_tfidf_docs = []
        for rank, pid in enumerate(tfidf_ranking[:20]):
            doc_text = str(pid_to_text[pid])
            verbatim_with_query = calculate_verbatim_percentage(query_text, doc_text)
            top_tfidf_docs.append(
                {
                    "rank": int(rank),
                    "celex": str(paragraph_celex[pid]),
                    "number": int(paragraphs[pid]["number"]),
                    "text": doc_text,
                    "is_correct": bool(pid == correct_doc_pid),
                    "verbatim_pct": float(verbatim_with_query),
                    "similarity": float(tfidf_sims[pid]),
                }
            )

        case_details.append(
            {
                "query_key": list(query_key),
                "doc_key": list(doc_key),
                "query_text": query_text,
                "correct_doc_text": correct_doc_text,
                "verbatim_pct": float(r["verbatim_pct"]),
                "dense_rank": int(r["dense_rank"]),
                "tfidf_rank": int(r["tfidf_rank"]),
                "dense_sim": float(r["dense_sim"]),
                "tfidf_sim": float(r["tfidf_sim"]),
                "top_dense_docs": top_dense_docs,
                "top_tfidf_docs": top_tfidf_docs,
            }
        )

    # Save results to JSON for AI analysis
    json_output_path = Path("artifacts/dense_vs_tfidf_cases.json")
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(case_details, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(case_details)} cases to {json_output_path}")

    # Generate HTML
    html_content = (
        """<!DOCTYPE html>
<html>
<head>
    <title>Dense vs TF-IDF Analysis - Cases where TF-IDF wins</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 { color: #333; }
        .summary { background: #fff; padding: 20px; border-radius: 8px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .nav { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .nav button { padding: 10px 20px; cursor: pointer; border: none; background: #007bff; color: white; border-radius: 4px; font-size: 12px; }
        .nav button:hover { background: #0056b3; }
        .nav button.active { background: #28a745; }
        .case { display: none; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .case.active { display: block; }
        .case-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; flex-wrap: wrap; gap: 10px; }
        .case-title { font-size: 18px; font-weight: bold; color: #333; }
        .stats { display: flex; gap: 15px; flex-wrap: wrap; }
        .stat { background: #e9ecef; padding: 8px 15px; border-radius: 4px; font-size: 14px; }
        .stat.good { background: #d4edda; color: #155724; }
        .stat.bad { background: #f8d7da; color: #721c24; }
        .section { margin-top: 20px; }
        .section-title { font-weight: bold; color: #495057; margin-bottom: 10px; border-bottom: 2px solid #dee2e6; padding-bottom: 5px; }
        .text-box { background: #f8f9fa; padding: 15px; border-radius: 4px; border-left: 4px solid #007bff; margin-bottom: 10px; white-space: pre-wrap; word-wrap: break-word; font-size: 14px; }
        .text-box.correct { border-left-color: #28a745; }
        .text-box.query { border-left-color: #6f42c1; }
        .toggle-container { margin: 15px 0; }
        .toggle-btn { padding: 10px 20px; cursor: pointer; border: 2px solid #007bff; background: white; color: #007bff; border-radius: 4px; margin-right: 10px; font-weight: bold; }
        .toggle-btn.active { background: #007bff; color: white; }
        .toggle-btn.dense { border-color: #dc3545; color: #dc3545; }
        .toggle-btn.dense.active { background: #dc3545; color: white; }
        .toggle-btn.tfidf { border-color: #28a745; color: #28a745; }
        .toggle-btn.tfidf.active { background: #28a745; color: white; }
        .ranking-section { display: none; }
        .ranking-section.active { display: block; }
        .ranking-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        .ranking-table th, .ranking-table td { padding: 8px 10px; text-align: left; border-bottom: 1px solid #dee2e6; }
        .ranking-table th { background: #e9ecef; font-size: 13px; }
        .ranking-table tr.correct { background: #d4edda; }
        .ranking-table tr:hover { background: #f1f3f5; }
        .ranking-table .rank { font-weight: bold; width: 80px; }
        .ranking-table .celex { width: 150px; font-family: monospace; font-size: 12px; }
        .ranking-table .similarity { width: 70px; text-align: center; font-family: monospace; }
        .ranking-table .verbatim { width: 70px; text-align: center; }
        .ranking-table .verbatim.high-verbatim { background: #fff3cd; color: #856404; font-weight: bold; }
        .ranking-table .text { font-size: 12px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Dense vs TF-IDF Analysis</h1>
        <div class="summary">
            <h3>Cases where TF-IDF ranks correct document higher than Dense</h3>
            <p>Total cases: <strong>"""
        + str(len(case_details))
        + """</strong></p>
            <p>Click on a case number to inspect. Use the toggle to switch between Dense and TF-IDF rankings.</p>
        </div>
        
        <div class="nav">
"""
    )

    # Add navigation buttons
    for i, case in enumerate(case_details):
        diff = case["dense_rank"] - case["tfidf_rank"]
        html_content += f'            <button onclick="showCase({i})" id="btn-{i}">#{i+1} (Δ{diff})</button>\n'

    html_content += """        </div>
        
"""

    # Helper function to build ranking table
    def build_ranking_table(
        docs: list, correct_rank: int, doc_key: list, verbatim_pct: float
    ) -> str:
        table = """                <table class="ranking-table">
                    <thead>
                        <tr>
                            <th class="rank">Rank</th>
                            <th class="celex">CELEX ¶</th>
                            <th class="similarity">Sim</th>
                            <th class="verbatim">Verbatim</th>
                            <th class="text">Text</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for doc in docs:
            correct_class = "correct" if doc["is_correct"] else ""
            correct_marker = " ✓" if doc["is_correct"] else ""
            doc_verbatim = doc["verbatim_pct"] * 100
            verbatim_class = "high-verbatim" if doc_verbatim >= 50 else ""
            text_preview = (
                doc["text"][:250] + "..." if len(doc["text"]) > 250 else doc["text"]
            )
            text_preview = (
                text_preview.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            table += f"""                        <tr class="{correct_class}">
                            <td class="rank">{doc['rank']}{correct_marker}</td>
                            <td class="celex">{doc['celex']} ¶{doc['number']}</td>
                            <td class="similarity">{doc['similarity']:.3f}</td>
                            <td class="verbatim {verbatim_class}">{doc_verbatim:.0f}%</td>
                            <td class="text">{text_preview}</td>
                        </tr>
"""
        # If correct doc not in top 20
        if correct_rank >= 20:
            table += f"""                        <tr class="correct">
                            <td class="rank">{correct_rank} ✓</td>
                            <td class="celex">{doc_key[0]} ¶{doc_key[1]}</td>
                            <td class="similarity">-</td>
                            <td class="verbatim">{verbatim_pct*100:.0f}%</td>
                            <td class="text">(Correct document at rank {correct_rank})</td>
                        </tr>
"""
        table += """                    </tbody>
                </table>
"""
        return table

    # Add case content
    for i, case in enumerate(case_details):
        diff = case["dense_rank"] - case["tfidf_rank"]
        active = "active" if i == 0 else ""

        # Escape query and doc text
        query_text_escaped = (
            case["query_text"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        doc_text_escaped = (
            case["correct_doc_text"]
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

        html_content += f"""        <div class="case {active}" id="case-{i}">
            <div class="case-header">
                <div class="case-title">Case #{i+1}: {case['query_key']} → {case['doc_key']}</div>
                <div class="stats">
                    <div class="stat bad">Dense: rank {case['dense_rank']} (sim {case['dense_sim']:.3f})</div>
                    <div class="stat good">TF-IDF: rank {case['tfidf_rank']} (sim {case['tfidf_sim']:.3f})</div>
                    <div class="stat">Verbatim: {case['verbatim_pct']*100:.0f}%</div>
                </div>
            </div>
            
            <div class="section">
                <div class="section-title">Query Text (from {case['query_key'][0]} ¶{case['query_key'][1]})</div>
                <div class="text-box query">{query_text_escaped}</div>
            </div>
            
            <div class="section">
                <div class="section-title">Correct Document (from {case['doc_key'][0]} ¶{case['doc_key'][1]})</div>
                <div class="text-box correct">{doc_text_escaped}</div>
            </div>
            
            <div class="section">
                <div class="toggle-container">
                    <button class="toggle-btn dense active" onclick="showRanking({i}, 'dense')">Dense Ranking (rank {case['dense_rank']})</button>
                    <button class="toggle-btn tfidf" onclick="showRanking({i}, 'tfidf')">TF-IDF Ranking (rank {case['tfidf_rank']})</button>
                </div>
                
                <div class="ranking-section active" id="dense-{i}">
                    <div class="section-title">Dense Top 20</div>
"""
        html_content += build_ranking_table(
            case["top_dense_docs"],
            case["dense_rank"],
            case["doc_key"],
            case["verbatim_pct"],
        )

        html_content += f"""                </div>
                
                <div class="ranking-section" id="tfidf-{i}">
                    <div class="section-title">TF-IDF Top 20</div>
"""
        html_content += build_ranking_table(
            case["top_tfidf_docs"],
            case["tfidf_rank"],
            case["doc_key"],
            case["verbatim_pct"],
        )

        html_content += """                </div>
            </div>
        </div>
        
"""

    html_content += """    </div>
    
    <script>
        function showCase(index) {
            document.querySelectorAll('.case').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.nav button').forEach(b => b.classList.remove('active'));
            document.getElementById('case-' + index).classList.add('active');
            document.getElementById('btn-' + index).classList.add('active');
        }
        
        function showRanking(caseIndex, type) {
            const caseEl = document.getElementById('case-' + caseIndex);
            caseEl.querySelectorAll('.ranking-section').forEach(s => s.classList.remove('active'));
            caseEl.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(type + '-' + caseIndex).classList.add('active');
            caseEl.querySelector('.toggle-btn.' + type).classList.add('active');
        }
        
        document.getElementById('btn-0').classList.add('active');
    </script>
</body>
</html>
"""

    # Write HTML file
    output_path = Path("artifacts/dense_vs_tfidf_analysis.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Generated interactive report: {output_path}")
    print(f"Open in browser: file://{output_path.absolute()}")


if __name__ == "__main__":
    main()

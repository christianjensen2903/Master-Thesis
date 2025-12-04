"""
Compute verbatim percentages for citation pairs and save to cache.

This script calculates what percentage of each citing paragraph's text
is copied verbatim from the cited paragraph. Higher percentages indicate
"easier" citations (more lexical overlap).

Usage:
    python data_analysis/compute_verbatim_cache.py
    python data_analysis/compute_verbatim_cache.py --output artifacts/my_cache.json
"""

import argparse
import csv
import json
from datetime import datetime as dt
from pathlib import Path

import numpy as np
from tqdm import tqdm

from data_processing.mask_verbatim_passages import find_verbatim_passages


def calculate_verbatim_percentage(
    text_from: str, text_to: str, min_length: int = 50
) -> float:
    """Calculate the percentage of text_from that is verbatim from text_to."""
    if not text_from or not text_to:
        return 0.0

    passages = find_verbatim_passages(text_from, [text_to], min_length=min_length)

    if not passages:
        return 0.0

    total_verbatim_chars = sum(end - start for start, end in passages)
    return total_verbatim_chars / len(text_from)


def compute_verbatim_cache(
    par_to_par_path: str,
    judgments_path: str,
    output_path: str,
    train_cutoff_year: int = 2018,
    min_verbatim_length: int = 50,
) -> None:
    """Compute verbatim percentages for all test citation pairs."""
    print("Loading judgments...")
    with open(judgments_path) as f:
        judgments = json.load(f)

    # Build paragraph data to determine train/test split
    paragraphs: list[dict] = []
    for celex, judgment in tqdm(judgments.items(), desc="Processing judgments"):
        meta = judgment.get("meta", {})
        date_str = meta.get("date")

        try:
            date = dt.strptime(date_str, "%Y-%m-%d")
        except:
            continue

        year = date.year
        set_type = "train" if year < train_cutoff_year else "test"

        for par_num, text in judgment["paragraphs"].items():
            paragraphs.append({
                "celex": celex,
                "number": int(par_num),
                "set_type": set_type,
            })

    paragraphs.sort(key=lambda p: (p["celex"], p["number"]))

    celex_number_to_pid = {
        (p["celex"], p["number"]): pid for pid, p in enumerate(paragraphs)
    }
    paragraph_set = np.array([p["set_type"] for p in paragraphs], dtype=object)

    # Load citation pairs and compute verbatim percentages
    print(f"\nLoading citation pairs from {par_to_par_path}...")
    citation_pairs: list[dict] = []

    with open(par_to_par_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Computing verbatim percentages for {len(rows)} citation pairs...")
    for row in tqdm(rows, desc="Computing verbatim"):
        celex_from = str(row["CELEX_FROM"])
        number_from = int(row["NUMBER_FROM"])
        text_from = str(row["TEXT_FROM"])
        celex_to = str(row["CELEX_TO"])
        number_to = int(row["NUMBER_TO"])
        text_to = str(row["TEXT_TO"])

        query_key = (celex_from, number_from)
        doc_key = (celex_to, number_to)

        if (
            query_key not in celex_number_to_pid
            or doc_key not in celex_number_to_pid
        ):
            continue

        query_pid = celex_number_to_pid[query_key]

        # Only use test queries
        if paragraph_set[query_pid] != "test":
            continue

        verbatim_pct = calculate_verbatim_percentage(
            text_from, text_to, min_verbatim_length
        )

        citation_pairs.append({
            "query_key": list(query_key),
            "doc_key": list(doc_key),
            "verbatim_pct": verbatim_pct,
        })

    print(f"\nComputed verbatim percentages for {len(citation_pairs)} test citation pairs")

    # Print distribution summary
    pcts = [p["verbatim_pct"] for p in citation_pairs]
    print(f"\nVerbatim percentage distribution:")
    print(f"  Min: {min(pcts):.1%}")
    print(f"  Max: {max(pcts):.1%}")
    print(f"  Mean: {np.mean(pcts):.1%}")
    print(f"  Median: {np.median(pcts):.1%}")

    # Count by bins
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    print(f"\nDistribution by bin:")
    for i in range(len(bins) - 1):
        count = sum(1 for p in pcts if bins[i] <= p < bins[i + 1])
        upper = min(int(bins[i + 1] * 100), 100)
        print(f"  {int(bins[i]*100)}-{upper}%: {count} ({count/len(pcts)*100:.1f}%)")

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(citation_pairs, f)

    print(f"\nSaved cache to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute verbatim percentages for citation pairs"
    )
    parser.add_argument(
        "--par-to-par",
        default="data/par-to-par-cleaned.csv",
        help="Path to par-to-par CSV",
    )
    parser.add_argument(
        "--judgments",
        default="data/judgments_cleaned.json",
        help="Path to judgments JSON",
    )
    parser.add_argument(
        "--output",
        default="artifacts/citation_pairs_with_verbatim.json",
        help="Output path for cache",
    )
    parser.add_argument(
        "--train-cutoff-year",
        type=int,
        default=2018,
        help="Year to split train/test",
    )
    parser.add_argument(
        "--min-verbatim-length",
        type=int,
        default=50,
        help="Minimum characters for verbatim detection",
    )
    args = parser.parse_args()

    compute_verbatim_cache(
        par_to_par_path=args.par_to_par,
        judgments_path=args.judgments,
        output_path=args.output,
        train_cutoff_year=args.train_cutoff_year,
        min_verbatim_length=args.min_verbatim_length,
    )


if __name__ == "__main__":
    main()


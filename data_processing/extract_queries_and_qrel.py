import json
import csv
from pathlib import Path
from datetime import datetime as dt
from collections import defaultdict
from typing import Any

import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore


def load_judgments(judgments_path: str) -> dict[str, dict[str, Any]]:
    """Load judgments.json file."""
    print(f"Loading judgments from {judgments_path}...")
    with open(judgments_path, "r", encoding="utf-8") as f:
        judgments = json.load(f)
    print(f"Loaded {len(judgments)} judgments")
    return judgments


def build_paragraph_index_from_judgments(
    judgments: dict[str, dict[str, Any]], train_cutoff_year: int = 2018
) -> tuple[dict[tuple[str, int], int], dict[int, dict[str, Any]]]:
    """
    Build paragraph index from judgments.json.

    Returns:
        celex_number_to_pid: Mapping from (celex, number) to paragraph ID
        pid_to_info: Mapping from paragraph ID to paragraph info (text, celex, number, date, set_type)
    """
    paragraphs: list[dict[str, Any]] = []

    for celex, judgment in tqdm(judgments.items(), desc="Processing judgments"):
        # Get date from meta
        meta = judgment.get("meta", {})
        date_str = meta.get("date")

        try:
            date = dt.strptime(date_str, "%Y-%m-%d")
        except:
            continue

        year = date.year
        set_type = "train" if year < train_cutoff_year else "test"

        for par_num, text in judgment["paragraphs"].items():
            paragraphs.append(
                {
                    "text": text,
                    "celex": celex,
                    "date": date,
                    "number": int(par_num),
                    "set_type": set_type,
                }
            )

    # Sort paragraphs by (celex, number) to maintain document order
    paragraphs.sort(key=lambda p: (p["celex"], p["number"]))

    # Build mappings
    celex_number_to_pid: dict[tuple[str, int], int] = {}
    pid_to_info: dict[int, dict[str, Any]] = {}

    for pid, p in enumerate(paragraphs):
        key = (p["celex"], p["number"])
        celex_number_to_pid[key] = pid
        pid_to_info[pid] = p

    return celex_number_to_pid, pid_to_info


def extract_queries_and_qrel(
    judgments_path: str,
    par_to_par_path: str,
    output_dir: str,
    train_cutoff_year: int = 2018,
) -> None:
    """
    Extract queries from judgments.json based on par-to-par.csv and create qrel file.

    Args:
        judgments_path: Path to judgments.json
        par_to_par_path: Path to par-to-par.csv
        output_dir: Directory to save output files
        train_cutoff_year: Year cutoff for train/test split
        clean_queries: Whether to clean query text
    """
    # Load data
    judgments = load_judgments(judgments_path)
    print(f"Loading par-to-par data from {par_to_par_path}...")
    df = pd.read_csv(par_to_par_path).dropna()
    print(f"Loaded {len(df)} citation pairs")

    # Build paragraph index from judgments
    celex_number_to_pid, pid_to_info = build_paragraph_index_from_judgments(
        judgments, train_cutoff_year
    )
    print(f"Built index with {len(celex_number_to_pid)} paragraphs")

    # Extract unique queries (FROM paragraphs) and their relevant documents (TO paragraphs)
    query_to_relevant: dict[tuple[str, int], set[tuple[str, int]]] = defaultdict(set)
    query_keys: list[tuple[str, int]] = []
    query_texts: dict[tuple[str, int], str] = {}

    skipped = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing citation pairs"):
        celex_from = str(row["CELEX_FROM"])
        number_from = int(row["NUMBER_FROM"])
        celex_to = str(row["CELEX_TO"])
        number_to = int(row["NUMBER_TO"])

        src_key = (celex_from, number_from)
        tgt_key = (celex_to, number_to)

        # Skip if either key not in our index
        if src_key not in celex_number_to_pid or tgt_key not in celex_number_to_pid:
            skipped += 1
            continue

        # Store query text (original from judgments.json)
        if src_key not in query_texts:
            query_keys.append(src_key)
            src_pid = celex_number_to_pid[src_key]
            query_texts[src_key] = pid_to_info[src_pid]["text"]

        # Add relevant document
        query_to_relevant[src_key].add(tgt_key)

    if skipped > 0:
        print(f"Skipped {skipped}/{len(df)} citation pairs (paragraphs not in index)")

    print(f"Found {len(query_keys)} unique queries")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write queries file (original text)
    queries_file = output_path / "queries.tsv"
    print(f"Writing queries to {queries_file}...")
    with open(queries_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["celex", "paragraph_number", "query"])
        for celex, par_num in query_keys:
            writer.writerow([celex, par_num, query_texts[(celex, par_num)]])
    # Write qrel file
    # Format: query_id 0 doc_id relevance_score
    # Using composite format: celex_paragraph_number
    qrel_file = output_path / "qrel.txt"
    print(f"Writing qrel to {qrel_file}...")
    with open(qrel_file, "w", encoding="utf-8") as f:
        for query_key in sorted(query_keys):
            celex_q, par_num_q = query_key
            query_id = f"{celex_q}_{par_num_q}"
            relevant_docs = sorted(query_to_relevant[query_key])
            for doc_key in relevant_docs:
                celex_d, par_num_d = doc_key
                doc_id = f"{celex_d}_{par_num_d}"
                f.write(f"{query_id} 0 {doc_id} 1\n")

    print(f"\nExtraction complete!")
    print(f"Queries: {len(query_keys)}")
    print(f"Qrel entries: {sum(len(docs) for docs in query_to_relevant.values())}")
    print(f"\nOutput files:")
    print(f"  - {queries_file}")
    print(f"  - {qrel_file}")


def main():
    base_dir = Path(__file__).parent.parent
    judgments_path = base_dir / "data" / "judgments_cleaned.json"
    par_to_par_path = base_dir / "data" / "par-to-par-og.csv"
    output_dir = base_dir / "data" / "evaluation"

    extract_queries_and_qrel(
        judgments_path=str(judgments_path),
        par_to_par_path=str(par_to_par_path),
        output_dir=str(output_dir),
        train_cutoff_year=2018,
    )


if __name__ == "__main__":
    main()

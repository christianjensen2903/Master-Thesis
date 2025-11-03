import json
from datetime import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore


def load_citation_data(
    csv_path: str = "data/par-to-par-cleaned.csv",
    metadata_path: str = "data/par-to-par.json",
) -> tuple[pd.DataFrame, dict]:
    """
    Load citation data from Excel and metadata from JSON.

    Args:
        csv_path: Path to paragraph-to-paragraph CSV file
        metadata_path: Path to metadata JSON file

    Returns:
        Tuple of (DataFrame, metadata_dict)
    """
    df = pd.read_csv(csv_path)
    df = df.dropna()

    with open(metadata_path) as f:
        metadata = json.load(f)

    return df, metadata


def split_train_test(metadata: dict, cutoff_year: int) -> tuple[list[dict], list[dict]]:
    """
    Split metadata into train and test based on year cutoff.

    Args:
        metadata: Dictionary mapping case_id to metadata
        cutoff_year: Years < cutoff go to train, >= go to test

    Returns:
        Tuple of (train_list, test_list)
    """
    train: list[dict] = []
    test: list[dict] = []

    for case_id, m in metadata.items():
        m = dict(m)
        m["case_id"] = case_id
        year = int(m["meta"]["date"].split("-")[0])
        (train if year < cutoff_year else test).append(m)

    return train, test


def build_paragraph_index(
    df: pd.DataFrame,
    train_meta: list[dict],
    test_meta: list[dict],
) -> tuple[
    np.ndarray,
    dict[tuple[str, int], int],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Build paragraph index from citation DataFrame using (celex, number) as key.

    Args:
        df: DataFrame with citation pairs
        train_meta: List of train metadata dicts
        test_meta: List of test metadata dicts

    Returns:
        Tuple of:
        - pid_to_text: Array mapping paragraph ID to text
        - celex_number_to_pid: Dict mapping (celex, number) to paragraph ID
        - paragraph_dates: Array of dates for each paragraph
        - paragraph_celex: Array of CELEX IDs for each paragraph
        - paragraph_number: Array of paragraph numbers for each paragraph
        - paragraph_set: Array of "train"/"test"/None for each paragraph
    """
    train_celex = {m["case_id"] for m in train_meta}
    test_celex = {m["case_id"] for m in test_meta}

    # Collect all unique (celex, number) combinations with their first text and earliest date
    tmp_info: dict[tuple[str, int], dict] = {}

    # Pass 1: Process FROM rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index (FROM)"):
        celex = str(row["CELEX_FROM"])
        number = int(row["NUMBER_FROM"])
        text = row["TEXT_FROM"]
        date_str = row["DATE_FROM"]

        if not isinstance(text, str):
            continue

        d = dt.strptime(date_str, "%Y-%m-%d")
        key = (celex, number)

        if key not in tmp_info:
            tmp_info[key] = {
                "text": text,
                "date": d,
                "celex": celex,
                "number": number,
                "set_type": (
                    "train"
                    if celex in train_celex
                    else ("test" if celex in test_celex else None)
                ),
            }
        elif d < tmp_info[key]["date"]:
            tmp_info[key]["date"] = d

    # Pass 2: Process TO rows
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building index (TO)"):
        celex = str(row["CELEX_TO"])
        number = int(row["NUMBER_TO"])
        text = row["TEXT_TO"]
        date_str = row["DATE_TO"]

        if not isinstance(text, str):
            continue

        d = dt.strptime(date_str, "%Y-%m-%d")
        key = (celex, number)

        if key not in tmp_info:
            tmp_info[key] = {
                "text": text,
                "date": d,
                "celex": celex,
                "number": number,
                "set_type": (
                    "train"
                    if celex in train_celex
                    else ("test" if celex in test_celex else None)
                ),
            }
        elif d < tmp_info[key]["date"]:
            tmp_info[key]["date"] = d

    # Sort by (celex, number) for deterministic ordering
    sorted_keys = sorted(tmp_info.keys())

    # Build arrays
    celex_number_to_pid = {key: pid for pid, key in enumerate(sorted_keys)}
    pid_to_text = np.array([tmp_info[key]["text"] for key in sorted_keys], dtype=object)
    paragraph_dates = np.array(
        [tmp_info[key]["date"] for key in sorted_keys], dtype="datetime64[ns]"
    )
    paragraph_celex = np.array(
        [tmp_info[key]["celex"] for key in sorted_keys], dtype=object
    )
    paragraph_number = np.array(
        [tmp_info[key]["number"] for key in sorted_keys], dtype=object
    )
    paragraph_set = np.array(
        [tmp_info[key]["set_type"] for key in sorted_keys], dtype=object
    )

    return (
        pid_to_text,
        celex_number_to_pid,
        paragraph_dates,
        paragraph_celex,
        paragraph_number,
        paragraph_set,
    )


def build_citation_graph(
    df: pd.DataFrame,
    celex_number_to_pid: dict[tuple[str, int], int],
) -> dict[int, list[int]]:
    """
    Build citation graph mapping source paragraph to cited paragraphs.

    Args:
        df: DataFrame with citation pairs
        celex_number_to_pid: Mapping from (celex, number) to paragraph ID

    Returns:
        Dictionary mapping source pid to sorted list of cited pids
    """
    cited_by_pid = defaultdict(set)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building citations"):
        celex_from = row["CELEX_FROM"]
        number_from = row["NUMBER_FROM"]
        celex_to = row["CELEX_TO"]
        number_to = row["NUMBER_TO"]

        src_key = (str(celex_from), int(number_from))
        tgt_key = (str(celex_to), int(number_to))

        if src_key not in celex_number_to_pid or tgt_key not in celex_number_to_pid:
            continue

        src_pid = celex_number_to_pid[src_key]
        tgt_pid = celex_number_to_pid[tgt_key]
        cited_by_pid[src_pid].add(tgt_pid)

    # Make deterministic and convert to dict
    result: dict[int, list[int]] = {k: sorted(v) for k, v in cited_by_pid.items()}

    return result

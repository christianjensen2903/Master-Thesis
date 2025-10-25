import json
from datetime import datetime as dt
from collections import defaultdict

import numpy as np
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore


def load_citation_data(
    csv_path: str = "data/par-to-par-og.csv",
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
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build paragraph index from citation DataFrame.

    Args:
        df: DataFrame with citation pairs
        train_meta: List of train metadata dicts
        test_meta: List of test metadata dicts

    Returns:
        Tuple of:
        - pid_to_text: Array mapping paragraph ID to text
        - text_to_pid: Dict mapping text to paragraph ID
        - paragraph_dates: Array of dates for each paragraph
        - paragraph_celex: Array of CELEX IDs for each paragraph
        - paragraph_set: Array of "train"/"test"/None for each paragraph
    """
    train_celex = {m["case_id"] for m in train_meta}
    test_celex = {m["case_id"] for m in test_meta}

    # Collect all unique paragraph texts
    all_texts = pd.unique(
        pd.concat([df["TEXT_FROM"], df["TEXT_TO"]], ignore_index=True)
    )
    all_texts = [t for t in all_texts if isinstance(t, str)]

    text_to_pid = {t: i for i, t in enumerate(all_texts)}
    pid_to_text = np.array(all_texts, dtype=object)
    n_par = len(pid_to_text)

    # Temporary storage to resolve earliest date + metadata per paragraph
    tmp_info: dict[int, dict] = {
        pid: {
            "date": None,
            "celex": None,
            "number": None,
            "set_type": None,
        }
        for pid in range(n_par)
    }

    # Pass 1: Fill from TEXT_FROM rows
    for (celex_from, number_from), sub in tqdm(
        df.groupby(["CELEX_FROM", "NUMBER_FROM"]), desc="Building index (FROM)"
    ):
        paragraph_text = sub["TEXT_FROM"].iloc[0]
        date_str = sub["DATE_FROM"].iloc[0]
        if not isinstance(paragraph_text, str):
            continue
        pid = text_to_pid[paragraph_text]
        d = dt.strptime(date_str, "%Y-%m-%d")

        info = tmp_info[pid]
        if info["date"] is None or d < info["date"]:
            info["date"] = d
            info["celex"] = celex_from
            info["number"] = number_from
            info["set_type"] = (
                "train"
                if celex_from in train_celex
                else ("test" if celex_from in test_celex else info["set_type"])
            )

    # Pass 2: Fill from TEXT_TO rows
    for (celex_to, number_to), sub in tqdm(
        df.groupby(["CELEX_TO", "NUMBER_TO"]), desc="Building index (TO)"
    ):
        paragraph_text = sub["TEXT_TO"].iloc[0]
        date_str = sub["DATE_TO"].iloc[0]
        if not isinstance(paragraph_text, str):
            continue
        pid = text_to_pid[paragraph_text]
        d = dt.strptime(date_str, "%Y-%m-%d")

        info = tmp_info[pid]
        if info["date"] is None or d < info["date"]:
            info["date"] = d
            info["celex"] = celex_to
            info["number"] = number_to
            if info["set_type"] is None:
                info["set_type"] = (
                    "train"
                    if celex_to in train_celex
                    else ("test" if celex_to in test_celex else None)
                )

    # Finalize arrays
    paragraph_dates = np.array(
        [tmp_info[pid]["date"] for pid in range(n_par)],
        dtype="datetime64[ns]",
    )
    paragraph_celex = np.array(
        [tmp_info[pid]["celex"] for pid in range(n_par)],
        dtype=object,
    )
    paragraph_set = np.array(
        [tmp_info[pid]["set_type"] for pid in range(n_par)],
        dtype=object,
    )

    return pid_to_text, text_to_pid, paragraph_dates, paragraph_celex, paragraph_set


def build_citation_graph(
    df: pd.DataFrame,
    text_to_pid: dict,
) -> dict[int, list[int]]:
    """
    Build citation graph mapping source paragraph to cited paragraphs.

    Args:
        df: DataFrame with citation pairs
        text_to_pid: Mapping from text to paragraph ID

    Returns:
        Dictionary mapping source pid to sorted list of cited pids
    """
    cited_by_pid = defaultdict(set)

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building citations"):
        src_txt = row["TEXT_FROM"]
        tgt_txt = row["TEXT_TO"]
        if not isinstance(src_txt, str) or not isinstance(tgt_txt, str):
            continue
        src_pid = text_to_pid[src_txt]
        tgt_pid = text_to_pid[tgt_txt]
        cited_by_pid[src_pid].add(tgt_pid)

    # Make deterministic and convert to dict
    result: dict[int, list[int]] = {k: sorted(v) for k, v in cited_by_pid.items()}

    return result

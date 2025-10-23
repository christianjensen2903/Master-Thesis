import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
from pathlib import Path
from typing import Any
import json


def load_candidate_documents(
    csv_path: str, cutoff_date: str, use_all_paragraphs: bool = False
) -> pd.DataFrame:
    """Load candidate documents (either all paragraphs or just target paragraphs)."""
    cutoff = pd.Timestamp(cutoff_date)
    df = pd.read_csv(csv_path)
    if use_all_paragraphs:
        return _load_all_paragraphs_before_cutoff(cutoff)
    else:
        return _load_target_paragraphs_before_cutoff(df, cutoff)


def _load_all_paragraphs_before_cutoff(cutoff: pd.Timestamp) -> pd.DataFrame:
    """Load all paragraphs from judgments_cleaned.json before the cutoff date."""
    with open("data/judgments_cleaned.json", "r") as f:
        judgments = json.load(f)

    # Extract all paragraphs with their metadata
    all_paragraphs = []
    for judgment in judgments:
        celex_id = judgment["celex_id"]
        date = judgment.get("meta", {}).get("date", "")
        if date:
            try:
                judgment_date = pd.to_datetime(date)
                if judgment_date < cutoff:
                    for para_num, para_text in judgment["paragraphs"].items():
                        para_id = f"{celex_id}::{para_num}"
                        all_paragraphs.append(
                            {
                                "CELEX": celex_id,
                                "PARA_NO": para_num,
                                "DATE": judgment_date,
                                "TEXT": para_text,
                                "TITLE": judgment.get("meta", {}).get("title", ""),
                                "TO_ID": para_id,
                            }
                        )
            except:
                # Skip judgments with invalid dates
                continue

    docs = pd.DataFrame(all_paragraphs)
    print(f"Loaded {len(docs)} paragraphs from all judgments before cutoff date")
    return docs


def _load_target_paragraphs_before_cutoff(
    df: pd.DataFrame, cutoff: pd.Timestamp
) -> pd.DataFrame:
    """Load only target paragraphs from par-to-par dataset before cutoff date."""
    docs = (
        df[["CELEX_TO", "NUMBER_TO", "DATE_TO", "TEXT_TO", "TITLE_TO", "TO_ID"]]
        .drop_duplicates("TO_ID")
        .copy()
    )
    docs.rename(
        columns={
            "CELEX_TO": "CELEX",
            "NUMBER_TO": "PARA_NO",
            "DATE_TO": "DATE",
            "TEXT_TO": "TEXT",
            "TITLE_TO": "TITLE",
        },
        inplace=True,
    )
    docs["DATE"] = pd.to_datetime(docs["DATE"])
    docs = docs.loc[docs["DATE"] < cutoff].copy()
    return docs


def load_queries(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Load queries (source paragraphs on or after cutoff date)."""
    cutoff = pd.Timestamp(cutoff_date)

    queries = (
        df[
            [
                "CELEX_FROM",
                "NUMBER_FROM",
                "DATE_FROM",
                "TEXT_FROM",
                "TITLE_FROM",
                "FROM_ID",
            ]
        ]
        .drop_duplicates("FROM_ID")
        .copy()
    )
    queries.rename(
        columns={
            "CELEX_FROM": "CELEX",
            "NUMBER_FROM": "PARA_NO",
            "DATE_FROM": "DATE",
            "TEXT_FROM": "TEXT",
            "TITLE_FROM": "TITLE",
            "FROM_ID": "QID",
        },
        inplace=True,
    )
    queries["DATE"] = pd.to_datetime(queries["DATE"])
    queries = queries.loc[queries["DATE"] >= cutoff].copy()

    # Create PyTerrier queries dataframe with required fields
    queries_df = pd.DataFrame(
        {
            "qid": queries["QID"],
            "query": queries["TEXT"],
        }
    )
    return queries_df


def load_relevance_judgments(df: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Load relevance judgments (qrels) for queries after cutoff and documents before cutoff."""
    cutoff = pd.Timestamp(cutoff_date)

    # Filter to include only queries after cutoff and documents before cutoff
    df_filtered = df.loc[df["DATE_FROM"] >= cutoff].copy()

    qrels_df = pd.DataFrame(
        {
            "qid": df_filtered["FROM_ID"],
            "docno": df_filtered["TO_ID"],
            "label": 1,  # Binary relevance (1 = relevant)
        }
    )

    # Remove duplicate qid-docno pairs (if any)
    qrels_df = qrels_df.drop_duplicates(subset=["qid", "docno"])
    return qrels_df


def load_and_prepare_data(
    csv_path: str, cutoff_date: str, use_all_paragraphs: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and prepare data for evaluation using the specified mode."""
    # Load and preprocess the main dataset
    df = pd.read_csv(csv_path)
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    # Load candidate documents
    docs = load_candidate_documents(df, cutoff_date, use_all_paragraphs)

    # Create PyTerrier documents dataframe with required fields
    documents_df = pd.DataFrame(
        {
            "docno": docs["TO_ID"],
            "text": docs["TEXT"],
        }
    )

    # Load queries
    queries_df = load_queries(df, cutoff_date)

    # Load relevance judgments
    qrels_df = load_relevance_judgments(df, cutoff_date)

    print(f"Documents: {len(documents_df)}")
    print(f"Queries: {len(queries_df)}")
    print(f"Qrels (relevance judgments): {len(qrels_df)}")
    print(f"Unique queries with judgments: {qrels_df['qid'].nunique()}")

    return documents_df, queries_df, qrels_df

import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
from pathlib import Path
from typing import Any


def load_and_prepare_data(
    csv_path: str, cutoff_date: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    cutoff = pd.Timestamp(cutoff_date)

    # Build candidate pool (documents) - targets before cutoff date
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

    # Create PyTerrier documents dataframe with required fields
    documents_df = pd.DataFrame(
        {
            "docno": docs["TO_ID"],
            "text": docs["TEXT"],
        }
    )

    # Build queries - sources on or after cutoff date
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

    # Build qrels (relevance judgments)
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

    print(f"Documents: {len(documents_df)}")
    print(f"Queries: {len(queries_df)}")
    print(f"Qrels (relevance judgments): {len(qrels_df)}")
    print(f"Unique queries with judgments: {qrels_df['qid'].nunique()}")

    return documents_df, queries_df, qrels_df

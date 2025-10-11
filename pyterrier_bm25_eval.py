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


def build_index(documents_df: pd.DataFrame, index_path: str) -> Any:
    # Convert to absolute path and create directory if it doesn't exist
    index_path = str(Path(index_path).resolve())
    Path(index_path).mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding index at: {index_path}")

    # Filter out documents with empty text
    documents_df = documents_df[
        documents_df["text"].notna() & (documents_df["text"].str.strip() != "")
    ].copy()
    print(f"Indexing {len(documents_df)} documents (after filtering empty text)")

    indexer = pt.terrier.IterDictIndexer(
        index_path,
        overwrite=True,
        meta={"docno": 100},
        meta_reverse=["docno"],
        tokeniser="utf",
    )

    index_ref = indexer.index(documents_df.to_dict(orient="records"))

    print(f"Index built successfully")

    return index_ref


def evaluate_bm25(
    index_ref: Any,
    queries_df: pd.DataFrame,
    qrels_df: pd.DataFrame,
    k_values: list[int] = [],
) -> pd.DataFrame:

    bm25 = pt.rewrite.tokenise("utf") >> pt.terrier.Retriever(
        index_ref, wmodel="BM25", verbose=True
    )

    metrics = ["map"]
    for k in k_values:
        metrics.extend([f"P_{k}", f"recall_{k}"])

    results = pt.Experiment(
        [bm25],
        queries_df,
        qrels_df,
        metrics,
        names=["BM25"],
    )

    return results


def main() -> None:

    csv_path = "data/clean_data.csv"
    index_path = "artifacts/pyterrier_index"
    cutoff_date = "2018-01-01"
    k_values = [5, 10, 50, 100]

    print("=" * 80)
    print("PyTerrier BM25 Evaluation")
    print("=" * 80)

    # Step 1: Load and prepare data
    print("\n[1/3] Loading and preparing data...")
    documents_df, queries_df, qrels_df = load_and_prepare_data(csv_path, cutoff_date)

    # Step 2: Build index
    print("\n[2/3] Building PyTerrier index...")
    index_ref = build_index(documents_df, index_path)

    # Step 3: Evaluate BM25
    print("\n[3/3] Evaluating BM25...")
    results = evaluate_bm25(index_ref, queries_df, qrels_df, k_values)

    # Display results
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(results.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()

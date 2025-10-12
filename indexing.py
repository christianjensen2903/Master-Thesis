import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
from pathlib import Path
from typing import Any
from utils import load_and_prepare_data


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


if __name__ == "__main__":
    documents_df, queries_df, qrels_df = load_and_prepare_data(
        "data/clean_data.csv", "2018-01-01"
    )
    build_index(documents_df, "artifacts/index")

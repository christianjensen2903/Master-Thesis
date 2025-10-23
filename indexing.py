import pandas as pd  # type: ignore
import pyterrier as pt  # type: ignore
from pathlib import Path
from typing import Any
from utils import load_candidate_documents


def build_index(index_path: str) -> Any:
    # Convert to absolute path and create directory if it doesn't exist
    index_path = str(Path(index_path).resolve())
    Path(index_path).mkdir(parents=True, exist_ok=True)

    print(f"\nBuilding index at: {index_path}")
    print(f"Indexing {len(docs)} documents")

    indexer = pt.terrier.IterDictIndexer(
        index_path,
        overwrite=True,
        meta={"docno": 100},
        meta_reverse=["docno"],
        tokeniser="utf",
    )

    index_ref = indexer.index([doc.model_dump() for doc in docs])

    print(f"Index built successfully")

    return index_ref


if __name__ == "__main__":
    docs = load_candidate_documents("2018-01-01", use_all_paragraphs=False)
    build_index("artifacts/index")

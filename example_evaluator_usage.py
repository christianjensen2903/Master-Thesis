import pandas as pd  # type: ignore
from langchain_core.documents import Document
from retrievers import (
    TFIDFRetriever,
    BM25Retriever,
    preprocess_utils,
    SentenceBERTRetriever,
)
from nltk.corpus import stopwords  # type: ignore
import nltk  # type: ignore

from evaluator import Evaluator

nltk.download("stopwords", quiet=True)


def build_candidate_pool(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> list[Document]:
    """Build the candidate pool of unique target paragraphs strictly before a cutoff date."""
    cands = (
        df[["CELEX_TO", "NUMBER_TO", "DATE_TO", "TEXT_TO", "TITLE_TO", "TO_ID"]]
        .drop_duplicates("TO_ID")
        .copy()
    )
    cands.rename(
        columns={
            "CELEX_TO": "CELEX",
            "NUMBER_TO": "PARA_NO",
            "DATE_TO": "DATE",
            "TEXT_TO": "TEXT",
            "TITLE_TO": "TITLE",
        },
        inplace=True,
    )
    cands["DATE"] = pd.to_datetime(cands["DATE"])
    cands = cands.loc[cands["DATE"] < cutoff_date].copy()
    return [
        Document(
            page_content=row["TEXT"],
            metadata={"id": row["TO_ID"], "celex": row["CELEX"], "date": row["DATE"]},
        )
        for _, row in cands.reset_index(drop=True).iterrows()
    ]


def build_queries(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> pd.DataFrame:
    """Build the unique query set from FROM-side paragraphs."""
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
    queries = queries.loc[queries["DATE"] >= cutoff_date].copy()
    return queries.reset_index(drop=True)


def build_rel_map(df: pd.DataFrame, cutoff_date: pd.Timestamp) -> dict[str, set[str]]:
    """Ground truth mapping from query id to the set of relevant target ids.

    Only includes targets with dates strictly before the cutoff date.
    """
    df_filtered = df.loc[df["DATE_TO"] < cutoff_date].copy()
    return (
        df_filtered.groupby("FROM_ID")["TO_ID"]
        .apply(lambda s: set(s.astype(str)))
        .to_dict()
    )


def example_single_model() -> None:
    """Example: Evaluate a single retriever model."""
    print("=" * 80)
    print("EXAMPLE 1: Single Model Evaluation")
    print("=" * 80)

    cutoff_date = pd.Timestamp("2018-01-01")
    k_list = [5, 10, 50, 100]

    # Load data
    df = pd.read_csv("data/clean_data.csv")
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

    # Build evaluation data
    cands = build_candidate_pool(df, cutoff_date=cutoff_date)
    queries = build_queries(df, cutoff_date=cutoff_date)
    rel_map = build_rel_map(df, cutoff_date=cutoff_date)

    print(f"Candidates: {len(cands)}, Queries: {len(queries)}")

    # Create and configure a retriever
    retriever = BM25Retriever(
        documents=cands,
        preprocess=preprocess_utils.compose(
            preprocess_utils.lowercase(),
            preprocess_utils.remove_punctuation(),
            preprocess_utils.stopword_filter(
                stopwords=set(
                    stopwords.words("english")
                    + [
                        "<DATE>",
                        "<QUOTED_TEXT>",
                        "<ECLI>",
                        "<ECR>",
                        "<PARAGRAPH>",
                        "<CASE>",
                    ]
                ),
            ),
        ),
    )

    # Initialize evaluator
    evaluator = Evaluator(k_values=k_list, show_progress=True)

    # Evaluate
    summary = evaluator.evaluate(
        retriever=retriever, queries=queries, relevance_map=rel_map
    )

    print("\n=== BM25 Summary Results ===")
    print(summary.to_string(index=False))


def main() -> None:
    """Run all examples."""
    # Example 1: Single model evaluation
    example_single_model()


if __name__ == "__main__":
    main()

import pandas as pd  # type: ignore
from sentence_transformers import InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split  # type: ignore


def split_data_by_date(
    df: pd.DataFrame,
    cutoff_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
    cutoff_date = pd.Timestamp(year=cutoff_year, month=1, day=1)

    train_df = df[df["DATE_FROM"] < cutoff_date].copy()
    val_df = df[df["DATE_FROM"] >= cutoff_date].copy()

    print(f"\n📅 Temporal Split:")
    print(f"  Train: before {cutoff_date.date()} ({len(train_df)} citations)")
    print(f"  Val: after {cutoff_date.date()} ({len(val_df)} citations)")

    return train_df, val_df


def create_ir_evaluator(val_df: pd.DataFrame) -> InformationRetrievalEvaluator:
    """
    Create InformationRetrievalEvaluator for validation.

    Args:
        val_df: Validation DataFrame with TEXT_FROM, TEXT_TO columns
        model_name: Base model name for evaluation

    Returns:
        InformationRetrievalEvaluator instance
    """
    # Prepare queries and documents for IR evaluation
    queries = {}
    documents = {}
    relevant_docs: dict[str, set[str]] = {}

    # Create unique IDs for queries and documents
    query_to_id = {}
    doc_to_id = {}

    for _, row in val_df.iterrows():
        text_from = str(row["TEXT_FROM"])
        text_to = str(row["TEXT_TO"])

        # Get or create query ID
        if text_from not in query_to_id:
            query_to_id[text_from] = text_from
            queries[text_from] = text_from
            relevant_docs[text_from] = set()

        # Get or create document ID
        if text_to not in doc_to_id:
            doc_to_id[text_to] = text_to
            documents[text_to] = text_to

        relevant_docs[text_from].add(text_to)

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=documents,
        relevant_docs=relevant_docs,
        name="validation_ir",
        show_progress_bar=True,
        map_at_k=[1000],
        precision_recall_at_k=[5, 10, 50, 100],
        mrr_at_k=[],
        ndcg_at_k=[],
    )

    return evaluator

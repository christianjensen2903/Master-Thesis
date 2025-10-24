import pandas as pd  # type: ignore
from sentence_transformers import InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split  # type: ignore


def split_data_by_date(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, validation_split: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets based on date cutoff.

    Args:
        df: DataFrame with DATE_FROM column
        cutoff_date: Date to split on
        validation_split: Fraction of data to use for validation

    Returns:
        Tuple of (train_df, val_df)
    """
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])

    # First split by date
    train_df = df[df["DATE_FROM"] < cutoff_date].copy()

    # Then split train into train/val
    if len(train_df) > 0:
        train_df, val_df = train_test_split(
            train_df,
            test_size=validation_split,
            random_state=42,
            stratify=None,  # No stratification for now
        )
    else:
        val_df = pd.DataFrame(columns=df.columns)

    return train_df, val_df


def create_ir_evaluator(
    val_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2"
) -> InformationRetrievalEvaluator:
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
    )

    return evaluator


def create_validation_examples(val_df: pd.DataFrame) -> list[InputExample]:
    """
    Create InputExamples for validation set.

    Args:
        val_df: Validation DataFrame with TEXT_FROM, TEXT_TO columns

    Returns:
        List of InputExamples for validation
    """
    val_examples = []
    for _, row in val_df.iterrows():
        text_from = str(row["TEXT_FROM"])
        text_to = str(row["TEXT_TO"])
        val_examples.append(InputExample(texts=[text_from, text_to]))

    return val_examples

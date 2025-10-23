import pandas as pd
import numpy as np
from typing import Tuple, List
from sentence_transformers import InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split


def split_data_by_date(
    df: pd.DataFrame, cutoff_date: pd.Timestamp, validation_split: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    relevant_docs = {}

    # Create unique IDs for queries and documents
    query_id = 0
    doc_id = 0
    query_to_id = {}
    doc_to_id = {}

    for _, row in val_df.iterrows():
        text_from = str(row["TEXT_FROM"])
        text_to = str(row["TEXT_TO"])

        # Get or create query ID
        if text_from not in query_to_id:
            query_to_id[text_from] = query_id
            queries[query_id] = text_from
            relevant_docs[query_id] = set()
            query_id += 1

        # Get or create document ID
        if text_to not in doc_to_id:
            doc_to_id[text_to] = doc_id
            documents[doc_id] = text_to
            doc_id += 1

        # Add relevant document
        relevant_docs[query_to_id[text_from]].add(doc_to_id[text_to])

    # Convert sets to lists for evaluator
    relevant_docs = {k: list(v) for k, v in relevant_docs.items()}

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        documents=documents,
        relevant_docs=relevant_docs,
        name="validation_ir",
        show_progress_bar=True,
    )

    return evaluator


def create_validation_examples(val_df: pd.DataFrame) -> List[InputExample]:
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

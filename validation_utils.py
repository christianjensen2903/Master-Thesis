import pandas as pd  # type: ignore
from sentence_transformers import InputExample
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sklearn.model_selection import train_test_split  # type: ignore


def split_data_by_date(
    df: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    validation_split: float = 0.1,
    use_temporal_validation: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and validation sets based on date cutoff.

    Args:
        df: DataFrame with DATE_FROM column
        cutoff_date: Test set cutoff date (e.g., 2018-01-01)
        validation_split: Fraction of pre-cutoff period to use for validation (only used if use_temporal_validation=False)
        use_temporal_validation: If True, use temporal split (train=pre-val_date, val=val_date to cutoff_date).
                                If False, use random split of pre-cutoff data.

    Returns:
        Tuple of (train_df, val_df)
    """
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])

    if use_temporal_validation:
        # Temporal split: validation period is 1 year before test cutoff
        # E.g., if cutoff_date is 2018-01-01:
        #   - Train: pre-2017
        #   - Val: 2017-2018
        #   - Test: post-2018 (not returned here, handled by caller)
        val_start_date = pd.Timestamp(
            year=cutoff_date.year - 1, month=cutoff_date.month, day=cutoff_date.day
        )

        train_df = df[df["DATE_FROM"] < val_start_date].copy()
        val_df = df[
            (df["DATE_FROM"] >= val_start_date) & (df["DATE_FROM"] < cutoff_date)
        ].copy()

        print(f"\n📅 Temporal Split:")
        print(f"  Train: before {val_start_date.date()} ({len(train_df)} citations)")
        print(
            f"  Val: {val_start_date.date()} to {cutoff_date.date()} ({len(val_df)} citations)"
        )
        print(f"  Test: after {cutoff_date.date()} (handled separately)")

    else:
        # Random split (original behavior)
        train_df = df[df["DATE_FROM"] < cutoff_date].copy()

        if len(train_df) > 0:
            train_df, val_df = train_test_split(
                train_df,
                test_size=validation_split,
                random_state=42,
                stratify=None,
            )
        else:
            val_df = pd.DataFrame(columns=df.columns)

        print(f"\n📊 Random Split:")
        print(f"  Train: {len(train_df)} citations (random {1-validation_split:.0%})")
        print(f"  Val: {len(val_df)} citations (random {validation_split:.0%})")

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

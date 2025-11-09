import os
import pandas as pd  # type: ignore
from abc import ABC, abstractmethod
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from validation_utils import split_data_by_date


class BaseDenseRetrieverTrainer(ABC):
    """Base class for dense retriever trainers with shared functionality."""

    def __init__(
        self,
        model_name: str,
        training_args: SentenceTransformerTrainingArguments,
        validation_split: float = 0.1,
        use_wandb: bool = True,
        max_seq_length: int | None = None,
    ):
        self.model_name = model_name
        self.training_args = training_args
        self.validation_split = validation_split
        self.use_wandb = use_wandb
        self.max_seq_length = max_seq_length

        if self.training_args is not None:
            output_dir_init = self.training_args.output_dir
            if output_dir_init is not None:
                os.makedirs(output_dir_init, exist_ok=True)
        else:
            os.makedirs("output/model", exist_ok=True)

    def load_and_split_data(
        self, paragraph_file: str, cutoff_year: int
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and split into train/validation sets."""
        df = pd.read_csv(paragraph_file)
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])
        df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])

        df["FROM_ID"] = (
            df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
        )
        df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)

        return split_data_by_date(df, cutoff_year)

    def create_ir_evaluator(
        self, val_df: pd.DataFrame, train_df: pd.DataFrame
    ) -> InformationRetrievalEvaluator:
        """Create an Information Retrieval evaluator for validation."""
        queries = {}
        documents = {}
        relevant_docs: dict[str, set[str]] = {}

        for _, row in train_df.iterrows():
            text_from = str(row["TEXT_FROM"])
            text_to = str(row["TEXT_TO"])
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])

            documents[from_id] = text_from
            documents[to_id] = text_to

        for _, row in val_df.iterrows():
            text_from = str(row["TEXT_FROM"])
            text_to = str(row["TEXT_TO"])
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])

            if to_id not in documents:
                continue

            if from_id not in queries:
                queries[from_id] = text_from
                relevant_docs[from_id] = set()

            relevant_docs[from_id].add(to_id)

        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=documents,
            relevant_docs=relevant_docs,
            name="validation_ir",
            show_progress_bar=True,
            map_at_k=[1000],
            precision_recall_at_k=[5, 10, 50, 100],
        )

        return evaluator

    @abstractmethod
    def train(self, paragraph_file: str, cutoff_year: int) -> SentenceTransformer:
        """Train the model. Must be implemented by subclasses."""
        pass

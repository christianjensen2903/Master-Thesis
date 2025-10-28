import os
import pandas as pd  # type: ignore
import wandb
from sentence_transformers import SentenceTransformer
from typing import Any
from abc import ABC, abstractmethod
from validation_utils import split_data_by_date
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers import SentenceTransformer
import torch


class BaseTrainer(ABC):
    """Base class for training different types of models."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        output_path: str = "output/model",
        batch_size: int = 16,
        epochs: int = 5,
        warmup_steps: int = 100,
        checkpoint_save_steps: int = 1000,
        evaluation_steps: int = 1000,
        eval_every_n_epochs: int | None = None,
        show_progress_bar: bool = True,
        validation_split: float = 0.1,
        use_wandb: bool = True,
        project_name: str = "training-project",
    ):
        self.model_name = model_name
        self.output_path = output_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.warmup_steps = warmup_steps
        self.checkpoint_save_steps = checkpoint_save_steps
        self.evaluation_steps = evaluation_steps
        self.eval_every_n_epochs = eval_every_n_epochs
        self.show_progress_bar = show_progress_bar
        self.validation_split = validation_split
        self.use_wandb = use_wandb
        self.project_name = project_name

        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

    def load_and_split_data(
        self, paragraph_file: str, cutoff_date: pd.Timestamp
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load data and split into train/validation sets."""
        df = pd.read_csv(paragraph_file)
        df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])

        # Add DATE_TO if it exists
        if "DATE_TO" in df.columns:
            df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])

        # Add ID columns if they don't exist
        if "FROM_ID" not in df.columns and "CELEX_FROM" in df.columns:
            df["FROM_ID"] = (
                df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
            )
        if "TO_ID" not in df.columns and "CELEX_TO" in df.columns:
            df["TO_ID"] = (
                df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)
            )

        return split_data_by_date(df, cutoff_date, self.validation_split)

    @abstractmethod
    def train(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
    ) -> torch.nn.Module:
        """
        Abstract method for training models.

        Args:
            paragraph_file: Path to the paragraph pairs CSV file
            cutoff_date: Date cutoff for training data
        """
        pass

    def setup_wandb(self, config: dict[str, Any]) -> None:
        """Initialize wandb with the given config."""
        if self.use_wandb:
            wandb.init(project=self.project_name, config=config)

    def cleanup_wandb(self) -> None:
        """Save model to wandb and finish the run."""
        if self.use_wandb:
            wandb.save(f"{self.output_path}/*")
            wandb.finish()

    def create_ir_evaluator(
        self, val_df: pd.DataFrame
    ) -> InformationRetrievalEvaluator:

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
        )

        return evaluator

    def _train_model(
        self,
        train_dataloader: Any,
        train_loss: Any,
        evaluator: Any | None = None,
        description: str = "Training model",
    ) -> SentenceTransformer:
        """Train the model with the given dataloader and loss."""
        model = SentenceTransformer(self.model_name)

        print(f"\n{description}...")
        print(f"Total batches: {len(train_dataloader)}")

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=self.epochs,
            warmup_steps=self.warmup_steps,
            output_path=self.output_path,
            scheduler="WarmupLinear",
            show_progress_bar=self.show_progress_bar,
            evaluator=evaluator,
            evaluation_steps=self.evaluation_steps,
            checkpoint_save_steps=self.checkpoint_save_steps,
            save_best_model=True,
        )
        model.save(self.output_path)

        print(f"Training finished. Model saved to {self.output_path}")
        return model

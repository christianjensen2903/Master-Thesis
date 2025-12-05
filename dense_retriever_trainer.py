import json
import os
from typing import Any

import pandas as pd  # type: ignore
from datasets import Dataset  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from tqdm import tqdm  # type: ignore

from validation_utils import split_data_by_date


class DenseRetrieverTrainer:

    def __init__(
        self,
        model_name: str,
        training_args: SentenceTransformerTrainingArguments,
        validation_split: float = 0.1,
        use_wandb: bool = True,
        max_seq_length: int | None = None,
        loss_scale: float = 20.0,
        use_prefixes: bool = True,
        include_metadata: bool = False,
        judgments_path: str | None = None,
    ):
        self.model_name = model_name
        self.training_args = training_args
        self.validation_split = validation_split
        self.use_wandb = use_wandb
        self.max_seq_length = max_seq_length
        self.loss_scale = loss_scale
        self.use_prefixes = use_prefixes
        self.include_metadata = include_metadata
        self.judgments_path = judgments_path

        # Load judgment metadata if needed
        self.judgment_metadata: dict[str, dict[str, Any]] = {}
        if include_metadata:
            if judgments_path is None:
                raise ValueError(
                    "judgments_path must be provided when include_metadata=True"
                )
            self._load_judgment_metadata(judgments_path)

        if self.training_args is not None:
            output_dir_init = self.training_args.output_dir
            if output_dir_init is not None:
                os.makedirs(output_dir_init, exist_ok=True)
        else:
            os.makedirs("output/model", exist_ok=True)

    def _load_judgment_metadata(self, judgments_path: str) -> None:
        """Load judgment metadata from JSON file."""
        print(f"Loading judgment metadata from {judgments_path}...")
        with open(judgments_path) as f:
            judgments = json.load(f)
        for celex, judgment in judgments.items():
            self.judgment_metadata[celex] = judgment.get("meta", {})
        print(f"Loaded metadata for {len(self.judgment_metadata)} judgments")

    def _format_list(self, items: list[str] | None) -> str:
        """Format a list of strings by joining with '. '."""
        if not items:
            return ""
        return ". ".join(str(item) for item in items if item)

    def _format_case_law_about(self, case_law_about: dict[str, Any] | None) -> str:
        """Format case law about dict by extracting values and joining with '. '."""
        if not case_law_about:
            return ""
        values = [str(v) for v in case_law_about.values() if v]
        return ". ".join(values)

    def _get_metadata_text(self, celex: str) -> str:
        """Get formatted metadata text for a given CELEX number."""
        meta = self.judgment_metadata.get(celex, {})
        lines = []

        subject_matter = self._format_list(meta.get("subject_matter"))
        if subject_matter:
            lines.append(f"Subject: {subject_matter}")

        keywords = self._format_list(meta.get("keywords"))
        if keywords:
            lines.append(f"Keywords: {keywords}")

        case_law_about = self._format_case_law_about(meta.get("case_law_about"))
        if case_law_about:
            lines.append(f"About: {case_law_about}")

        return "\n\n".join(lines)

    def _format_text(self, text: str, celex: str, is_query: bool) -> str:
        """Format text with optional prefix and metadata.

        Format:
            Subject: ...

            Keywords: ...

            About: ...


            <text>
        """
        # Add prefix if enabled
        if self.use_prefixes:
            prefix = "query: " if is_query else "passage: "
            text = f"{prefix}{text}"

        # Add metadata before text if enabled
        if self.include_metadata:
            metadata_text = self._get_metadata_text(celex)
            if metadata_text:
                text = f"{metadata_text}\n\n\n{text}"

        return text

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
            text_to = str(row["TEXT_TO"])
            to_id = str(row["TO_ID"])
            celex_to = str(row["CELEX_TO"])
            documents[to_id] = self._format_text(text_to, celex_to, is_query=False)

        for _, row in val_df.iterrows():
            text_from = str(row["TEXT_FROM"])
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])
            celex_from = str(row["CELEX_FROM"])

            if to_id not in documents:
                continue

            if from_id not in queries:
                queries[from_id] = self._format_text(
                    text_from, celex_from, is_query=True
                )
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

    def get_simcse_data(
        self, paragraph_file: str, cutoff_year: int
    ) -> tuple[Dataset, pd.DataFrame, pd.DataFrame]:
        """Get data for sentence pair training."""
        train_df, val_df = self.load_and_split_data(paragraph_file, cutoff_year)

        train_data = []
        for _, row in tqdm(
            train_df.iterrows(),
            total=len(train_df),
            desc="Creating training dataset",
        ):
            text_from = str(row["TEXT_FROM"])
            text_to = str(row["TEXT_TO"])
            celex_from = str(row["CELEX_FROM"])
            celex_to = str(row["CELEX_TO"])

            sentence1 = self._format_text(text_from, celex_from, is_query=True)
            sentence2 = self._format_text(text_to, celex_to, is_query=False)

            train_data.append({"sentence1": sentence1, "sentence2": sentence2})

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, val_df, train_df

    def train(
        self,
        paragraph_file: str,
        cutoff_year: int,
        resume_from_checkpoint: str | bool | None = None,
    ) -> SentenceTransformer:
        """Train a sentence embedding model using SIMCSE with MultipleNegativesRankingLoss.

        Args:
            paragraph_file: Path to the CSV file with paragraph data.
            cutoff_year: Year to split train/validation data.
            resume_from_checkpoint: Path to checkpoint directory to resume from,
                True to resume from latest checkpoint in output_dir, or None to start fresh.
        """
        train_dataset, val_df, train_df = self.get_simcse_data(
            paragraph_file, cutoff_year
        )

        # Load model from checkpoint if resuming, otherwise from base model
        if isinstance(resume_from_checkpoint, str) and os.path.isdir(
            resume_from_checkpoint
        ):
            print(f"Resuming from checkpoint: {resume_from_checkpoint}")
            model = SentenceTransformer(resume_from_checkpoint)
        elif resume_from_checkpoint is True:
            # Find latest checkpoint in output_dir
            checkpoint = self._get_latest_checkpoint()
            if checkpoint:
                print(f"Resuming from latest checkpoint: {checkpoint}")
                model = SentenceTransformer(checkpoint)
                resume_from_checkpoint = checkpoint
            else:
                print("No checkpoint found, starting fresh")
                model = SentenceTransformer(self.model_name)
                resume_from_checkpoint = None
        else:
            model = SentenceTransformer(self.model_name)

        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length

        train_loss = losses.MultipleNegativesRankingLoss(
            model=model, scale=self.loss_scale
        )
        evaluator = self.create_ir_evaluator(val_df, train_df)

        print(f"\nTraining {self.model_name} with MultipleNegativesRankingLoss...")
        print(f"Total training examples: {len(train_dataset)}")

        trainer = SentenceTransformerTrainer(
            model=model,
            train_dataset=train_dataset,
            loss=train_loss,
            evaluator=evaluator,
            args=self.training_args,
        )

        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        output_dir = self.training_args.output_dir
        if output_dir is None:
            raise ValueError("output_dir must be set in training_args")
        model.save(output_dir)

        print(f"Training finished. Model saved to {output_dir}")
        return model

    def _get_latest_checkpoint(self) -> str | None:
        """Find the latest checkpoint in the output directory."""
        output_dir = self.training_args.output_dir
        if output_dir is None or not os.path.isdir(output_dir):
            return None

        checkpoints = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-")
            and os.path.isdir(os.path.join(output_dir, d))
        ]
        if not checkpoints:
            return None

        # Sort by step number and return the latest
        checkpoints.sort(key=lambda x: int(x.split("-")[1]))
        return os.path.join(output_dir, checkpoints[-1])


if __name__ == "__main__":
    trainer = DenseRetrieverTrainer(
        model_name="all-MiniLM-L6-v2",
        training_args=SentenceTransformerTrainingArguments(
            output_dir="checkpoints/dense_retriever",
            num_train_epochs=10,
            weight_decay=0.01,
            bf16=False,
            fp16=False,
            gradient_accumulation_steps=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            metric_for_best_model="validation_ir_map@1000",
            gradient_checkpointing=True,
            warmup_ratio=0.1,
            learning_rate=2e-5,
            report_to="wandb",
            eval_strategy="steps",
            eval_steps=1000,
            save_steps=1000,
            save_total_limit=10,
            lr_scheduler_type="cosine",
            save_strategy="best",
            batch_sampler=BatchSamplers.NO_DUPLICATES,
        ),
        use_prefixes=True,  # Add "query:" and "passage:" prefixes
        include_metadata=False,  # Add subject matter, keywords, case law about
        judgments_path="data/judgments_cleaned.json",  # Required if include_metadata=True
    )
    trainer.train(paragraph_file="data/par-to-par-cleaned.csv", cutoff_year=2018)

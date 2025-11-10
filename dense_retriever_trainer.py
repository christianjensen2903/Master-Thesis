import os
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from datasets import Dataset  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
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
    ):
        self.model_name = model_name
        self.training_args = training_args
        self.validation_split = validation_split
        self.use_wandb = use_wandb
        self.max_seq_length = max_seq_length
        self.loss_scale = loss_scale

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
            train_data.append({"sentence1": text_from, "sentence2": text_to})

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, val_df, train_df

    def train(self, paragraph_file: str, cutoff_year: int) -> SentenceTransformer:
        """Train a sentence embedding model using SIMCSE with MultipleNegativesRankingLoss."""

        train_dataset, val_df, train_df = self.get_simcse_data(
            paragraph_file, cutoff_year
        )

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

        trainer.train()
        output_dir = self.training_args.output_dir
        if output_dir is None:
            raise ValueError("output_dir must be set in training_args")
        model.save(output_dir)

        print(f"Training finished. Model saved to {output_dir}")
        return model


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
    )
    trainer.train(paragraph_file="data/par-to-par-cleaned.csv", cutoff_year=2018)

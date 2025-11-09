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

from dense_retriever_trainers.base_trainer import BaseDenseRetrieverTrainer


class InBatchNegativeDenseRetrieverTrainer(BaseDenseRetrieverTrainer):

    def __init__(
        self,
        model_name: str,
        training_args: SentenceTransformerTrainingArguments,
        validation_split: float = 0.1,
        use_wandb: bool = True,
        max_seq_length: int | None = None,
        loss_scale: float = 20.0,
    ):
        super().__init__(
            model_name, training_args, validation_split, use_wandb, max_seq_length
        )
        self.loss_scale = loss_scale

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
    trainer = InBatchNegativeDenseRetrieverTrainer(
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

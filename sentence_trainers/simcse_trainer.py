import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from .base_trainer import BaseTrainer
from validation_utils import create_validation_examples


class SimCSETrainer(BaseTrainer):
    """Trainer for SimCSE models using Multiple Negatives Ranking Loss."""

    def get_simcse_data(
        self, paragraph_file: str, cutoff_date: pd.Timestamp
    ) -> tuple[list[InputExample], list[InputExample], pd.DataFrame]:
        """Get data for SimCSE training."""
        train_df, val_df = self.load_and_split_data(paragraph_file, cutoff_date)

        train_examples = []
        for _, row in tqdm(
            train_df.iterrows(),
            total=len(train_df),
            desc="Creating training InputExamples",
        ):
            text_from = row["TEXT_FROM"]
            text_to = row["TEXT_TO"]
            train_examples.append(InputExample(texts=[text_from, text_to]))

        val_examples = create_validation_examples(val_df)

        return train_examples, val_examples, val_df

    def train(
        self,
        paragraph_file: str,
        cutoff_date: pd.Timestamp,
    ) -> SentenceTransformer:
        """Train a SimCSE model."""
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "validation_split": self.validation_split,
            "cutoff_date": str(cutoff_date),
        }

        self.setup_wandb(config)

        train_dataset, val_dataset, val_df = self.get_simcse_data(
            paragraph_file, cutoff_date
        )
        train_dataloader = NoDuplicatesDataLoader(
            train_dataset, batch_size=self.batch_size
        )

        model = SentenceTransformer(self.model_name)
        train_loss = losses.MultipleNegativesRankingLoss(model=model)

        evaluator = self.create_ir_evaluator(val_df)

        print(f"\nTraining {self.model_name} with Supervised SimCSE (MNRL)...")
        print(f"Total training examples: {len(train_dataset)}")
        print(f"Total validation examples: {len(val_dataset)}")

        trained_model = self._train_model(
            train_dataloader, train_loss, evaluator, "Training SimCSE model"
        )

        self.cleanup_wandb()
        return trained_model

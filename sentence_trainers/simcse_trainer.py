import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from typing import Any
from sentence_transformers import SentenceTransformer, losses, InputExample
from sentence_transformers.datasets import NoDuplicatesDataLoader
from .base_trainer import BaseTrainer
from validation_utils import create_validation_examples


class SentencePairTrainer(BaseTrainer):
    """Trainer for sentence embedding models with configurable loss functions using sentence pairs."""

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
        gradient_accumulation_steps: int = 1,
        use_wandb: bool = True,
        project_name: str = "training-project",
        max_seq_length: int | None = None,
        loss_type: str = "MultipleNegativesRankingLoss",
        loss_scale: float = 20.0,
        loss_mini_batch_size: int | None = None,
        guide_model_name: str | None = None,
        gist_temperature: float = 0.01,
        gist_margin_strategy: str = "absolute",
        gist_margin: float = 0.0,
        gist_contrast_anchors: bool = True,
        gist_contrast_positives: bool = True,
    ):
        super().__init__(
            model_name=model_name,
            output_path=output_path,
            batch_size=batch_size,
            epochs=epochs,
            warmup_steps=warmup_steps,
            checkpoint_save_steps=checkpoint_save_steps,
            evaluation_steps=evaluation_steps,
            eval_every_n_epochs=eval_every_n_epochs,
            show_progress_bar=show_progress_bar,
            validation_split=validation_split,
            gradient_accumulation_steps=gradient_accumulation_steps,
            use_wandb=use_wandb,
            project_name=project_name,
            max_seq_length=max_seq_length,
        )
        self.loss_type = loss_type
        self.loss_scale = loss_scale
        self.loss_mini_batch_size = loss_mini_batch_size
        self.guide_model_name = guide_model_name
        self.gist_temperature = gist_temperature
        self.gist_margin_strategy = gist_margin_strategy
        self.gist_margin = gist_margin
        self.gist_contrast_anchors = gist_contrast_anchors
        self.gist_contrast_positives = gist_contrast_positives

    def _create_loss(self, model: SentenceTransformer) -> Any:
        """Create the loss function based on loss_type."""
        loss_map = {
            "MultipleNegativesRankingLoss": losses.MultipleNegativesRankingLoss,
            "CachedMultipleNegativesRankingLoss": losses.CachedMultipleNegativesRankingLoss,
            "MultipleNegativesSymmetricRankingLoss": losses.MultipleNegativesSymmetricRankingLoss,
            "CachedMultipleNegativesSymmetricRankingLoss": losses.CachedMultipleNegativesSymmetricRankingLoss,
            "GISTEmbedLoss": losses.GISTEmbedLoss,
            "CachedGISTEmbedLoss": losses.CachedGISTEmbedLoss,
        }

        if self.loss_type not in loss_map:
            raise ValueError(
                f"Unknown loss_type: {self.loss_type}. "
                f"Available options: {list(loss_map.keys())}"
            )

        loss_class = loss_map[self.loss_type]
        is_cached = "Cached" in self.loss_type
        is_gist = "GIST" in self.loss_type

        if is_gist:
            if self.guide_model_name is None:
                raise ValueError(
                    f"{self.loss_type} requires a guide_model_name. "
                    "Please provide guide_model_name parameter."
                )
            guide_model = SentenceTransformer(self.guide_model_name)
            if is_cached:
                if self.loss_mini_batch_size is None:
                    self.loss_mini_batch_size = 32
                return loss_class(
                    model=model,
                    guide=guide_model,
                    temperature=self.gist_temperature,
                    margin_strategy=self.gist_margin_strategy,
                    margin=self.gist_margin,
                    contrast_anchors=self.gist_contrast_anchors,
                    contrast_positives=self.gist_contrast_positives,
                    mini_batch_size=self.loss_mini_batch_size,
                )
            else:
                return loss_class(
                    model=model,
                    guide=guide_model,
                    temperature=self.gist_temperature,
                    margin_strategy=self.gist_margin_strategy,
                    margin=self.gist_margin,
                    contrast_anchors=self.gist_contrast_anchors,
                    contrast_positives=self.gist_contrast_positives,
                )
        elif is_cached:
            if self.loss_mini_batch_size is None:
                self.loss_mini_batch_size = 32
            return loss_class(
                model=model,
                scale=self.loss_scale,
                mini_batch_size=self.loss_mini_batch_size,
            )
        else:
            return loss_class(model=model, scale=self.loss_scale)

    def get_simcse_data(
        self, paragraph_file: str, cutoff_date: pd.Timestamp
    ) -> tuple[list[InputExample], list[InputExample], pd.DataFrame]:
        """Get data for sentence pair training."""
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
        """Train a sentence embedding model using sentence pairs."""
        config = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "effective_batch_size": self.effective_batch_size,
            "epochs": self.epochs,
            "warmup_steps": self.warmup_steps,
            "validation_split": self.validation_split,
            "cutoff_date": str(cutoff_date),
            "loss_type": self.loss_type,
            "loss_scale": self.loss_scale,
        }
        if self.guide_model_name is not None:
            config["guide_model_name"] = self.guide_model_name

        self.setup_wandb(config)

        train_dataset, val_dataset, val_df = self.get_simcse_data(
            paragraph_file, cutoff_date
        )
        train_dataloader = NoDuplicatesDataLoader(
            train_dataset, batch_size=self.batch_size
        )

        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length
        train_loss = self._create_loss(model)

        evaluator = self.create_ir_evaluator(val_df)

        print(f"\nTraining {self.model_name} with {self.loss_type}...")
        print(f"Total training examples: {len(train_dataset)}")
        print(f"Total validation examples: {len(val_dataset)}")

        trained_model = self._train_model(
            train_dataloader, train_loss, evaluator, "Training sentence pair model"
        )

        self.cleanup_wandb()
        return trained_model

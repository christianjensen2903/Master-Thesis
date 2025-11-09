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
        use_wandb: bool = True,
        max_seq_length: int | None = None,
        loss_scale: float = 20.0,
    ):
        super().__init__(model_name, training_args, use_wandb, max_seq_length)
        self.loss_scale = loss_scale

    def get_training_data(
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> tuple[Dataset, dict[str, str], dict[str, str], dict[str, set[str]]]:
        """Get data for sentence pair training."""
        train_pairs, corpus, queries, relevant_docs = self.load_and_split_data(
            judgments_path, queries_path, qrel_path, cutoff_year
        )

        train_data = []
        for query_id, query_text, doc_id, doc_text in tqdm(
            train_pairs,
            desc="Creating training dataset",
        ):
            train_data.append({"sentence1": query_text, "sentence2": doc_text})

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, corpus, queries, relevant_docs

    def train(
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> SentenceTransformer:
        """Train a sentence embedding model using SIMCSE with MultipleNegativesRankingLoss."""

        train_dataset, corpus, queries, relevant_docs = self.get_training_data(
            judgments_path, queries_path, qrel_path, cutoff_year
        )

        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length

        train_loss = losses.MultipleNegativesRankingLoss(
            model=model, scale=self.loss_scale
        )
        evaluator = self.create_ir_evaluator(corpus, queries, relevant_docs)

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
    trainer.train(
        judgments_path="data/judgments_cleaned.json",
        queries_path="data/evaluation/queries.tsv",
        qrel_path="data/evaluation/qrel.txt",
        cutoff_year=2018,
    )

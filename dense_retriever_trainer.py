import os
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from datasets import Dataset  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    util,
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from validation_utils import split_data_by_date
from torch import nn
from torch import Tensor
from typing import Iterable, Any
import torch


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
        gather_across_devices: bool = False,
        citation_graph: dict[str, set[str]] | None = None,
    ) -> None:
        """
        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value
            similarity_fct: similarity function between sentence embeddings
            gather_across_devices: If True, gather embeddings across all devices
            citation_graph: Dictionary mapping from_id -> set of all to_ids it cites
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.gather_across_devices = gather_across_devices
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.citation_graph = citation_graph or {}

        # Convert citation graph to use hashed IDs for faster lookup
        self.citation_graph_hashed = {}
        if citation_graph:
            for from_id, to_id_set in citation_graph.items():
                from_hash = self._hash_id(from_id)
                to_hash_set = {self._hash_id(to_id) for to_id in to_id_set}
                self.citation_graph_hashed[from_hash] = to_hash_set

    @staticmethod
    def _hash_id(id_str: str) -> int:
        """Create a stable hash for an ID string."""
        return hash(id_str) % (2**31)

    def forward(
        self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor
    ) -> Tensor:
        embeddings = [
            self.model(sentence_feature)["sentence_embedding"]
            for sentence_feature in sentence_features
        ]

        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(
        self, embeddings: list[Tensor], labels: Tensor
    ) -> Tensor:
        """
        Compute the multiple negatives ranking loss from embeddings.

        Args:
            embeddings: List of embeddings [anchors, positives]
            labels: Tensor of shape (batch_size, 2) where each row is:
                   [anchor_id_hash, positive_id_hash]
                   - anchor_id_hash: hash of the FROM_ID
                   - positive_id_hash: hash of the TO_ID

        Returns:
            Loss value
        """
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = embeddings[1:]
        batch_size = anchors.size(0)
        offset = 0

        # Extract IDs from labels
        anchor_ids = labels[:, 0] if labels is not None else None  # FROM_IDs (hashed)
        positive_ids = labels[:, 1] if labels is not None else None  # TO_IDs (hashed)

        if self.gather_across_devices:
            candidates = [
                util.all_gather_with_grad(embedding_column)
                for embedding_column in candidates
            ]

            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

                # Gather the IDs from all devices
                if positive_ids is not None:
                    world_size = torch.distributed.get_world_size()
                    gathered_anchor_ids = [
                        torch.zeros_like(anchor_ids) for _ in range(world_size)
                    ]
                    gathered_positive_ids = [
                        torch.zeros_like(positive_ids) for _ in range(world_size)
                    ]
                    torch.distributed.all_gather(gathered_anchor_ids, anchor_ids)
                    torch.distributed.all_gather(gathered_positive_ids, positive_ids)
                    all_anchor_ids = torch.cat(gathered_anchor_ids, dim=0)
                    all_positive_ids = torch.cat(gathered_positive_ids, dim=0)
                else:
                    all_anchor_ids = anchor_ids
                    all_positive_ids = positive_ids
            else:
                all_anchor_ids = anchor_ids
                all_positive_ids = positive_ids
        else:
            all_anchor_ids = anchor_ids
            all_positive_ids = positive_ids

        candidates = torch.cat(candidates, dim=0)

        # Compute similarity scores
        scores = self.similarity_fct(anchors, candidates) * self.scale
        # (batch_size, num_candidates)

        # Create labels for the target (which candidate is the designated positive)
        range_labels = torch.arange(offset, offset + batch_size, device=anchors.device)

        # Mask out false negatives using the citation graph
        if (
            anchor_ids is not None
            and all_positive_ids is not None
            and self.citation_graph_hashed
        ):
            # Create false negative mask: (batch_size, num_candidates)
            false_negative_mask = torch.zeros_like(scores, dtype=torch.bool)

            # For each anchor, check if any candidate is actually a positive based on citation graph
            for i in range(batch_size):
                anchor_id = anchor_ids[i].item()

                # Get all documents that this anchor cites
                cited_docs = self.citation_graph_hashed.get(anchor_id, set())

                if cited_docs:
                    # Check each candidate
                    for j in range(all_positive_ids.size(0)):
                        candidate_id = all_positive_ids[j].item()

                        # If this candidate is cited by the anchor, it's a false negative
                        if candidate_id in cited_docs:
                            false_negative_mask[i, j] = True

            # The designated positive should still be included (not masked)
            # So we unmask it
            designated_positive_mask = torch.zeros_like(false_negative_mask)
            designated_positive_mask[
                torch.arange(batch_size, device=scores.device), range_labels
            ] = True

            # Remove the designated positive from false negatives
            false_negative_mask = false_negative_mask & ~designated_positive_mask

            # Set false negative scores to -inf
            scores = scores.masked_fill(false_negative_mask, float("-inf"))

        return self.cross_entropy_loss(scores, range_labels)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "gather_across_devices": self.gather_across_devices,
        }


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

    def build_citation_graph(self, df: pd.DataFrame) -> dict[str, set[str]]:
        """Build a complete citation graph from the dataframe."""
        citation_graph: dict[str, set[str]] = {}

        for _, row in tqdm(
            df.iterrows(), total=len(df), desc="Building citation graph"
        ):
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])

            if from_id not in citation_graph:
                citation_graph[from_id] = set()
            citation_graph[from_id].add(to_id)

        print(f"Built citation graph with {len(citation_graph)} documents")
        total_citations = sum(len(v) for v in citation_graph.values())
        print(f"Total citation relationships: {total_citations}")

        return citation_graph

    def get_simcse_data(
        self, paragraph_file: str, cutoff_year: int
    ) -> tuple[Dataset, pd.DataFrame, pd.DataFrame, dict[str, set[str]]]:
        """Get data for sentence pair training with citation graph."""
        train_df, val_df = self.load_and_split_data(paragraph_file, cutoff_year)

        # Build citation graph from training data
        citation_graph = self.build_citation_graph(train_df)

        # Hash function for consistency
        def hash_id(id_str: str) -> int:
            return hash(id_str) % (2**31)

        train_data = []
        for idx, row in tqdm(
            train_df.iterrows(),
            total=len(train_df),
            desc="Creating training dataset",
        ):
            text_from = str(row["TEXT_FROM"])
            text_to = str(row["TEXT_TO"])
            from_id = str(row["FROM_ID"])
            to_id = str(row["TO_ID"])

            train_data.append(
                {
                    "sentence1": text_from,
                    "sentence2": text_to,
                    "label": [hash_id(from_id), hash_id(to_id)],
                }
            )

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, val_df, train_df, citation_graph

    def train(self, paragraph_file: str, cutoff_year: int) -> SentenceTransformer:
        """Train a sentence embedding model using SIMCSE with MultipleNegativesRankingLoss."""

        train_dataset, val_df, train_df, citation_graph = self.get_simcse_data(
            paragraph_file, cutoff_year
        )

        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length

        # Pass the citation graph to the loss function
        train_loss = MultipleNegativesRankingLoss(
            model=model, scale=self.loss_scale, citation_graph=citation_graph
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
            # output_dir="checkpoints/test_dense_retriever",
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
            # report_to="wandb",
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

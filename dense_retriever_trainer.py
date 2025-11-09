import os
import json
from datetime import datetime as dt

import numpy as np
from tqdm import tqdm  # type: ignore
from datasets import Dataset  # type: ignore
import bm25s  # type: ignore
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator


class DenseRetrieverTrainer:
    """Trainer for dense retrievers with optional BM25-based hard negative sampling."""

    def __init__(
        self,
        model_name: str,
        training_args: SentenceTransformerTrainingArguments,
        use_wandb: bool = True,
        max_seq_length: int | None = None,
        loss_scale: float = 20.0,
        num_negatives: int = 1,
    ):
        self.model_name = model_name
        self.training_args = training_args
        self.use_wandb = use_wandb
        self.max_seq_length = max_seq_length
        self.loss_scale = loss_scale
        self.num_negatives = num_negatives

        if self.training_args is not None:
            output_dir_init = self.training_args.output_dir
            if output_dir_init is not None:
                os.makedirs(output_dir_init, exist_ok=True)
        else:
            os.makedirs("output/model", exist_ok=True)

    def load_and_split_data(
        self,
        judgments_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> tuple[
        dict[str, list[str]],
        dict[str, str],
        dict[str, str],
        dict[str, str],
        dict[str, set[str]],
    ]:
        """Load data from judgments and qrel files, grouped by query."""
        print("Loading judgments...")
        with open(judgments_path) as f:
            judgments = json.load(f)

        # Build paragraph index: (celex, number) -> text and date
        paragraph_data: dict[tuple[str, int], tuple[str, dt]] = {}

        for celex, judgment in judgments.items():
            meta = judgment.get("meta", {}).get("meta", {})
            date_str = meta.get("date")

            try:
                date = dt.strptime(date_str, "%Y-%m-%d")
            except:
                continue

            for par_num, text in judgment["paragraphs"].items():
                key = (celex, int(par_num))
                paragraph_data[key] = (text, date)

        print(f"Loaded {len(paragraph_data)} paragraphs from judgments")

        # Load qrel grouped by query
        print("Loading qrel...")
        train_qrel: dict[str, list[str]] = {}
        train_queries: dict[str, str] = {}
        train_corpus: dict[str, str] = {}
        val_queries: dict[str, str] = {}
        val_relevant: dict[str, set[str]] = {}

        cutoff_date = dt(cutoff_year, 1, 1)

        with open(qrel_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                query_id_raw = parts[0]
                doc_id_raw = parts[2]

                # Parse celex_paragraph_number format
                celex_q, par_num_q = query_id_raw.rsplit("_", 1)
                celex_d, par_num_d = doc_id_raw.rsplit("_", 1)

                query_key = (celex_q, int(par_num_q))
                doc_key = (celex_d, int(par_num_d))

                # Only include if both exist
                if query_key not in paragraph_data or doc_key not in paragraph_data:
                    continue

                query_text, query_date = paragraph_data[query_key]
                doc_text, _ = paragraph_data[doc_key]

                # Create IDs in format celex::paragraph_number
                query_id_str = f"{query_key[0]}::{query_key[1]}"
                doc_id_str = f"{doc_key[0]}::{doc_key[1]}"

                # Add to corpus
                train_corpus[query_id_str] = query_text
                train_corpus[doc_id_str] = doc_text

                if query_date < cutoff_date:
                    # Training data
                    if query_id_str not in train_qrel:
                        train_qrel[query_id_str] = []
                        train_queries[query_id_str] = query_text
                    train_qrel[query_id_str].append(doc_id_str)
                else:
                    # Validation data
                    if doc_id_str in train_corpus:
                        if query_id_str not in val_queries:
                            val_queries[query_id_str] = query_text
                            val_relevant[query_id_str] = set()
                        val_relevant[query_id_str].add(doc_id_str)

        train_citations = sum(len(docs) for docs in train_qrel.values())
        val_citations = sum(len(docs) for docs in val_relevant.values())

        print(f"\n📅 Temporal Split:")
        print(
            f"  Train: before {cutoff_date.date()} ({len(train_qrel)} queries, {train_citations} citations)"
        )
        print(
            f"  Val: after {cutoff_date.date()} ({len(val_queries)} queries, {val_citations} citations)"
        )

        return train_qrel, train_queries, train_corpus, val_queries, val_relevant

    def create_ir_evaluator(
        self,
        corpus: dict[str, str],
        queries: dict[str, str],
        relevant_docs: dict[str, set[str]],
    ) -> InformationRetrievalEvaluator:
        """Create an Information Retrieval evaluator for validation."""
        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="validation_ir",
            show_progress_bar=True,
            map_at_k=[1000],
            precision_recall_at_k=[5, 10, 50, 100],
        )

        return evaluator

    def _select_hard_negatives(
        self, ranked_indices: np.ndarray, positive_indices: set[int]
    ) -> list[int]:
        """Select top hard negatives from ranked candidates, excluding all positives."""
        # ranked_indices contains candidate indices sorted by similarity (highest first)
        # Exclude all positives from ranked list
        candidates = ranked_indices[~np.isin(ranked_indices, list(positive_indices))]

        if len(candidates) == 0:
            return []

        # Select top negatives (already sorted by similarity)
        num_samples = min(self.num_negatives, len(candidates))
        return candidates[:num_samples].tolist()

    def get_training_data(
        self,
        judgments_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> tuple[Dataset, dict[str, str], dict[str, str], dict[str, set[str]]]:
        """Get training data with optional BM25-based hard negatives."""
        train_qrel, train_queries, corpus, val_queries, val_relevant = (
            self.load_and_split_data(judgments_path, qrel_path, cutoff_year)
        )

        train_data = []

        if self.num_negatives == 0:
            # Simple pair format when no negatives are used
            for query_id, query_text in tqdm(
                train_queries.items(),
                desc="Creating training dataset",
            ):
                for doc_id in train_qrel[query_id]:
                    doc_text = corpus[doc_id]
                    train_data.append({"sentence1": query_text, "sentence2": doc_text})
        else:
            # Use BM25 to find hard negatives
            candidate_texts = list(corpus.values())
            candidate_ids = list(corpus.keys())

            print("Fitting BM25 on training data...")
            tokenized_corpus = bm25s.tokenize(candidate_texts)
            retriever = bm25s.BM25(corpus=candidate_texts)
            retriever.index(tokenized_corpus)

            # Create mapping from document ID to index for efficient lookup
            id_to_idx = {doc_id: idx for idx, doc_id in enumerate(candidate_ids)}

            for query_id, query_text in tqdm(
                train_queries.items(),
                desc="Creating training dataset with hard negatives",
            ):
                positive_doc_ids = train_qrel[query_id]
                num_positives = len(positive_doc_ids)

                # Get indices of all positive documents
                positive_indices = set()
                for doc_id in positive_doc_ids:
                    idx = id_to_idx.get(doc_id)
                    if idx is not None:
                        positive_indices.add(idx)

                if len(positive_indices) == 0:
                    continue

                # Rank candidates using BM25 with k = num_positives + num_negatives
                k = num_positives + self.num_negatives
                tokenized_query = bm25s.tokenize(query_text, show_progress=False)
                docs, _ = retriever.retrieve(
                    tokenized_query,
                    k=k,
                    show_progress=False,
                )
                ranked_indices = np.array(docs[0])

                # Select hard negatives (excluding all positives)
                hard_negative_indices = self._select_hard_negatives(
                    ranked_indices, positive_indices
                )

                # Create training examples: each positive gets n negatives
                for doc_id in positive_doc_ids:
                    doc_text = corpus[doc_id]

                    if len(hard_negative_indices) == 0:
                        # Fallback: use positive pair only if no negatives found
                        train_data.append(
                            {
                                "sentence1": query_text,
                                "sentence2": doc_text,
                            }
                        )
                    else:
                        # Create one example per negative for this positive
                        for neg_idx in hard_negative_indices:
                            neg_text = candidate_texts[neg_idx]
                            train_data.append(
                                {
                                    "sentence1": query_text,
                                    "sentence2": doc_text,
                                    "sentence3": neg_text,
                                }
                            )

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, corpus, val_queries, val_relevant

    def train(
        self,
        judgments_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> SentenceTransformer:
        """Train the dense retriever model."""
        train_dataset, corpus, queries, relevant_docs = self.get_training_data(
            judgments_path, qrel_path, cutoff_year
        )

        model = SentenceTransformer(self.model_name)
        if self.max_seq_length is not None:
            model.max_seq_length = self.max_seq_length

        train_loss = losses.MultipleNegativesRankingLoss(
            model=model, scale=self.loss_scale
        )
        evaluator = self.create_ir_evaluator(corpus, queries, relevant_docs)

        if self.num_negatives == 0:
            print(f"\nTraining {self.model_name} with MultipleNegativesRankingLoss...")
        else:
            print(
                f"\nTraining {self.model_name} with Hard Negatives (top {self.num_negatives} from BM25)..."
            )
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

import os
import json
import random
from datetime import datetime as dt

import numpy as np
from tqdm import tqdm  # type: ignore
from datasets import Dataset  # type: ignore
from rank_bm25 import BM25Okapi  # type: ignore
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
        hard_negative_min_rank: int = 100,
        hard_negative_max_rank: int = 300,
    ):
        self.model_name = model_name
        self.training_args = training_args
        self.use_wandb = use_wandb
        self.max_seq_length = max_seq_length
        self.loss_scale = loss_scale
        self.num_negatives = num_negatives
        self.hard_negative_min_rank = hard_negative_min_rank
        self.hard_negative_max_rank = hard_negative_max_rank

        if self.training_args is not None:
            output_dir_init = self.training_args.output_dir
            if output_dir_init is not None:
                os.makedirs(output_dir_init, exist_ok=True)
        else:
            os.makedirs("output/model", exist_ok=True)

    def load_and_split_data(
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> tuple[
        list[tuple[str, str, str, str]],
        dict[str, str],
        dict[str, str],
        dict[str, set[str]],
    ]:
        """Load data from judgments, queries, and qrel files and split by date."""
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

        # Load qrel to get citation pairs
        print("Loading qrel...")
        citation_pairs: list[tuple[tuple[str, int], tuple[str, int]]] = []

        with open(qrel_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                query_id = parts[0]
                doc_id = parts[2]

                # Parse celex_paragraph_number format
                celex_q, par_num_q = query_id.rsplit("_", 1)
                celex_d, par_num_d = doc_id.rsplit("_", 1)

                query_key = (celex_q, int(par_num_q))
                doc_key = (celex_d, int(par_num_d))

                # Only include if both exist
                if query_key in paragraph_data and doc_key in paragraph_data:
                    citation_pairs.append((query_key, doc_key))

        print(f"Loaded {len(citation_pairs)} citation pairs from qrel")

        # Split into train and validation based on query date
        train_pairs: list[tuple[str, str, str, str]] = []
        val_pairs: list[tuple[str, str, str, str]] = []

        cutoff_date = dt(cutoff_year, 1, 1)

        for query_key, doc_key in citation_pairs:
            query_text, query_date = paragraph_data[query_key]
            doc_text, _ = paragraph_data[doc_key]

            # Create IDs in format celex::paragraph_number
            query_id_str = f"{query_key[0]}::{query_key[1]}"
            doc_id_str = f"{doc_key[0]}::{doc_key[1]}"

            if query_date < cutoff_date:
                train_pairs.append((query_id_str, query_text, doc_id_str, doc_text))
            else:
                val_pairs.append((query_id_str, query_text, doc_id_str, doc_text))

        print(f"\n📅 Temporal Split:")
        print(f"  Train: before {cutoff_date.date()} ({len(train_pairs)} citations)")
        print(f"  Val: after {cutoff_date.date()} ({len(val_pairs)} citations)")

        # Build evaluator structures
        train_corpus: dict[str, str] = {}
        val_queries: dict[str, str] = {}
        val_relevant: dict[str, set[str]] = {}

        # Add all train documents to corpus
        for qid, qtext, did, dtext in train_pairs:
            train_corpus[qid] = qtext
            train_corpus[did] = dtext

        # Build validation queries and relevant docs
        for qid, qtext, did, dtext in val_pairs:
            # Only include if document exists in train corpus
            if did in train_corpus:
                if qid not in val_queries:
                    val_queries[qid] = qtext
                    val_relevant[qid] = set()
                val_relevant[qid].add(did)

        return train_pairs, train_corpus, val_queries, val_relevant

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

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()

    def _select_hard_negatives(
        self, ranked_indices: np.ndarray, positive_idx: int
    ) -> list[int]:
        """Select random hard negatives from the specified rank range."""
        # ranked_indices contains candidate indices sorted by similarity (highest first)
        # Exclude the positive from ranked list
        valid_ranked = ranked_indices[ranked_indices != positive_idx]

        # Get rank range (0-indexed positions in ranked list)
        min_rank = min(self.hard_negative_min_rank, len(valid_ranked))
        max_rank = min(self.hard_negative_max_rank, len(valid_ranked))

        if min_rank >= max_rank:
            # Fallback: use all available candidates after positive
            candidates = valid_ranked.tolist()
        else:
            candidates = valid_ranked[min_rank:max_rank].tolist()

        if len(candidates) == 0:
            return []

        # Ensure positive is not in candidates (defensive check)
        candidates = [idx for idx in candidates if idx != positive_idx]

        if len(candidates) == 0:
            return []

        # Sample random negatives
        num_samples = min(self.num_negatives, len(candidates))
        return random.sample(candidates, num_samples)

    def get_training_data(
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> tuple[Dataset, dict[str, str], dict[str, str], dict[str, set[str]]]:
        """Get training data with optional BM25-based hard negatives."""
        train_pairs, corpus, queries, relevant_docs = self.load_and_split_data(
            judgments_path, queries_path, qrel_path, cutoff_year
        )

        train_data = []

        if self.num_negatives == 0:
            # Simple pair format when no negatives are used
            for query_id, query_text, doc_id, doc_text in tqdm(
                train_pairs,
                desc="Creating training dataset",
            ):
                train_data.append({"sentence1": query_text, "sentence2": doc_text})
        else:
            # Use BM25 to find hard negatives
            candidate_texts = list(corpus.values())
            candidate_ids = list(corpus.keys())

            print("Fitting BM25 on training data...")
            tokenized_corpus = [self._tokenize(text) for text in candidate_texts]
            bm25 = BM25Okapi(tokenized_corpus)

            for query_id, query_text, doc_id, doc_text in tqdm(
                train_pairs,
                desc="Creating training dataset with hard negatives",
            ):
                # Rank all candidates using BM25
                tokenized_query = self._tokenize(query_text)
                scores = bm25.get_scores(tokenized_query)
                ranked_indices = np.argsort(-scores)

                # Find positive index in candidate list
                positive_idx = None
                for i, cand_id in enumerate(candidate_ids):
                    if cand_id == doc_id:
                        positive_idx = i
                        break

                if positive_idx is None:
                    # Fallback: use positive pair only
                    train_data.append(
                        {
                            "sentence1": query_text,
                            "sentence2": doc_text,
                        }
                    )
                    continue

                # Select hard negatives
                hard_negative_indices = self._select_hard_negatives(
                    ranked_indices, positive_idx
                )

                if len(hard_negative_indices) == 0:
                    # Fallback: use positive pair only if no negatives found
                    train_data.append(
                        {
                            "sentence1": query_text,
                            "sentence2": doc_text,
                        }
                    )
                    continue

                # Use first hard negative for triplet format (anchor, positive, negative)
                neg_idx = hard_negative_indices[0]
                neg_text = candidate_texts[neg_idx]
                train_data.append(
                    {
                        "sentence1": query_text,
                        "sentence2": doc_text,
                        "sentence3": neg_text,
                    }
                )

        train_dataset = Dataset.from_list(train_data)
        return train_dataset, corpus, queries, relevant_docs

    def train(
        self,
        judgments_path: str,
        queries_path: str,
        qrel_path: str,
        cutoff_year: int,
    ) -> SentenceTransformer:
        """Train the dense retriever model."""
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

        if self.num_negatives == 0:
            print(f"\nTraining {self.model_name} with MultipleNegativesRankingLoss...")
        else:
            print(
                f"\nTraining {self.model_name} with Hard Negatives (BM25 ranks {self.hard_negative_min_rank}-{self.hard_negative_max_rank})..."
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

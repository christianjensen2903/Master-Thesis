import json
import re
from datetime import datetime as dt
from collections import defaultdict
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore

from data_loader import (
    load_citation_data,
    split_train_test,
    build_paragraph_index,
    build_citation_graph,
)
from retrievers.base_retriever import BaseRetriever

# Type alias for evaluator modes
EvaluatorMode = Literal["citation_pairs", "all_paragraphs"]


class Evaluator:
    def __init__(
        self,
        retriever: BaseRetriever,
        embeddings: NDArray | None = None,
        mode: EvaluatorMode = "citation_pairs",
        csv_path: str = "data/par-to-par-cleaned.csv",
        metadata_path: str = "data/par-to-par.json",
        judgments_path: str = "data/judgments_cleaned.json",
        train_cutoff_year: int = 2018,
        top_k: int | None = None,
    ):
        self.retriever = retriever
        self.embeddings = embeddings
        self.mode: EvaluatorMode = mode
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.judgments_path = judgments_path
        self.train_cutoff_year = train_cutoff_year
        self.top_k = top_k

        # Validate mode
        if mode not in ["citation_pairs", "all_paragraphs"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'citation_pairs' or 'all_paragraphs'"
            )

        # Data structures (populated by load_and_prepare)
        self.df: pd.DataFrame | None = None
        self.metadata: dict | None = None
        self.train_meta: list[dict] | None = None
        self.test_meta: list[dict] | None = None

        self.pid_to_text: NDArray[np.object_] | None = None
        self.celex_number_to_pid: dict[tuple[str, int], int] | None = None
        self.paragraph_dates: NDArray | None = None
        self.paragraph_celex: NDArray[np.object_] | None = None
        self.paragraph_number: NDArray[np.object_] | None = None
        self.paragraph_set: NDArray[np.object_] | None = None

        self.cited_by_pid: dict[int, list[int]] | None = None

        self.sort_idx: NDArray | None = None
        self.sorted_dates: NDArray | None = None

        self.map_score: float | None = None
        self.recall_scores: dict[int, float] | None = None

    def load_and_prepare(self) -> None:
        if self.mode == "citation_pairs":
            self._load_citation_pairs_mode()
        else:
            self._load_all_paragraphs_mode()

        # Prepare temporal index
        self._prepare_temporal_index()

    def _load_citation_pairs_mode(self) -> None:
        """Load data in citation pairs mode (using par-to-par CSV)"""
        self.df, self.metadata = load_citation_data(self.csv_path, self.metadata_path)
        self.train_meta, self.test_meta = split_train_test(
            self.metadata, self.train_cutoff_year
        )

        (
            self.pid_to_text,
            self.celex_number_to_pid,
            self.paragraph_dates,
            self.paragraph_celex,
            self.paragraph_number,
            self.paragraph_set,
        ) = build_paragraph_index(self.df, self.train_meta, self.test_meta)

        # Build citation graph
        self.cited_by_pid = build_citation_graph(self.df, self.celex_number_to_pid)

    def _load_all_paragraphs_mode(self) -> None:
        """
        Load data in all paragraphs mode (using judgments_cleaned.json + par-to-par for citations)

        This mode:
        - Loads ALL paragraphs from judgments_cleaned.json as candidates
        - Uses par-to-par-cleaned.csv for ground truth citations
        - Allows evaluation against all possible paragraphs, not just those in par-to-par
        """
        print("Loading judgments...")
        with open(self.judgments_path) as f:
            judgments = json.load(f)

        # Build paragraph index from all judgments
        paragraphs = []
        for judgment in tqdm(judgments, desc="Processing judgments"):
            # Extract CELEX ID from file path
            files = judgment.get("meta", {}).get("files", {})
            if not files:
                continue

            # Get first available file path
            first_lang = list(files.values())[0] if files else None
            if not first_lang:
                continue

            file_path = (
                first_lang.get("judgment", "") if isinstance(first_lang, dict) else ""
            )
            if not file_path:
                continue

            # Extract CELEX ID from path (e.g., "61954CJ0001")
            # Path format: F:\ECJ\new_files\61954CJ0001\FR\judgment.html
            match = re.search(r"(\d+[A-Z]+\d+)", file_path)
            if not match:
                continue
            celex = match.group(1)

            # Get date from meta
            meta = judgment.get("meta", {}).get("meta", {})
            date_str = meta.get("date")

            try:
                date = dt.strptime(date_str, "%Y-%m-%d")
            except:
                continue

            year = date.year
            set_type = "train" if year < self.train_cutoff_year else "test"

            for par_num, text in judgment["paragraphs"].items():
                paragraphs.append(
                    {
                        "text": text,
                        "celex": celex,
                        "date": date,
                        "number": int(par_num),
                        "set_type": set_type,
                    }
                )

        # Sort paragraphs by (celex, number) to maintain document order
        paragraphs.sort(key=lambda p: (p["celex"], p["number"]))

        # Build arrays
        self.pid_to_text = np.array([p["text"] for p in paragraphs], dtype=object)
        self.celex_number_to_pid = {
            (p["celex"], p["number"]): pid for pid, p in enumerate(paragraphs)
        }
        self.paragraph_dates = np.array(
            [p["date"] for p in paragraphs], dtype="datetime64[ns]"
        )
        self.paragraph_celex = np.array([p["celex"] for p in paragraphs], dtype=object)
        self.paragraph_number = np.array(
            [p["number"] for p in paragraphs], dtype=object
        )
        self.paragraph_set = np.array([p["set_type"] for p in paragraphs], dtype=object)

        # Load citation pairs from par-to-par for ground truth
        print("Loading citation pairs for ground truth...")
        self.df = pd.read_csv(self.csv_path).dropna()

        # Build citation graph from par-to-par (only for paragraphs that exist in our index)
        self.cited_by_pid = self._build_citation_graph_safe(
            self.df, self.celex_number_to_pid
        )

    def _build_citation_graph_safe(
        self, df: pd.DataFrame, celex_number_to_pid: dict[tuple[str, int], int]
    ) -> dict[int, list[int]]:
        """
        Build citation graph using (celex, number) keys, skipping paragraphs that don't exist in the index.
        """
        cited_by_pid = defaultdict(set)
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building citations"):
            celex_from = row["CELEX_FROM"]
            number_from = row["NUMBER_FROM"]
            celex_to = row["CELEX_TO"]
            number_to = row["NUMBER_TO"]

            src_key = (str(celex_from), int(number_from))
            tgt_key = (str(celex_to), int(number_to))

            # Skip if either key not in our index
            if src_key not in celex_number_to_pid or tgt_key not in celex_number_to_pid:
                skipped += 1
                continue

            src_pid = celex_number_to_pid[src_key]
            tgt_pid = celex_number_to_pid[tgt_key]
            cited_by_pid[src_pid].add(tgt_pid)

        if skipped > 0:
            print(
                f"Skipped {skipped}/{len(df)} citation pairs (paragraphs not in index)"
            )

        # Make deterministic and convert to dict
        result: dict[int, list[int]] = {k: sorted(v) for k, v in cited_by_pid.items()}
        return result

    def _prepare_temporal_index(self) -> None:
        """Pre-sort paragraph IDs by date for temporal filtering."""
        assert self.paragraph_dates is not None
        self.sort_idx = np.argsort(self.paragraph_dates)
        self.sorted_dates = self.paragraph_dates[self.sort_idx]

    def evaluate_map(self) -> float:
        assert self.embeddings is not None
        assert self.pid_to_text is not None
        assert self.paragraph_set is not None
        assert self.cited_by_pid is not None

        test_source_pids = [
            pid
            for pid in range(len(self.pid_to_text))
            if self.paragraph_set[pid] == "test"
            and len(self.cited_by_pid.get(pid, [])) > 0
        ]

        avg_precs = []

        desc = f"Evaluating MAP@{self.top_k}" if self.top_k else "Evaluating MAP"

        for src_pid in tqdm(test_source_pids, desc=desc):  # type: ignore
            assert self.paragraph_dates is not None
            assert self.sorted_dates is not None
            assert self.sort_idx is not None

            src_date = self.paragraph_dates[src_pid]

            # Get all paragraphs strictly older than source
            cutoff = int(np.searchsorted(self.sorted_dates, src_date, side="left"))
            cand_pids = self.sort_idx[:cutoff]

            if len(cand_pids) == 0:
                continue

            # Ground truth: cited paragraphs that are also older
            relevant = set(self.cited_by_pid[src_pid]).intersection(set(cand_pids))
            num_rel = len(relevant)
            if num_rel == 0:
                continue

            # Retrieve and rank candidates (with optional top_k limit)
            ranked_pids = self.retriever.retrieve(
                src_pid, self.embeddings, cand_pids, top_k=self.top_k
            )

            # Compute average precision (only up to top_k if specified)
            good = 0
            precisions = []
            max_rank = (
                len(ranked_pids)
                if self.top_k is None
                else min(len(ranked_pids), self.top_k)
            )

            for rank_pos, pid_candidate in enumerate(ranked_pids[:max_rank], start=1):
                if pid_candidate in relevant:
                    good += 1
                    precisions.append(good / rank_pos)
                    if good == num_rel:
                        break

            ap = float(np.sum(precisions) / num_rel) if precisions else 0.0
            avg_precs.append(ap)

        self.map_score = float(np.mean(avg_precs)) if avg_precs else 0.0
        return self.map_score

    def evaluate_recall(
        self, k_values: list[int] = [5, 10, 50, 100]
    ) -> dict[int, float]:
        assert self.embeddings is not None
        assert self.pid_to_text is not None
        assert self.paragraph_set is not None
        assert self.cited_by_pid is not None

        test_source_pids = [
            pid
            for pid in range(len(self.pid_to_text))
            if self.paragraph_set[pid] == "test"
            and len(self.cited_by_pid.get(pid, [])) > 0
        ]

        recall_at_k: dict[int, list[float]] = {k: [] for k in k_values}

        desc = "Evaluating Recall"
        for src_pid in tqdm(test_source_pids, desc=desc):  # type: ignore
            assert self.paragraph_dates is not None
            assert self.sorted_dates is not None
            assert self.sort_idx is not None

            src_date = self.paragraph_dates[src_pid]

            # Get all paragraphs strictly older than source
            cutoff = int(np.searchsorted(self.sorted_dates, src_date, side="left"))
            cand_pids = self.sort_idx[:cutoff]

            if len(cand_pids) == 0:
                continue

            # Ground truth: cited paragraphs that are also older
            relevant = set(self.cited_by_pid[src_pid]).intersection(set(cand_pids))
            num_rel = len(relevant)
            if num_rel == 0:
                continue

            # Retrieve and rank candidates (with optional top_k limit)
            ranked_pids = self.retriever.retrieve(
                src_pid, self.embeddings, cand_pids, top_k=self.top_k
            )

            # Compute recall at each k
            for k in k_values:
                top_k_pids = ranked_pids[:k]
                num_retrieved_relevant = len(set(top_k_pids).intersection(relevant))
                recall = num_retrieved_relevant / num_rel if num_rel > 0 else 0.0
                recall_at_k[k].append(recall)

        self.recall_scores = {
            k: float(np.mean(recalls)) if recalls else 0.0
            for k, recalls in recall_at_k.items()
        }
        return self.recall_scores

    def run(self) -> float:
        print(f"Mode: {self.mode}")
        print("Loading and preparing data...")
        self.load_and_prepare()

        assert self.pid_to_text is not None
        assert self.paragraph_set is not None

        print(f"Unique paragraphs: {len(self.pid_to_text)}")
        print(f"Train paragraphs: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test paragraphs: {np.sum(self.paragraph_set == 'test')}")

        # Generate embeddings if not provided
        if self.embeddings is None:
            print("\nGenerating embeddings from retriever...")
            train_mask = self.paragraph_set == "train"
            # Fit on training data, transform on all data
            self.retriever.fit(self.pid_to_text, mask=train_mask)
            self.embeddings = self.retriever.transform(self.pid_to_text)
            print(f"Embeddings shape: {self.embeddings.shape}")
        else:
            # Validate embeddings match paragraph index
            if len(self.embeddings) != len(self.pid_to_text):
                raise ValueError(
                    f"Embeddings size mismatch: got {len(self.embeddings)} embeddings "
                    f"but have {len(self.pid_to_text)} paragraphs. "
                    f"You must regenerate embeddings in '{self.mode}' mode."
                )

        if self.mode == "citation_pairs":
            assert self.df is not None
            print(f"Citation pairs: {len(self.df)}")

        metric_name = f"MAP@{self.top_k}" if self.top_k else "MAP"
        print(f"\nComputing {metric_name}...")
        score = self.evaluate_map()

        print(f"\n{metric_name}: {score:.4f}")

        print("\nComputing Recall@k...")
        recall_scores = self.evaluate_recall([5, 10, 50, 100])
        for k, recall in sorted(recall_scores.items()):
            print(f"Recall@{k}: {recall:.4f}")

        return score


if __name__ == "__main__":
    from retrievers import TfidfRetriever

    print("Initializing TF-IDF retriever...")
    retriever = TfidfRetriever(
        stop_words="english",
        strip_accents="ascii",
        norm="l2",
    )

    evaluator = Evaluator(
        retriever=retriever,
        # mode="all_paragraphs",
        csv_path="data/par-to-par-cleaned.csv",
        metadata_path="data/par-to-par.json",
        judgments_path="data/judgments_cleaned.json",
        train_cutoff_year=2018,
        top_k=10000,
    )

    score = evaluator.run()
    print(f"Final MAP: {score:.4f}")

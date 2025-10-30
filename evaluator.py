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
        embeddings: NDArray,
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
        self.text_to_pid: dict | None = None
        self.paragraph_dates: NDArray | None = None
        self.paragraph_celex: NDArray[np.object_] | None = None
        self.paragraph_number: NDArray[np.object_] | None = None
        self.paragraph_set: NDArray[np.object_] | None = None

        self.cited_by_pid: dict[int, list[int]] | None = None

        self.sort_idx: NDArray | None = None
        self.sorted_dates: NDArray | None = None

        self.map_score: float | None = None

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
            self.text_to_pid,
            self.paragraph_dates,
            self.paragraph_celex,
            self.paragraph_set,
        ) = build_paragraph_index(self.df, self.train_meta, self.test_meta)

        self.paragraph_number = np.array([None] * len(self.pid_to_text), dtype=object)

        # Build citation graph
        self.cited_by_pid = build_citation_graph(self.df, self.text_to_pid)

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
        self.text_to_pid = {text: pid for pid, text in enumerate(self.pid_to_text)}
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
        self.cited_by_pid = self._build_citation_graph_safe(self.df, self.text_to_pid)

    def _build_citation_graph_safe(
        self, df: pd.DataFrame, text_to_pid: dict
    ) -> dict[int, list[int]]:
        """
        Build citation graph, skipping paragraphs that don't exist in the index.
        """
        cited_by_pid = defaultdict(set)
        skipped = 0

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building citations"):
            src_txt = row["TEXT_FROM"]
            tgt_txt = row["TEXT_TO"]

            if not isinstance(src_txt, str) or not isinstance(tgt_txt, str):
                skipped += 1
                continue

            # Skip if either text not in our index
            if src_txt not in text_to_pid or tgt_txt not in text_to_pid:
                skipped += 1
                continue

            src_pid = text_to_pid[src_txt]
            tgt_pid = text_to_pid[tgt_txt]
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

    def run(self) -> float:
        print(f"Mode: {self.mode}")
        print("Loading and preparing data...")
        self.load_and_prepare()

        assert self.pid_to_text is not None
        assert self.paragraph_set is not None

        print(f"Unique paragraphs: {len(self.pid_to_text)}")
        print(f"Train paragraphs: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test paragraphs: {np.sum(self.paragraph_set == 'test')}")

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

        return score


if __name__ == "__main__":
    from retrievers import TfidfRetriever
    from data_loader import load_citation_data, split_train_test, build_paragraph_index

    print("Loading data...")
    csv_path = "data/par-to-par-cleaned.csv"
    metadata_path = "data/par-to-par.json"
    judgments_path = "data/judgments_cleaned.json"
    cutoff_year = 2018
    df, metadata = load_citation_data(csv_path, metadata_path)
    train_meta, test_meta = split_train_test(metadata, cutoff_year=cutoff_year)

    print("Building paragraph index...")
    pid_to_text, text_to_pid, paragraph_dates, paragraph_celex, paragraph_set = (
        build_paragraph_index(df, train_meta, test_meta)
    )

    print(f"Total paragraphs: {len(pid_to_text)}")
    print(f"Train paragraphs: {np.sum(paragraph_set == 'train')}")

    # Initialize and fit retriever
    print("\nFitting TF-IDF retriever...")
    retriever = TfidfRetriever(
        stop_words="english",
        strip_accents="ascii",
        norm="l2",
    )

    train_mask = paragraph_set == "train"
    embeddings = retriever.fit_transform(pid_to_text, train_mask)
    print(f"Embeddings shape: {embeddings.shape}")

    evaluator = Evaluator(
        retriever=None,  # We'll set this later
        embeddings=None,  # We'll set this later
        mode="all_paragraphs",
        csv_path=csv_path,
        metadata_path=metadata_path,
        judgments_path=judgments_path,
        train_cutoff_year=cutoff_year,
        top_k=10000,
    )

    # 2. Load the paragraph index
    evaluator.load_and_prepare()

    # Generate embeddings for ALL paragraphs in all_paragraphs mode
    assert evaluator.pid_to_text is not None
    embeddings = retriever.transform(evaluator.pid_to_text)

    # 4. Now set the retriever and embeddings
    evaluator.retriever = retriever
    evaluator.embeddings = embeddings

    # 5. Run evaluation (skip load_and_prepare since we already did it)
    score = evaluator.evaluate_map()
    print(f"MAP: {score:.4f}")

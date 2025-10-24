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


class Evaluator:
    def __init__(
        self,
        retriever: BaseRetriever,
        embeddings: NDArray,
        excel_path: str = "data/par-to-par-2.xlsx",
        metadata_path: str = "data/par-to-par.json",
        train_cutoff_year: int = 2018,
        top_k: int | None = None,
    ):
        self.retriever = retriever
        self.embeddings = embeddings
        self.excel_path = excel_path
        self.metadata_path = metadata_path
        self.train_cutoff_year = train_cutoff_year
        self.top_k = top_k

        # Data structures (populated by load_and_prepare)
        self.df: pd.DataFrame | None = None
        self.metadata: dict | None = None
        self.train_meta: list[dict] | None = None
        self.test_meta: list[dict] | None = None

        self.pid_to_text: NDArray[np.object_] | None = None
        self.text_to_pid: dict | None = None
        self.paragraph_dates: NDArray | None = None
        self.paragraph_celex: NDArray[np.object_] | None = None
        self.paragraph_set: NDArray[np.object_] | None = None

        self.cited_by_pid: dict[int, list[int]] | None = None

        self.sort_idx: NDArray | None = None
        self.sorted_dates: NDArray | None = None

        self.map_score: float | None = None

    def load_and_prepare(self) -> None:
        self.df, self.metadata = load_citation_data(self.excel_path, self.metadata_path)
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

        # Build citation graph
        self.cited_by_pid = build_citation_graph(self.df, self.text_to_pid)

        # Prepare temporal index
        self._prepare_temporal_index()

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

            ap = float(np.mean(precisions)) if precisions else 0.0
            avg_precs.append(ap)

        self.map_score = float(np.mean(avg_precs)) if avg_precs else 0.0
        return self.map_score

    def run(self) -> float:
        print("Loading and preparing data...")
        self.load_and_prepare()

        assert self.df is not None
        assert self.pid_to_text is not None
        assert self.paragraph_set is not None

        print(f"Rows (after dropna): {len(self.df)}")
        print(f"Unique paragraphs: {len(self.pid_to_text)}")
        print(f"Train paragraphs: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test paragraphs: {np.sum(self.paragraph_set == 'test')}")

        metric_name = f"MAP@{self.top_k}" if self.top_k else "MAP"
        print(f"\nComputing {metric_name}...")
        score = self.evaluate_map()

        print(f"\n{metric_name}: {score:.4f}")

        return score


if __name__ == "__main__":
    from retrievers import TfidfRetriever
    from data_loader import load_citation_data, split_train_test, build_paragraph_index

    print("Loading data...")
    df, metadata = load_citation_data()
    train_meta, test_meta = split_train_test(metadata, cutoff_year=2018)

    print("Building paragraph index...")
    pid_to_text, text_to_pid, paragraph_dates, paragraph_celex, paragraph_set = (
        build_paragraph_index(df, train_meta, test_meta)
    )

    print(f"Total paragraphs: {len(pid_to_text)}")
    print(f"Train paragraphs: {np.sum(paragraph_set == 'train')}")
    print(f"Test paragraphs: {np.sum(paragraph_set == 'test')}")

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

    # Run evaluation (with optional top_k for faster evaluation)
    print("\nEvaluating...")
    evaluator = Evaluator(
        retriever=retriever,
        embeddings=embeddings,
        top_k=1000,  # Use MAP@1000 for faster evaluation, or None for full MAP
    )
    evaluator.run()

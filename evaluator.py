import json
import csv
from datetime import datetime as dt
from collections import defaultdict
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore
from numba import njit, prange  # type: ignore

from retrievers.base_retriever import BaseRetriever

EvaluatorMode = Literal["citation_pairs", "all_paragraphs"]


@njit
def compute_ap_fast(
    ranked_pids: NDArray, relevant_pids: NDArray, max_rank: int
) -> float:
    num_rel = len(relevant_pids)
    if num_rel == 0:
        return 0.0

    precision_sum = 0.0
    num_found = 0

    for i in range(min(max_rank, len(ranked_pids))):
        is_relevant = False
        for j in range(len(relevant_pids)):
            if ranked_pids[i] == relevant_pids[j]:
                is_relevant = True
                break

        if is_relevant:
            num_found += 1
            precision_sum += num_found / (i + 1)

    return precision_sum / num_rel


@njit
def compute_recall_at_k_fast(
    ranked_pids: NDArray, relevant_pids: NDArray, k: int
) -> float:
    num_rel = len(relevant_pids)
    if num_rel == 0:
        return 0.0

    num_found = 0
    for i in range(min(k, len(ranked_pids))):
        for j in range(len(relevant_pids)):
            if ranked_pids[i] == relevant_pids[j]:
                num_found += 1
                break

    return num_found / num_rel


@njit(parallel=True)
def compute_metrics_batch(
    ranked_pids_list: list[NDArray],
    relevant_pids_list: list[NDArray],
    max_ranks: NDArray,
    k_values: NDArray,
) -> tuple[NDArray, NDArray]:
    n_queries = len(ranked_pids_list)
    n_k = len(k_values)

    avg_precs = np.zeros(n_queries, dtype=np.float64)
    recall_matrix = np.zeros((n_queries, n_k), dtype=np.float64)

    for idx in prange(n_queries):
        ranked = ranked_pids_list[idx]
        relevant = relevant_pids_list[idx]
        max_rank = max_ranks[idx]

        avg_precs[idx] = compute_ap_fast(ranked, relevant, max_rank)

        for k_idx in range(n_k):
            recall_matrix[idx, k_idx] = compute_recall_at_k_fast(
                ranked, relevant, k_values[k_idx]
            )

    return avg_precs, recall_matrix


class Evaluator:
    def __init__(
        self,
        retriever: BaseRetriever,
        embeddings: NDArray | None = None,
        mode: EvaluatorMode = "citation_pairs",
        judgments_path: str = "data/judgments_cleaned.json",
        par_to_par_path: str = "data/par-to-par-cleaned.csv",
        train_cutoff_year: int = 2018,
        top_k: int | None = None,
    ):
        self.retriever = retriever
        self.embeddings = embeddings
        self.mode: EvaluatorMode = mode
        self.judgments_path = judgments_path
        self.par_to_par_path = par_to_par_path
        self.train_cutoff_year = train_cutoff_year
        self.top_k = top_k

        # Data structures
        self.pid_to_text: NDArray[np.object_] | None = None
        self.celex_number_to_pid: dict[tuple[str, int], int] | None = None
        self.paragraph_dates: NDArray | None = None
        self.paragraph_celex: NDArray[np.object_] | None = None
        self.paragraph_number: NDArray[np.object_] | None = None
        self.paragraph_set: NDArray[np.object_] | None = None

        self.query_pids: list[int] | None = None
        self.query_texts: NDArray[np.object_] | None = None
        self.query_embeddings: NDArray | None = None
        self.qrel: dict[int, list[int]] | None = None

        self.map_score: float | None = None
        self.recall_scores: dict[int, float] | None = None
        self.map_ci: tuple[float, float] | None = None
        self.recall_cis: dict[int, tuple[float, float]] | None = None

    def load_and_prepare(self) -> None:
        self._load_paragraphs()
        self._load_citation_pairs()

        if self.mode == "citation_pairs":
            self._filter_to_citation_paragraphs()

    def _load_paragraphs(self) -> None:
        print("Loading judgments...")
        with open(self.judgments_path) as f:
            judgments = json.load(f)

        paragraphs = []
        for celex, judgment in tqdm(judgments.items(), desc="Processing judgments"):
            meta = judgment.get("meta", {})
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

        paragraphs.sort(key=lambda p: (p["celex"], p["number"]))

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

    def _load_citation_pairs(self) -> None:
        assert self.celex_number_to_pid is not None
        assert self.pid_to_text is not None

        print(f"Loading citation pairs from {self.par_to_par_path}...")

        query_texts_dict: dict[tuple[str, int], str] = {}
        qrel_dict: dict[tuple[str, int], list[tuple[str, int]]] = defaultdict(list)

        skipped = 0
        with open(self.par_to_par_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                celex_from = str(row["CELEX_FROM"])
                number_from = int(row["NUMBER_FROM"])
                text_from = str(row["TEXT_FROM"])
                celex_to = str(row["CELEX_TO"])
                number_to = int(row["NUMBER_TO"])

                query_key = (celex_from, number_from)
                doc_key = (celex_to, number_to)

                if (
                    query_key not in self.celex_number_to_pid
                    or doc_key not in self.celex_number_to_pid
                ):
                    skipped += 1
                    continue

                if query_key not in query_texts_dict:
                    query_texts_dict[query_key] = text_from

                qrel_dict[query_key].append(doc_key)

        if skipped > 0:
            print(f"Skipped {skipped} citation pairs (paragraphs not in index)")

        query_keys = sorted(query_texts_dict.keys())
        self.query_pids = [self.celex_number_to_pid[key] for key in query_keys]
        self.query_texts = np.array(
            [query_texts_dict[key] for key in query_keys], dtype=object
        )

        self.qrel = {}
        for query_key in query_keys:
            query_pid = self.celex_number_to_pid[query_key]
            doc_pids = [
                self.celex_number_to_pid[doc_key] for doc_key in qrel_dict[query_key]
            ]
            if doc_pids:
                self.qrel[query_pid] = doc_pids

        print(
            f"Loaded {len(self.query_pids)} queries with {sum(len(v) for v in self.qrel.values())} citation pairs"
        )

    def _filter_to_citation_paragraphs(self) -> None:
        assert self.qrel is not None
        assert self.pid_to_text is not None
        assert self.celex_number_to_pid is not None
        assert self.paragraph_dates is not None
        assert self.paragraph_celex is not None
        assert self.paragraph_number is not None
        assert self.paragraph_set is not None
        assert self.query_pids is not None
        assert self.query_texts is not None

        print("Filtering to citation-involved paragraphs...")

        citation_involved_pids: set[int] = set()
        for query_pid, cited_pids in self.qrel.items():
            citation_involved_pids.add(query_pid)
            citation_involved_pids.update(cited_pids)

        old_pids = sorted(citation_involved_pids)
        old_to_new_pid = {old_pid: new_pid for new_pid, old_pid in enumerate(old_pids)}

        self.pid_to_text = self.pid_to_text[old_pids]
        self.paragraph_dates = self.paragraph_dates[old_pids]
        self.paragraph_celex = self.paragraph_celex[old_pids]
        self.paragraph_number = self.paragraph_number[old_pids]
        self.paragraph_set = self.paragraph_set[old_pids]

        new_celex_number_to_pid = {}
        for (celex, number), old_pid in self.celex_number_to_pid.items():
            if old_pid in old_to_new_pid:
                new_celex_number_to_pid[(celex, number)] = old_to_new_pid[old_pid]
        self.celex_number_to_pid = new_celex_number_to_pid

        new_query_pids = []
        new_query_texts = []
        old_query_texts = self.query_texts
        for idx, old_pid in enumerate(self.query_pids):
            if old_pid in old_to_new_pid:
                new_query_pids.append(old_to_new_pid[old_pid])
                new_query_texts.append(old_query_texts[idx])
        self.query_pids = new_query_pids
        self.query_texts = np.array(new_query_texts, dtype=object)

        new_qrel: dict[int, list[int]] = {}
        for old_query_pid, old_cited_pids in self.qrel.items():
            if old_query_pid in old_to_new_pid:
                new_query_pid = old_to_new_pid[old_query_pid]
                new_cited_pids = [
                    old_to_new_pid[old_cited_pid]
                    for old_cited_pid in old_cited_pids
                    if old_cited_pid in old_to_new_pid
                ]
                if new_cited_pids:
                    new_qrel[new_query_pid] = new_cited_pids
        self.qrel = new_qrel

        print(
            f"Filtered from {len(citation_involved_pids)} to {len(old_pids)} citation-involved paragraphs"
        )

    def _embed_queries(self) -> None:
        assert self.query_texts is not None
        assert self.query_pids is not None
        assert self.paragraph_celex is not None
        assert self.paragraph_number is not None

        print("Embedding queries...")

        query_paragraph_ids = [
            (self.paragraph_celex[pid], int(self.paragraph_number[pid]))
            for pid in self.query_pids
        ]

        if hasattr(self.retriever, "transform_queries"):
            query_embeddings = self.retriever.transform_queries(
                self.query_texts, paragraph_ids=query_paragraph_ids
            )
        else:
            query_embeddings = self.retriever.transform(
                self.query_texts, paragraph_ids=query_paragraph_ids
            )

        self.query_embeddings = query_embeddings
        print(f"Query embeddings shape: {query_embeddings.shape}")

    def evaluate_iterative(
        self,
        k_values: list[int] = [5, 10, 50, 100],
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> tuple[float, dict[int, float]]:
        """Iterative evaluation: process queries chronologically, building index incrementally."""
        assert self.embeddings is not None
        assert self.pid_to_text is not None
        assert self.paragraph_set is not None
        assert self.qrel is not None
        assert self.query_pids is not None
        assert self.query_embeddings is not None
        assert self.paragraph_dates is not None

        # Get test queries with relevance judgments
        test_query_pids = [
            pid
            for pid in self.query_pids
            if self.paragraph_set[pid] == "test" and pid in self.qrel
        ]

        if not test_query_pids:
            self.map_score = 0.0
            self.recall_scores = {k: 0.0 for k in k_values}
            return self.map_score, self.recall_scores

        pid_to_query_idx = {pid: idx for idx, pid in enumerate(self.query_pids)}

        # Convert dates to timestamps for grouping
        all_times = self.paragraph_dates.astype("datetime64[s]").astype(np.int64)

        # Group all paragraphs by time
        time_to_doc_pids: dict[int, list[int]] = defaultdict(list)
        for pid in range(len(self.pid_to_text)):
            time_to_doc_pids[all_times[pid]].append(pid)

        # Group test queries by time
        time_to_query_pids: dict[int, list[int]] = defaultdict(list)
        for pid in test_query_pids:
            time_to_query_pids[all_times[pid]].append(pid)

        # Get all unique times and sort
        all_unique_times = sorted(
            set(time_to_doc_pids.keys()) | set(time_to_query_pids.keys())
        )

        # Create index
        emb_dim = self.embeddings.shape[1]
        self.retriever.create_index(emb_dim)

        # Results storage
        ranked_pids_list: list[NDArray] = []
        relevant_pids_list: list[NDArray] = []
        max_ranks: list[int] = []

        top_k = self.top_k if self.top_k else 10000
        metric_name = f"MAP@{self.top_k}" if self.top_k else "MAP"

        print(f"Processing {len(all_unique_times)} time groups chronologically...")
        pbar = tqdm(
            total=len(test_query_pids), desc=f"Iterative {metric_name} + Recall"
        )

        for t in all_unique_times:
            # First, process queries at this timestamp
            # They search against documents already in index (time < t)
            query_pids_at_t = time_to_query_pids.get(t, [])
            if query_pids_at_t:
                query_indices = [pid_to_query_idx[pid] for pid in query_pids_at_t]
                query_embs = self.query_embeddings[query_indices]

                # Search against accumulated index
                retrieved_pids, _ = self.retriever.search_index(query_embs, top_k)

                # Process results for each query
                for i, query_pid in enumerate(query_pids_at_t):
                    relevant_list = self.qrel.get(query_pid, [])
                    if not relevant_list:
                        pbar.update(1)
                        continue

                    relevant_array = np.array(relevant_list, dtype=np.int64)
                    ranked = retrieved_pids[i]

                    # Filter to valid (non-negative) results
                    valid_mask = ranked >= 0
                    ranked = ranked[valid_mask]

                    if len(ranked) == 0:
                        pbar.update(1)
                        continue

                    # Filter relevant to those with time < query time
                    relevant_in_candidates = relevant_array[
                        all_times[relevant_array] < t
                    ]

                    if len(relevant_in_candidates) == 0:
                        pbar.update(1)
                        continue

                    max_rank = min(len(ranked), top_k)

                    ranked_pids_list.append(ranked.astype(np.int64))
                    relevant_pids_list.append(relevant_in_candidates)
                    max_ranks.append(max_rank)

                    pbar.update(1)

            # Then, add documents with this timestamp to the index
            # (they become available as candidates for future queries)
            doc_pids_at_t = time_to_doc_pids.get(t, [])
            if doc_pids_at_t:
                doc_pids_array = np.array(doc_pids_at_t, dtype=np.int64)
                doc_embeddings = self.embeddings[doc_pids_array]
                self.retriever.add_to_index(doc_embeddings, doc_pids_array)

        pbar.close()
        self.retriever.reset_index()

        if not ranked_pids_list:
            self.map_score = 0.0
            self.recall_scores = {k: 0.0 for k in k_values}
            return self.map_score, self.recall_scores

        # Compute metrics using numba
        print("Computing metrics with numba...")
        k_values_array = np.array(k_values, dtype=np.int64)
        max_ranks_array = np.array(max_ranks, dtype=np.int64)

        avg_precs, recall_matrix = compute_metrics_batch(
            ranked_pids_list, relevant_pids_list, max_ranks_array, k_values_array
        )

        self.map_score = float(np.mean(avg_precs))
        self.map_ci = self._bootstrap_confidence_interval(
            avg_precs, confidence=confidence, n_bootstrap=n_bootstrap
        )
        self.recall_scores = {
            k: float(np.mean(recall_matrix[:, idx])) for idx, k in enumerate(k_values)
        }
        self.recall_cis = {
            k: self._bootstrap_confidence_interval(
                recall_matrix[:, idx], confidence=confidence, n_bootstrap=n_bootstrap
            )
            for idx, k in enumerate(k_values)
        }

        return self.map_score, self.recall_scores

    def _bootstrap_confidence_interval(
        self,
        values: NDArray,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> tuple[float, float]:
        if len(values) == 0:
            return 0.0, 0.0

        rng = np.random.default_rng()
        means = np.empty(n_bootstrap, dtype=np.float64)
        n = len(values)

        for i in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            sample = values[indices]
            means[i] = float(np.mean(sample))

        alpha = 1.0 - confidence
        lower = float(np.quantile(means, alpha / 2.0))
        upper = float(np.quantile(means, 1.0 - alpha / 2.0))
        return lower, upper

    def run(self) -> float:
        print(f"Mode: {self.mode}")
        print("Loading and preparing data...")
        self.load_and_prepare()

        assert self.pid_to_text is not None
        assert self.paragraph_set is not None
        assert self.query_pids is not None

        print(f"Unique paragraphs: {len(self.pid_to_text)}")
        print(f"Train paragraphs: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test paragraphs: {np.sum(self.paragraph_set == 'test')}")
        print(f"Total queries: {len(self.query_pids)}")

        assert self.paragraph_celex is not None
        assert self.paragraph_number is not None

        # Generate embeddings using retriever
        if self.embeddings is None:
            print("\nGenerating embeddings from retriever...")
            train_mask = self.paragraph_set == "train"

            paragraph_ids = [
                (self.paragraph_celex[pid], int(self.paragraph_number[pid]))
                for pid in range(len(self.pid_to_text))
            ]

            self.retriever.fit(self.pid_to_text, mask=train_mask)
            self.embeddings = self.retriever.transform(
                self.pid_to_text, paragraph_ids=paragraph_ids
            )
            print(f"Embeddings shape: {self.embeddings.shape}")

            if hasattr(self.retriever, "save_embeddings"):
                save_path = getattr(self.retriever, "save_embeddings_path", None)
                if save_path:
                    self.retriever.save_embeddings(self.embeddings, save_path)
        else:
            if len(self.embeddings) != len(self.pid_to_text):
                raise ValueError(
                    f"Embeddings size mismatch: got {len(self.embeddings)} embeddings "
                    f"but have {len(self.pid_to_text)} paragraphs. "
                    f"You must regenerate embeddings in '{self.mode}' mode."
                )

        self._embed_queries()

        metric_name = f"MAP@{self.top_k}" if self.top_k else "MAP"
        print(f"\nComputing {metric_name} and Recall@k with confidence intervals...")
        score, recall_scores = self.evaluate_iterative([5, 10, 100])

        map_ci = self.map_ci if self.map_ci is not None else (score, score)
        print(
            f"\n{metric_name}: {score:.3f} "
            f"(95% CI [{map_ci[0]:.3f}, {map_ci[1]:.3f}])"
        )

        for k, recall in sorted(recall_scores.items()):
            ci = self.recall_cis[k] if self.recall_cis is not None else (recall, recall)
            print(f"Recall@{k}: {recall:.3f} (95% CI [{ci[0]:.3f}, {ci[1]:.3f}])")

        return score


if __name__ == "__main__":
    from retrievers import DenseRetriever, TfidfRetriever, BOWRetriever

    # retriever = TfidfRetriever(
    #     stop_words="english",
    #     strip_accents="ascii",
    #     norm="l2",
    # )
    # retriever = BOWRetriever(
    #     lowercase=True,
    #     stop_words="english",
    # )
    retriever = DenseRetriever(
        preprocessed_dir="data/preprocessed",
    )

    evaluator = Evaluator(
        retriever=retriever,
        # mode="all_paragraphs",
        judgments_path="data/judgments_cleaned.json",
        par_to_par_path="data/par-to-par-cleaned.csv",
        train_cutoff_year=2018,
        top_k=10000,
    )

    evaluator.run()

import json
import os
import csv
from datetime import datetime as dt
from collections import defaultdict
from typing import Literal, Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm  # type: ignore
from numba import njit, prange  # type: ignore

from retrievers.base_retriever import BaseRetriever

# Type alias for evaluator modes
EvaluatorMode = Literal["citation_pairs", "all_paragraphs"]


@njit
def compute_ap_fast(
    ranked_pids: NDArray, relevant_pids: NDArray, max_rank: int
) -> float:
    """Numba-optimized Average Precision calculation."""
    num_rel = len(relevant_pids)
    if num_rel == 0:
        return 0.0

    precision_sum = 0.0
    num_found = 0

    # Use simple loop - numba will optimize this
    for i in range(min(max_rank, len(ranked_pids))):
        # Check if ranked_pids[i] is in relevant_pids
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
    """Numba-optimized Recall@k calculation."""
    num_rel = len(relevant_pids)
    if num_rel == 0:
        return 0.0

    num_found = 0

    for i in range(min(k, len(ranked_pids))):
        # Check if ranked_pids[i] is in relevant_pids
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
    """
    Compute MAP and Recall@k for a batch of queries in parallel.

    Returns:
        avg_precs: Array of average precision scores
        recall_matrix: Shape (n_queries, n_k_values) of recall scores
    """
    n_queries = len(ranked_pids_list)
    n_k = len(k_values)

    avg_precs = np.zeros(n_queries, dtype=np.float64)
    recall_matrix = np.zeros((n_queries, n_k), dtype=np.float64)

    for idx in prange(n_queries):
        ranked = ranked_pids_list[idx]
        relevant = relevant_pids_list[idx]
        max_rank = max_ranks[idx]

        # Compute AP
        avg_precs[idx] = compute_ap_fast(ranked, relevant, max_rank)

        # Compute Recall@k for all k values
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
        queries_path: str = "data/evaluation/queries.tsv",
        qrel_path: str = "data/evaluation/qrel.txt",
        train_cutoff_year: int = 2018,
        top_k: int | None = None,
        save_embeddings_path: str | None = None,
    ):
        self.retriever = retriever
        self.embeddings = embeddings
        self.mode: EvaluatorMode = mode
        self.judgments_path = judgments_path
        self.queries_path = queries_path
        self.qrel_path = qrel_path
        self.train_cutoff_year = train_cutoff_year
        self.top_k = top_k
        self.save_embeddings_path = save_embeddings_path

        # Validate mode
        if mode not in ["citation_pairs", "all_paragraphs"]:
            raise ValueError(
                f"Invalid mode: {mode}. Must be 'citation_pairs' or 'all_paragraphs'"
            )

        # Data structures (populated by load_and_prepare)
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

        self.sort_idx: NDArray | None = None
        self.sorted_dates: NDArray | None = None

        self.map_score: float | None = None
        self.recall_scores: dict[int, float] | None = None

    def load_and_prepare(self) -> None:
        """Load all data and prepare for evaluation."""
        # Load all paragraphs from judgments.json
        self._load_paragraphs()

        # Load queries and qrel
        self._load_queries_and_qrel()

        # Filter paragraphs for citation_pairs mode
        if self.mode == "citation_pairs":
            self._filter_to_citation_paragraphs()

        # Prepare temporal index
        self._prepare_temporal_index()

    def _load_paragraphs(self) -> None:
        """Load all paragraphs from judgments.json."""
        print("Loading judgments...")
        with open(self.judgments_path) as f:
            judgments = json.load(f)

        paragraphs = []
        for celex, judgment in tqdm(judgments.items(), desc="Processing judgments"):
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

    def _load_queries_and_qrel(self) -> None:
        """Load queries and qrel files."""
        assert self.celex_number_to_pid is not None

        print("Loading queries...")
        query_data = []
        with open(self.queries_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # Skip header
            for celex, par_num, query_text in reader:
                key = (celex, int(par_num))
                if key in self.celex_number_to_pid:
                    query_data.append((self.celex_number_to_pid[key], query_text))

        self.query_pids = [pid for pid, _ in query_data]
        self.query_texts = np.array([text for _, text in query_data], dtype=object)

        print("Loading qrel...")
        self.qrel = defaultdict(list)
        with open(self.qrel_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                query_id = parts[0]
                doc_id = parts[2]

                # Parse celex_paragraph_number format
                celex_q, par_num_q = query_id.rsplit("_", 1)
                celex_d, par_num_d = doc_id.rsplit("_", 1)

                query_key = (celex_q, int(par_num_q))
                doc_key = (celex_d, int(par_num_d))

                if (
                    query_key in self.celex_number_to_pid
                    and doc_key in self.celex_number_to_pid
                ):
                    query_pid = self.celex_number_to_pid[query_key]
                    doc_pid = self.celex_number_to_pid[doc_key]
                    self.qrel[query_pid].append(doc_pid)

        print(
            f"Loaded {len(self.query_pids)} queries with {sum(len(v) for v in self.qrel.values())} qrel entries"
        )

    def _filter_to_citation_paragraphs(self) -> None:
        """Filter paragraphs to only those involved in citations for citation_pairs mode."""
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

        # Collect all paragraphs involved in any citation
        citation_involved_pids: set[int] = set()
        for query_pid, cited_pids in self.qrel.items():
            citation_involved_pids.add(query_pid)
            citation_involved_pids.update(cited_pids)

        # Create sorted list of citation-involved pids
        old_pids = sorted(citation_involved_pids)

        # Create mapping from old pid to new pid
        old_to_new_pid = {old_pid: new_pid for new_pid, old_pid in enumerate(old_pids)}

        # Filter all arrays to only citation-involved paragraphs
        self.pid_to_text = self.pid_to_text[old_pids]
        self.paragraph_dates = self.paragraph_dates[old_pids]
        self.paragraph_celex = self.paragraph_celex[old_pids]
        self.paragraph_number = self.paragraph_number[old_pids]
        self.paragraph_set = self.paragraph_set[old_pids]

        # Update celex_number_to_pid mapping
        new_celex_number_to_pid = {}
        for (celex, number), old_pid in self.celex_number_to_pid.items():
            if old_pid in old_to_new_pid:
                new_celex_number_to_pid[(celex, number)] = old_to_new_pid[old_pid]
        self.celex_number_to_pid = new_celex_number_to_pid

        # Update query_pids and query_texts
        new_query_pids = []
        new_query_texts = []
        old_query_texts = self.query_texts
        for idx, old_pid in enumerate(self.query_pids):
            if old_pid in old_to_new_pid:
                new_query_pids.append(old_to_new_pid[old_pid])
                new_query_texts.append(old_query_texts[idx])
        self.query_pids = new_query_pids
        self.query_texts = np.array(new_query_texts, dtype=object)

        # Update qrel
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

    def _prepare_temporal_index(self) -> None:
        """Pre-sort paragraph IDs by date for temporal filtering."""
        assert self.paragraph_dates is not None
        self.sort_idx = np.argsort(self.paragraph_dates)
        self.sorted_dates = self.paragraph_dates[self.sort_idx]

    def _embed_queries(self) -> None:
        """Embed query texts using the retriever."""
        assert self.query_texts is not None
        query_texts = self.query_texts

        print("Embedding queries...")

        # Check if retriever has a special method for queries (e.g., GNN)
        if hasattr(self.retriever, "transform_queries"):
            query_embeddings = self.retriever.transform_queries(query_texts)
        else:
            # Standard transform for all other retrievers
            query_embeddings = self.retriever.transform(query_texts)

        self.query_embeddings = query_embeddings
        print(f"Query embeddings shape: {query_embeddings.shape}")

    def evaluate_map_and_recall(
        self, k_values: list[int] = [5, 10, 50, 100]
    ) -> tuple[float, dict[int, float]]:
        """Combined MAP and Recall evaluation using numba for speed."""
        assert self.embeddings is not None
        assert self.pid_to_text is not None
        assert self.paragraph_set is not None
        assert self.qrel is not None
        assert self.query_pids is not None
        assert self.query_embeddings is not None
        assert self.paragraph_dates is not None
        assert self.sorted_dates is not None
        assert self.sort_idx is not None

        # Create pid to query index mapping
        pid_to_query_idx = {pid: idx for idx, pid in enumerate(self.query_pids)}

        # Filter queries to test set only
        test_query_pids = [
            pid
            for pid in self.query_pids
            if self.paragraph_set[pid] == "test" and pid in self.qrel
        ]

        # Prepare batch data for numba processing
        ranked_pids_list: list[NDArray] = []
        relevant_pids_list: list[NDArray] = []
        max_ranks: list[int] = []

        desc = (
            f"Retrieving for MAP@{self.top_k} + Recall"
            if self.top_k
            else "Retrieving for MAP + Recall"
        )

        # First pass: retrieve and collect data
        for query_pid in tqdm(test_query_pids, desc=desc):
            query_date = self.paragraph_dates[query_pid]

            # Get all paragraphs strictly older than query
            cutoff = int(np.searchsorted(self.sorted_dates, query_date, side="left"))
            if cutoff == 0:
                continue

            # Candidate set: all paragraphs before the query
            cand_pids = self.sort_idx[:cutoff]

            if len(cand_pids) == 0:
                continue

            # Ground truth: relevant paragraphs that are also in candidate set
            relevant_list = self.qrel[query_pid]
            relevant_array = np.array(relevant_list, dtype=np.int64)

            # Fast intersection using numpy
            relevant_mask = np.isin(relevant_array, cand_pids)
            relevant_pids = relevant_array[relevant_mask]
            num_rel = len(relevant_pids)

            if num_rel == 0:
                continue

            # Get query embedding
            query_idx = pid_to_query_idx[query_pid]
            query_embedding = self.query_embeddings[query_idx]

            # Retrieve and rank candidates once
            ranked_pids = self.retriever.retrieve(
                query_embedding, self.embeddings, cand_pids, top_k=self.top_k
            )

            if len(ranked_pids) == 0:
                continue

            # Compute max rank
            max_rank = (
                len(ranked_pids)
                if self.top_k is None
                else min(len(ranked_pids), self.top_k)
            )

            # Store for batch processing
            ranked_pids_list.append(ranked_pids.astype(np.int64))
            relevant_pids_list.append(relevant_pids)
            max_ranks.append(max_rank)

        if not ranked_pids_list:
            self.map_score = 0.0
            self.recall_scores = {k: 0.0 for k in k_values}
            return self.map_score, self.recall_scores

        # Second pass: compute metrics in parallel using numba
        print("Computing metrics with numba...")
        k_values_array = np.array(k_values, dtype=np.int64)
        max_ranks_array = np.array(max_ranks, dtype=np.int64)

        avg_precs, recall_matrix = compute_metrics_batch(
            ranked_pids_list, relevant_pids_list, max_ranks_array, k_values_array
        )

        self.map_score = float(np.mean(avg_precs))
        self.recall_scores = {
            k: float(np.mean(recall_matrix[:, idx])) for idx, k in enumerate(k_values)
        }
        return self.map_score, self.recall_scores

    def load_embeddings(self, path: str | None = None) -> NDArray | None:
        """Load embeddings from disk using numpy's load format."""
        load_path = path or self.save_embeddings_path
        if load_path is None:
            return None

        if not os.path.exists(load_path):
            return None

        try:
            embeddings = np.load(load_path)
            print(f"Loaded embeddings from {load_path} (shape: {embeddings.shape})")
            return embeddings
        except Exception as e:
            print(f"Failed to load embeddings from {load_path}: {e}")
            return None

    def save_embeddings(self, path: str | None = None) -> None:
        """Save embeddings to disk using numpy's save format."""
        if self.embeddings is None:
            raise ValueError("No embeddings to save. Run evaluation first.")

        save_path = path or self.save_embeddings_path
        if save_path is None:
            return

        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        np.save(save_path, self.embeddings)
        print(f"Saved embeddings to {save_path} (shape: {self.embeddings.shape})")

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

        # Try to load embeddings if not provided
        if self.embeddings is None and self.save_embeddings_path:
            print("\nAttempting to load embeddings from disk...")
            loaded_embeddings = self.load_embeddings()
            if loaded_embeddings is not None:
                # Validate loaded embeddings match paragraph index
                if len(loaded_embeddings) != len(self.pid_to_text):
                    print(
                        f"Warning: Loaded embeddings size mismatch "
                        f"({len(loaded_embeddings)} vs {len(self.pid_to_text)}). "
                        f"Regenerating embeddings..."
                    )
                else:
                    self.embeddings = loaded_embeddings

        # Generate embeddings if still not available
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

        # Save embeddings if path is specified
        if self.save_embeddings_path:
            self.save_embeddings()

        # Embed queries using cleaned query texts
        self._embed_queries()

        metric_name = f"MAP@{self.top_k}" if self.top_k else "MAP"
        print(f"\nComputing {metric_name} and Recall@k...")
        score, recall_scores = self.evaluate_map_and_recall([5, 10, 100])

        print(f"\n{metric_name}: {score:.3f}")
        for k, recall in sorted(recall_scores.items()):
            print(f"Recall@{k}: {recall:.3f}")

        return score


if __name__ == "__main__":
    from retrievers import DenseRetriever, GNNRetriever, TfidfRetriever
    from example_gnn_usage import CitationGNN
    import torch
    from sentence_transformers import SentenceTransformer

    retriever = TfidfRetriever(
        stop_words="english",
        strip_accents="ascii",
        norm="l2",
    )

    # retriever = DenseRetriever(
    #     model_name="checkpoints/simcse_citation_model",
    #     max_seq_length=256,
    # )

    # encoding_model = "checkpoints/simcse_citation_model"
    # text_encoder = SentenceTransformer(encoding_model)

    # in_channels = text_encoder.get_sentence_embedding_dimension()

    # model = CitationGNN(
    #     in_channels, hidden_dim=512, output_dim=in_channels, num_layers=3
    # )

    # model.load_state_dict(torch.load("checkpoints/gnn/best_model.pt"))

    # retriever = GNNRetriever(
    #     gnn_model=model,
    #     # model_path="checkpoints/gnn/best_model.pt",
    #     text_encoder_name=encoding_model,
    #     batch_size=32,
    # )

    evaluator = Evaluator(
        retriever=retriever,
        mode="all_paragraphs",
        judgments_path="data/judgments_cleaned.json",
        queries_path="data/evaluation/queries_cleaned.tsv",
        qrel_path="data/evaluation/qrel.txt",
        train_cutoff_year=2018,
        top_k=10000,
        # save_embeddings_path="artifacts/simcse_embeddings.npy",
    )

    score = evaluator.run()
    print(f"Final MAP: {score:.3f}")

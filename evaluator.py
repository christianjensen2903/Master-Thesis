import json
import os
import re
from datetime import datetime as dt
from collections import defaultdict
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray
import pandas as pd  # type: ignore
from tqdm import tqdm  # type: ignore
from numba import njit, prange  # type: ignore

from data_loader import (
    load_citation_data,
    split_train_test,
    build_paragraph_index,
    build_citation_graph,
)
from retrievers.base_retriever import BaseRetriever
from retrievers.ltr_retriever import LTRRetriever

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
        csv_path: str = "data/par-to-par-cleaned.csv",
        metadata_path: str = "data/par-to-par.json",
        judgments_path: str = "data/judgments_cleaned.json",
        train_cutoff_year: int = 2018,
        top_k: int | None = None,
        save_embeddings_path: str | None = None,
    ):
        self.retriever = retriever
        self.embeddings = embeddings
        self.mode: EvaluatorMode = mode
        self.csv_path = csv_path
        self.metadata_path = metadata_path
        self.judgments_path = judgments_path
        self.train_cutoff_year = train_cutoff_year
        self.top_k = top_k
        self.save_embeddings_path = save_embeddings_path

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

    def evaluate_map_and_recall(
        self, k_values: list[int] = [5, 10, 50, 100]
    ) -> tuple[float, dict[int, float]]:
        """Combined MAP and Recall evaluation using numba for speed."""
        assert self.embeddings is not None
        assert self.pid_to_text is not None
        assert self.paragraph_set is not None
        assert self.cited_by_pid is not None
        assert self.paragraph_dates is not None
        assert self.sorted_dates is not None
        assert self.sort_idx is not None

        # Precompute test source PIDs using numpy for speed
        test_mask = self.paragraph_set == "test"
        test_pids = np.where(test_mask)[0]

        # Filter to only those with citations
        test_source_pids = [
            pid for pid in test_pids if len(self.cited_by_pid.get(int(pid), [])) > 0
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
        for src_pid in tqdm(test_source_pids, desc=desc):  # type: ignore
            src_date = self.paragraph_dates[src_pid]

            # Get all paragraphs strictly older than source
            cutoff = int(np.searchsorted(self.sorted_dates, src_date, side="left"))
            if cutoff == 0:
                continue

            cand_pids = self.sort_idx[:cutoff]

            # Ground truth: cited paragraphs that are also older
            relevant_list = self.cited_by_pid[int(src_pid)]
            relevant_array = np.array(relevant_list, dtype=np.int64)

            # Fast intersection using numpy
            relevant_mask = np.isin(relevant_array, cand_pids)
            relevant_pids = relevant_array[relevant_mask]
            num_rel = len(relevant_pids)

            if num_rel == 0:
                continue

            # Retrieve and rank candidates once
            ranked_pids = self.retriever.retrieve(
                int(src_pid), self.embeddings, cand_pids, top_k=self.top_k
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

        print(f"Unique paragraphs: {len(self.pid_to_text)}")
        print(f"Train paragraphs: {np.sum(self.paragraph_set == 'train')}")
        print(f"Test paragraphs: {np.sum(self.paragraph_set == 'test')}")

        # Set metadata arrays in LTR retriever if applicable

        if isinstance(self.retriever, LTRRetriever):
            assert self.paragraph_celex is not None
            assert self.paragraph_number is not None
            assert self.paragraph_dates is not None
            print("Setting metadata arrays in LTR retriever...")
            self.retriever.set_metadata_arrays(
                paragraph_celex=self.paragraph_celex,
                paragraph_number=self.paragraph_number,
                paragraph_dates=self.paragraph_dates,
            )

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
            # Pass citation graph to GNN retrievers for structure-aware embeddings
            fit_kwargs = {"mask": train_mask}
            if hasattr(self.retriever, "build_graph"):  # GNN retriever
                fit_kwargs["citation_graph"] = self.cited_by_pid
                fit_kwargs["paragraph_dates"] = self.paragraph_dates
            self.retriever.fit(self.pid_to_text, **fit_kwargs)
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

        if self.mode == "citation_pairs":
            assert self.df is not None
            print(f"Citation pairs: {len(self.df)}")

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

    retriever = DenseRetriever(
        model_name="checkpoints/simcse_citation_model",
        max_seq_length=256,
    )

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

    # retriever = TfidfRetriever(
    #     stop_words="english",
    #     strip_accents="ascii",
    #     norm="l2",
    # )

    evaluator = Evaluator(
        retriever=retriever,
        # mode="all_paragraphs",
        csv_path="data/par-to-par-cleaned.csv",
        metadata_path="data/par-to-par.json",
        judgments_path="data/judgments_cleaned.json",
        train_cutoff_year=2018,
        top_k=10000,
        save_embeddings_path="artifacts/simcse_embeddings.npy",
    )

    score = evaluator.run()
    print(f"Final MAP: {score:.3f}")

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer  # type: ignore
import faiss  # type: ignore

faiss.omp_set_num_threads(1)

from .base_retriever import BaseRetriever


class RecencyBoostedDenseRetriever(BaseRetriever):
    """Dense retriever with recency boosting.

    final_score = (alpha * semantic_score) + (beta * recency_score)

    where recency_score is computed based on how recent a document is
    relative to a reference date (typically the oldest document in the index).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        max_seq_length: int | None = None,
        preprocessed_dir: str | None = None,
        alpha: float = 1.0,
        beta: float = 0.0,
        recency_decay: str = "linear",  # "linear", "exponential", "log"
        decay_rate: float = 1.0,  # For exponential decay
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.preprocessed_dir = preprocessed_dir

        # Recency boosting parameters
        self.alpha = alpha
        self.beta = beta
        self.recency_decay = recency_decay
        self.decay_rate = decay_rate

        # Precomputed embeddings
        self.precomputed_doc_embeddings: NDArray | None = None
        self.precomputed_query_embeddings: NDArray | None = None
        self.par_id_to_idx: dict[str, int] | None = None
        self.par_metadata: list[dict] | None = None

        # FAISS index for iterative evaluation
        self._index: faiss.IndexFlatIP | None = None
        self._index_to_original: list[int] = []

        # Date information for recency boosting
        self._index_dates: list[int] = []  # Timestamps for indexed documents
        self._min_date: int | None = None  # Earliest date in index (for normalization)
        self._max_date: int | None = None  # Latest date in index (for normalization)

        # Document dates (set via set_document_dates)
        self._document_dates: NDArray | None = None

        if preprocessed_dir:
            self._load_precomputed_embeddings()
        else:
            self.model = SentenceTransformer(model_name)
            if max_seq_length is not None:
                self.model.max_seq_length = max_seq_length

        self._is_fitted = True

    def set_document_dates(self, dates: NDArray) -> None:
        """Set document dates for recency computation.

        Args:
            dates: Array of dates as datetime64[ns] or timestamps (int64)
        """
        if dates.dtype == np.dtype("datetime64[ns]"):
            self._document_dates = dates.astype("datetime64[s]").astype(np.int64)
        else:
            self._document_dates = dates.astype(np.int64)

    def _load_precomputed_embeddings(self) -> None:
        if self.preprocessed_dir is None:
            return

        preprocessed_path = Path(self.preprocessed_dir)

        doc_emb_path = preprocessed_path / "paragraph_embeddings_doc.npy"
        if not doc_emb_path.exists():
            raise FileNotFoundError(
                f"Precomputed document embeddings not found at {doc_emb_path}. "
                "Run precompute_embeddings.py first."
            )

        doc_embeddings = np.load(doc_emb_path)
        self.precomputed_doc_embeddings = doc_embeddings
        print(f"Loaded precomputed document embeddings: {doc_embeddings.shape}")

        query_emb_path = preprocessed_path / "paragraph_embeddings_query.npy"
        if query_emb_path.exists():
            query_embeddings = np.load(query_emb_path)
            self.precomputed_query_embeddings = query_embeddings
            print(f"Loaded precomputed query embeddings: {query_embeddings.shape}")

        metadata_path = preprocessed_path / "paragraph_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata: list[dict] = pickle.load(f)
            self.par_metadata = metadata
            self.par_id_to_idx = {m["id"]: i for i, m in enumerate(metadata)}
            print(f"Loaded metadata for {len(metadata)} paragraphs")

    def _get_paragraph_id(self, celex: str, number: int) -> str:
        return f"par:{celex}:{number}"

    def fit(self, texts: NDArray, mask: NDArray | None = None) -> None:
        if self.precomputed_doc_embeddings is not None:
            return

        if not hasattr(self, "model"):
            self.model = SentenceTransformer(self.model_name)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length

    def transform(
        self, texts: NDArray, paragraph_ids: list[tuple[str, int]] | None = None
    ) -> NDArray:
        if self.precomputed_doc_embeddings is not None and paragraph_ids is not None:
            if len(paragraph_ids) != len(texts):
                raise ValueError(
                    f"paragraph_ids length ({len(paragraph_ids)}) must match texts length ({len(texts)})"
                )

            if self.par_id_to_idx is None:
                raise ValueError("Metadata not loaded.")

            embedding_indices = []
            missing = []
            for celex, number in paragraph_ids:
                par_id = self._get_paragraph_id(celex, number)
                if par_id in self.par_id_to_idx:
                    embedding_indices.append(self.par_id_to_idx[par_id])
                else:
                    missing.append(par_id)

            if missing:
                print(
                    f"Warning: {len(missing)} paragraphs not found. Falling back to encoding."
                )
                return self._encode_texts(texts)

            return self.precomputed_doc_embeddings[embedding_indices]

        return self._encode_texts(texts)

    def _encode_texts(self, texts: NDArray) -> NDArray:
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(self.model_name)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings

    def transform_queries(
        self,
        query_texts: NDArray,
        paragraph_ids: list[tuple[str, int]] | None = None,
    ) -> NDArray:
        if self.precomputed_query_embeddings is not None and paragraph_ids is not None:
            if len(paragraph_ids) != len(query_texts):
                raise ValueError(
                    f"paragraph_ids length ({len(paragraph_ids)}) must match query_texts length ({len(query_texts)})"
                )

            if self.par_id_to_idx is None:
                raise ValueError("Metadata not loaded.")

            embedding_indices = []
            missing = []
            for celex, number in paragraph_ids:
                par_id = self._get_paragraph_id(celex, number)
                if par_id in self.par_id_to_idx:
                    embedding_indices.append(self.par_id_to_idx[par_id])
                else:
                    missing.append(par_id)

            if missing:
                print(
                    f"Warning: {len(missing)} queries not found. Falling back to encoding."
                )
                return self._encode_texts(query_texts)

            return self.precomputed_query_embeddings[embedding_indices]

        return self._encode_texts(query_texts)

    # --- Iterative index methods with recency boosting ---

    def create_index(self, dim: int) -> None:
        self._index = faiss.IndexFlatIP(dim)
        self._index_to_original = []
        self._index_dates = []
        self._min_date = None
        self._max_date = None

    def add_to_index(self, embeddings: NDArray, indices: NDArray) -> None:
        if self._index is None:
            raise RuntimeError("Index not created. Call create_index first.")

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self._index.add(embeddings)
        self._index_to_original.extend(indices.tolist())

        # Track dates for recency scoring
        if self._document_dates is not None:
            for idx in indices:
                date = int(self._document_dates[idx])
                self._index_dates.append(date)

                # Update min/max dates
                if self._min_date is None or date < self._min_date:
                    self._min_date = date
                if self._max_date is None or date > self._max_date:
                    self._max_date = date

    def _compute_recency_scores(
        self, doc_indices: NDArray, query_date: int | None = None
    ) -> NDArray:
        """Compute recency scores for documents.

        Args:
            doc_indices: FAISS indices (not original indices)
            query_date: Optional query timestamp. If provided, recency is
                       relative to query date (older than query = higher recency for more recent).
                       If None, uses index date range for normalization.

        Returns:
            Recency scores in [0, 1], where 1 = most recent
        """
        if self._min_date is None or self._max_date is None:
            return np.zeros(len(doc_indices), dtype=np.float32)

        date_range = self._max_date - self._min_date
        if date_range == 0:
            return np.ones(len(doc_indices), dtype=np.float32)

        # Get dates for each document
        doc_dates = np.array(
            [
                self._index_dates[idx] if idx >= 0 else self._min_date
                for idx in doc_indices
            ],
            dtype=np.float64,
        )

        if query_date is not None:
            # Recency relative to query date
            # Documents closer to query date get higher scores
            # All docs are older than query (temporal constraint)
            time_diff = query_date - doc_dates  # Always positive since docs are older
            max_diff = query_date - self._min_date
            if max_diff == 0:
                return np.ones(len(doc_indices), dtype=np.float32)

            normalized_age = time_diff / max_diff  # 0 = same as query, 1 = oldest doc
        else:
            # Recency relative to index date range
            normalized_age = (
                self._max_date - doc_dates
            ) / date_range  # 0 = newest, 1 = oldest

        if self.recency_decay == "linear":
            # Linear: newer = higher score
            recency_scores = 1.0 - normalized_age
        elif self.recency_decay == "exponential":
            # Exponential decay: sharper preference for recent docs
            recency_scores = np.exp(-self.decay_rate * normalized_age)
        elif self.recency_decay == "log":
            # Log decay: slower decay for older docs
            recency_scores = 1.0 - np.log1p(
                normalized_age * self.decay_rate
            ) / np.log1p(self.decay_rate)
        else:
            recency_scores = 1.0 - normalized_age

        return recency_scores.astype(np.float32)

    def search_index(
        self,
        query_embeddings: NDArray,
        top_k: int,
        query_dates: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """Search with recency boosting.

        Args:
            query_embeddings: Query vectors
            top_k: Number of results to return
            query_dates: Optional timestamps for each query (for relative recency)

        Returns:
            (indices, scores) where scores are the combined semantic + recency scores
        """
        if self._index is None or self._index.ntotal == 0:
            n_queries = len(query_embeddings)
            return np.full((n_queries, 0), -1, dtype=np.int64), np.zeros(
                (n_queries, 0), dtype=np.float32
            )

        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))

        # If beta > 0, retrieve more candidates for re-ranking
        if self.beta > 0 and self._document_dates is not None:
            # Retrieve more candidates to re-rank
            retrieval_k = min(top_k * 3, self._index.ntotal)
        else:
            retrieval_k = min(top_k, self._index.ntotal)

        semantic_scores, faiss_indices = self._index.search(
            query_embeddings, retrieval_k
        )

        # If no recency boosting, just return top-k
        if self.beta == 0 or self._document_dates is None:
            # Map to original indices and return top-k
            original_indices = np.array(
                [
                    [self._index_to_original[idx] if idx >= 0 else -1 for idx in row]
                    for row in faiss_indices
                ],
                dtype=np.int64,
            )
            return original_indices[:, :top_k], semantic_scores[:, :top_k]

        # Apply recency boosting
        n_queries = len(query_embeddings)
        result_indices = np.full((n_queries, top_k), -1, dtype=np.int64)
        result_scores = np.zeros((n_queries, top_k), dtype=np.float32)

        for i in range(n_queries):
            row_faiss_indices = faiss_indices[i]
            row_semantic_scores = semantic_scores[i]

            # Filter valid indices
            valid_mask = row_faiss_indices >= 0
            valid_faiss_indices = row_faiss_indices[valid_mask]
            valid_semantic_scores = row_semantic_scores[valid_mask]

            if len(valid_faiss_indices) == 0:
                continue

            # Get query date if available
            query_date = None
            if query_dates is not None and i < len(query_dates):
                query_date = int(query_dates[i])

            # Compute recency scores
            recency_scores = self._compute_recency_scores(
                valid_faiss_indices, query_date
            )

            # Normalize semantic scores to [0, 1] for this query
            sem_min, sem_max = valid_semantic_scores.min(), valid_semantic_scores.max()
            if sem_max > sem_min:
                normalized_semantic = (valid_semantic_scores - sem_min) / (
                    sem_max - sem_min
                )
            else:
                normalized_semantic = np.ones_like(valid_semantic_scores)

            # Combine scores
            combined_scores = (self.alpha * normalized_semantic) + (
                self.beta * recency_scores
            )

            # Re-rank by combined score
            rerank_order = np.argsort(-combined_scores)[:top_k]

            # Map to original indices
            for j, rerank_idx in enumerate(rerank_order):
                faiss_idx = valid_faiss_indices[rerank_idx]
                result_indices[i, j] = self._index_to_original[faiss_idx]
                result_scores[i, j] = combined_scores[rerank_idx]

        return result_indices, result_scores

    def reset_index(self) -> None:
        self._index = None
        self._index_to_original = []
        self._index_dates = []
        self._min_date = None
        self._max_date = None

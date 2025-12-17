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


class MetadataBoostedDenseRetriever(BaseRetriever):
    """Dense retriever with learned metadata boosts.

    final_score = (alpha * semantic_score) + (beta * recency_score)
                + (gamma * language_boost) + (delta * duration_boost)

    Language and duration boosts can be:
    - Pre-defined based on empirical analysis
    - Learned from training data
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        max_seq_length: int | None = None,
        preprocessed_dir: str | None = None,
        # Score combination weights
        alpha: float = 1.0,
        beta: float = 0.0,
        gamma: float = 0.0,  # Language boost weight
        delta: float = 0.0,  # Duration boost weight
        # Recency settings
        recency_decay: str = "exponential",
        decay_rate: float = 3.0,
        # Pre-defined language boosts (from empirical analysis)
        language_boosts: dict[str, float] | None = None,
        # Duration boost settings
        duration_peak_years: float = 2.5,  # Optimal duration (from analysis)
        duration_sigma: float = 1.0,  # Gaussian width
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.preprocessed_dir = preprocessed_dir

        # Score weights
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

        # Recency
        self.recency_decay = recency_decay
        self.decay_rate = decay_rate

        # Language boosts - empirical values from citation ratio analysis
        # Ratio > 1 = over-cited, < 1 = under-cited
        self._default_language_boosts = {
            "DAN": 1.32,
            "ENG": 1.21,
            "NLD": 1.16,
            "DEU": 1.07,
            "FRA": 1.00,
            "MLT": 0.99,
            "ITA": 0.98,
            "FIN": 0.96,
            "SWE": 0.93,
            "SPA": 0.84,
            "BUL": 0.83,
            "HUN": 0.80,
            "SLK": 0.79,
            "ELL": 0.71,
            "POR": 0.70,
            "SLV": 0.62,
            "RON": 0.60,
            "POL": 0.57,
            "LAV": 0.50,
            "LIT": 0.42,
            "CES": 0.42,
            "EST": 0.40,
            "HRV": 0.23,
            "GLE": 0.00,
        }
        self.language_boosts = language_boosts or self._default_language_boosts

        # Duration boost (Gaussian around optimal duration)
        self.duration_peak_years = duration_peak_years
        self.duration_sigma = duration_sigma

        # Precomputed embeddings
        self.precomputed_doc_embeddings: NDArray | None = None
        self.precomputed_query_embeddings: NDArray | None = None
        self.par_id_to_idx: dict[str, int] | None = None
        self.par_metadata: list[dict] | None = None

        # FAISS index
        self._index: faiss.IndexFlatIP | None = None
        self._index_to_original: list[int] = []

        # Metadata arrays for indexed documents
        self._index_dates: list[int] = []
        self._index_languages: list[str] = []
        self._index_durations: list[float] = []  # Duration in years

        self._min_date: int | None = None
        self._max_date: int | None = None

        # Document metadata (set via setters)
        self._document_dates: NDArray | None = None
        self._document_languages: NDArray | None = None
        self._document_durations: NDArray | None = None

        if preprocessed_dir:
            self._load_precomputed_embeddings()
        else:
            self.model = SentenceTransformer(model_name)
            if max_seq_length is not None:
                self.model.max_seq_length = max_seq_length

        self._is_fitted = True

    def set_document_dates(self, dates: NDArray) -> None:
        if dates.dtype == np.dtype("datetime64[ns]"):
            self._document_dates = dates.astype("datetime64[s]").astype(np.int64)
        else:
            self._document_dates = dates.astype(np.int64)

    def set_document_languages(self, languages: NDArray) -> None:
        """Set language for each document (3-letter codes like 'ENG', 'DEU')."""
        self._document_languages = np.asarray(languages, dtype=object)

    def set_document_durations(self, durations: NDArray) -> None:
        """Set case duration in years for each document."""
        self._document_durations = np.asarray(durations, dtype=np.float32)

    def _load_precomputed_embeddings(self) -> None:
        if self.preprocessed_dir is None:
            return

        preprocessed_path = Path(self.preprocessed_dir)

        doc_emb_path = preprocessed_path / "paragraph_embeddings_doc.npy"
        if not doc_emb_path.exists():
            raise FileNotFoundError(
                f"Precomputed doc embeddings not found at {doc_emb_path}"
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
                raise ValueError("paragraph_ids length must match texts length")

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
                raise ValueError("paragraph_ids length must match query_texts length")

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

    # --- Index methods ---

    def create_index(self, dim: int) -> None:
        self._index = faiss.IndexFlatIP(dim)
        self._index_to_original = []
        self._index_dates = []
        self._index_languages = []
        self._index_durations = []
        self._min_date = None
        self._max_date = None

    def add_to_index(self, embeddings: NDArray, indices: NDArray) -> None:
        if self._index is None:
            raise RuntimeError("Index not created. Call create_index first.")

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self._index.add(embeddings)
        self._index_to_original.extend(indices.tolist())

        for idx in indices:
            # Track dates
            if self._document_dates is not None:
                date = int(self._document_dates[idx])
                self._index_dates.append(date)
                if self._min_date is None or date < self._min_date:
                    self._min_date = date
                if self._max_date is None or date > self._max_date:
                    self._max_date = date
            else:
                self._index_dates.append(0)

            # Track languages
            if self._document_languages is not None:
                self._index_languages.append(str(self._document_languages[idx]))
            else:
                self._index_languages.append("")

            # Track durations
            if self._document_durations is not None:
                self._index_durations.append(float(self._document_durations[idx]))
            else:
                self._index_durations.append(0.0)

    def _compute_recency_scores(
        self, doc_indices: NDArray, query_date: int | None = None
    ) -> NDArray:
        if self._min_date is None or self._max_date is None:
            return np.zeros(len(doc_indices), dtype=np.float32)

        date_range = self._max_date - self._min_date
        if date_range == 0:
            return np.ones(len(doc_indices), dtype=np.float32)

        doc_dates = np.array(
            [
                self._index_dates[idx] if idx >= 0 else self._min_date
                for idx in doc_indices
            ],
            dtype=np.float64,
        )

        if query_date is not None:
            time_diff = query_date - doc_dates
            max_diff = query_date - self._min_date
            if max_diff == 0:
                return np.ones(len(doc_indices), dtype=np.float32)
            normalized_age = time_diff / max_diff
        else:
            normalized_age = (self._max_date - doc_dates) / date_range

        if self.recency_decay == "linear":
            recency_scores = 1.0 - normalized_age
        elif self.recency_decay == "exponential":
            recency_scores = np.exp(-self.decay_rate * normalized_age)
        elif self.recency_decay == "log":
            recency_scores = 1.0 - np.log1p(
                normalized_age * self.decay_rate
            ) / np.log1p(self.decay_rate)
        else:
            recency_scores = 1.0 - normalized_age

        return recency_scores.astype(np.float32)

    def _compute_language_scores(self, doc_indices: NDArray) -> NDArray:
        """Compute language boost scores based on citation bias."""
        scores = np.zeros(len(doc_indices), dtype=np.float32)

        for i, idx in enumerate(doc_indices):
            if idx >= 0 and idx < len(self._index_languages):
                lang = self._index_languages[idx]
                # Normalize: under-cited languages get boost, over-cited get penalty
                # Convert ratio to boost: log(ratio) maps ratio=1->0, ratio>1->negative, ratio<1->positive
                ratio = self.language_boosts.get(lang, 1.0)
                # Invert so under-cited (low ratio) get higher score
                scores[i] = 1.0 / ratio if ratio > 0 else 1.0

        # Normalize to [0, 1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())

        return scores

    def _compute_duration_scores(self, doc_indices: NDArray) -> NDArray:
        """Compute duration boost using Gaussian centered on optimal duration."""
        scores = np.zeros(len(doc_indices), dtype=np.float32)

        for i, idx in enumerate(doc_indices):
            if idx >= 0 and idx < len(self._index_durations):
                duration = self._index_durations[idx]
                # Gaussian boost: cases near peak duration get highest score
                diff = duration - self.duration_peak_years
                scores[i] = np.exp(-0.5 * (diff / self.duration_sigma) ** 2)

        return scores

    def search_index(
        self,
        query_embeddings: NDArray,
        top_k: int,
        query_dates: NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        if self._index is None or self._index.ntotal == 0:
            n_queries = len(query_embeddings)
            return np.full((n_queries, 0), -1, dtype=np.int64), np.zeros(
                (n_queries, 0), dtype=np.float32
            )

        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))

        # Retrieve more candidates if we're re-ranking
        needs_rerank = self.beta > 0 or self.gamma > 0 or self.delta > 0
        if needs_rerank:
            retrieval_k = min(top_k * 3, self._index.ntotal)
        else:
            retrieval_k = min(top_k, self._index.ntotal)

        semantic_scores, faiss_indices = self._index.search(
            query_embeddings, retrieval_k
        )

        # If no boosting needed, return directly
        if not needs_rerank:
            original_indices = np.array(
                [
                    [self._index_to_original[idx] if idx >= 0 else -1 for idx in row]
                    for row in faiss_indices
                ],
                dtype=np.int64,
            )
            return original_indices[:, :top_k], semantic_scores[:, :top_k]

        # Apply all boosts and re-rank
        n_queries = len(query_embeddings)
        result_indices = np.full((n_queries, top_k), -1, dtype=np.int64)
        result_scores = np.zeros((n_queries, top_k), dtype=np.float32)

        for i in range(n_queries):
            row_faiss_indices = faiss_indices[i]
            row_semantic_scores = semantic_scores[i]

            valid_mask = row_faiss_indices >= 0
            valid_faiss_indices = row_faiss_indices[valid_mask]
            valid_semantic_scores = row_semantic_scores[valid_mask]

            if len(valid_faiss_indices) == 0:
                continue

            # Normalize semantic scores
            sem_min, sem_max = valid_semantic_scores.min(), valid_semantic_scores.max()
            if sem_max > sem_min:
                normalized_semantic = (valid_semantic_scores - sem_min) / (
                    sem_max - sem_min
                )
            else:
                normalized_semantic = np.ones_like(valid_semantic_scores)

            # Compute all boost scores
            combined_scores = self.alpha * normalized_semantic

            if self.beta > 0:
                query_date = (
                    int(query_dates[i])
                    if query_dates is not None and i < len(query_dates)
                    else None
                )
                recency_scores = self._compute_recency_scores(
                    valid_faiss_indices, query_date
                )
                combined_scores += self.beta * recency_scores

            if self.gamma > 0:
                language_scores = self._compute_language_scores(valid_faiss_indices)
                combined_scores += self.gamma * language_scores

            if self.delta > 0:
                duration_scores = self._compute_duration_scores(valid_faiss_indices)
                combined_scores += self.delta * duration_scores

            # Re-rank
            rerank_order = np.argsort(-combined_scores)[:top_k]

            for j, rerank_idx in enumerate(rerank_order):
                faiss_idx = valid_faiss_indices[rerank_idx]
                result_indices[i, j] = self._index_to_original[faiss_idx]
                result_scores[i, j] = combined_scores[rerank_idx]

        return result_indices, result_scores

    def reset_index(self) -> None:
        self._index = None
        self._index_to_original = []
        self._index_dates = []
        self._index_languages = []
        self._index_durations = []
        self._min_date = None
        self._max_date = None

    def get_boost_params(self) -> dict[str, float]:
        """Return current boost parameters for optimization."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "duration_peak_years": self.duration_peak_years,
            "duration_sigma": self.duration_sigma,
        }

    def set_boost_params(
        self,
        alpha: float | None = None,
        beta: float | None = None,
        gamma: float | None = None,
        delta: float | None = None,
        duration_peak_years: float | None = None,
        duration_sigma: float | None = None,
    ) -> None:
        """Update boost parameters (for hyperparameter tuning)."""
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        if gamma is not None:
            self.gamma = gamma
        if delta is not None:
            self.delta = delta
        if duration_peak_years is not None:
            self.duration_peak_years = duration_peak_years
        if duration_sigma is not None:
            self.duration_sigma = duration_sigma

import numpy as np
from numpy.typing import NDArray
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from scipy.sparse import csr_matrix, vstack  # type: ignore

from .base_retriever import BaseRetriever


class TfidfRetriever(BaseRetriever):
    """TF-IDF based retriever with iterative evaluation support."""

    def __init__(self, **tfidf_kwargs):
        if not tfidf_kwargs:
            tfidf_kwargs = {
                "stop_words": "english",
                "strip_accents": "ascii",
                "norm": "l2",
            }
        self.vectorizer = TfidfVectorizer(**tfidf_kwargs)
        self._is_fitted = False

        # Index state for iterative evaluation
        self._index_matrix: csr_matrix | None = None
        self._index_to_original: list[int] = []

    def fit(self, texts: NDArray, mask: NDArray | None = None) -> None:
        fit_texts = texts[mask] if mask is not None else texts
        self.vectorizer.fit(fit_texts)
        self._is_fitted = True

    def transform(
        self, texts: NDArray, paragraph_ids: list[tuple[str, int]] | None = None
    ) -> csr_matrix:
        if not self._is_fitted:
            raise RuntimeError("Retriever must be fitted before transform")
        return self.vectorizer.transform(texts)

    # --- Iterative index methods ---

    def create_index(self, dim: int) -> None:
        self._index_matrix = None
        self._index_to_original = []

    def add_to_index(self, embeddings: NDArray | csr_matrix, indices: NDArray) -> None:
        if not isinstance(embeddings, csr_matrix):
            embeddings = csr_matrix(embeddings)

        if self._index_matrix is None:
            self._index_matrix = embeddings
        else:
            self._index_matrix = vstack([self._index_matrix, embeddings])

        self._index_to_original.extend(indices.tolist())

    def search_index(
        self, query_embeddings: NDArray | csr_matrix, top_k: int
    ) -> tuple[NDArray, NDArray]:
        if self._index_matrix is None or self._index_matrix.shape[0] == 0:
            n_queries = (
                query_embeddings.shape[0]
                if hasattr(query_embeddings, "shape")
                else len(query_embeddings)
            )
            return np.full((n_queries, 0), -1, dtype=np.int64), np.zeros(
                (n_queries, 0), dtype=np.float32
            )

        if not isinstance(query_embeddings, csr_matrix):
            query_embeddings = csr_matrix(query_embeddings)

        # Compute all similarities at once
        similarities = query_embeddings.dot(self._index_matrix.T).toarray()

        n_queries = similarities.shape[0]
        n_candidates = similarities.shape[1]
        k = min(top_k, n_candidates)

        # Get top-k for each query
        if k < n_candidates:
            # Use argpartition for efficiency
            top_k_local = np.argpartition(-similarities, k, axis=1)[:, :k]
            # Sort the top-k
            row_indices = np.arange(n_queries)[:, None]
            top_k_scores = similarities[row_indices, top_k_local]
            sorted_order = np.argsort(-top_k_scores, axis=1)
            top_k_local = np.take_along_axis(top_k_local, sorted_order, axis=1)
            scores = np.take_along_axis(top_k_scores, sorted_order, axis=1)
        else:
            sorted_order = np.argsort(-similarities, axis=1)
            top_k_local = sorted_order[:, :k]
            scores = np.take_along_axis(similarities, top_k_local, axis=1)

        # Map to original indices
        original_indices = np.array(
            [[self._index_to_original[idx] for idx in row] for row in top_k_local],
            dtype=np.int64,
        )

        return original_indices, scores.astype(np.float32)

    def reset_index(self) -> None:
        self._index_matrix = None
        self._index_to_original = []

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore

from .base_retriever import BaseRetriever
from .tfidf_retriever import TfidfRetriever
from .dense_retriever import DenseRetriever


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining TF-IDF and dense retriever scores."""

    def __init__(
        self,
        td_idf_retriever: TfidfRetriever,
        dense_retriever: DenseRetriever,
        tfidf_weight: float = 0.5,
        dense_weight: float = 0.5,
    ):
        if abs(tfidf_weight + dense_weight - 1.0) > 1e-6:
            raise ValueError("tfidf_weight and dense_weight must sum to 1.0")

        self.tfidf_weight = tfidf_weight
        self.dense_weight = dense_weight

        self.tfidf_retriever = td_idf_retriever
        self.dense_retriever = dense_retriever

        self._is_fitted = False
        self._tfidf_embeddings: csr_matrix | None = None
        self._dense_embeddings: np.ndarray | None = None

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        self.tfidf_retriever.fit(texts, mask)
        self.dense_retriever.fit(texts, mask)
        self._is_fitted = True

    def transform(self, texts: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Retriever must be fitted before transform")

        self._tfidf_embeddings = self.tfidf_retriever.transform(texts)
        self._dense_embeddings = self.dense_retriever.transform(texts)

        # Return dense embeddings for shape compatibility with evaluator
        # The actual retrieval will use both embeddings stored internally
        return self._dense_embeddings

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        if self._tfidf_embeddings is None or self._dense_embeddings is None:
            raise RuntimeError(
                "Embeddings not available. Call transform() before retrieve()"
            )

        # Compute TF-IDF similarities
        query_tfidf = self._tfidf_embeddings[query_idx]
        candidate_tfidf = self._tfidf_embeddings[candidate_indices]
        tfidf_similarities = candidate_tfidf.dot(query_tfidf.T).toarray().ravel()

        # Compute dense similarities
        query_dense = self._dense_embeddings[query_idx]
        candidate_dense = self._dense_embeddings[candidate_indices]

        if self.dense_retriever.normalize_embeddings:
            dense_similarities = candidate_dense @ query_dense
        else:
            query_norm = np.linalg.norm(query_dense)
            candidate_norms = np.linalg.norm(candidate_dense, axis=1)
            dense_similarities = (candidate_dense @ query_dense) / (
                candidate_norms * query_norm + 1e-8
            )

        # Combine scores with weighted average
        combined_similarities = (
            self.tfidf_weight * tfidf_similarities
            + self.dense_weight * dense_similarities
        )

        # Use efficient top-k selection if requested
        if top_k is not None and top_k < len(combined_similarities):
            top_k_indices = np.argpartition(-combined_similarities, top_k)[:top_k]
            sorted_top_k = top_k_indices[
                np.argsort(-combined_similarities[top_k_indices])
            ]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-combined_similarities)
            return candidate_indices[ranked_order]

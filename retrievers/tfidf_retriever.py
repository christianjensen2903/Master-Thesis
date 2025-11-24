import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore

from .base_retriever import BaseRetriever


class TfidfRetriever(BaseRetriever):
    """TF-IDF based retriever using cosine similarity."""

    def __init__(self, **tfidf_kwargs):
        if not tfidf_kwargs:
            tfidf_kwargs = {
                "stop_words": "english",
                "strip_accents": "ascii",
                "norm": "l2",  # For cosine similarity via dot product
            }
        self.vectorizer = TfidfVectorizer(**tfidf_kwargs)
        self._is_fitted = False

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        if mask is not None:
            fit_texts = texts[mask]
        else:
            fit_texts = texts

        self.vectorizer.fit(fit_texts)
        self._is_fitted = True

    def transform(
        self, texts: np.ndarray, paragraph_ids: list[tuple[str, int]] | None = None
    ) -> csr_matrix:
        if not self._is_fitted:
            raise RuntimeError("Retriever must be fitted before transform")

        return self.vectorizer.transform(texts)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        embeddings: csr_matrix,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        candidate_vecs = embeddings[candidate_indices]

        # Cosine similarity via dot product (vectors are l2-normalized)
        similarities = candidate_vecs.dot(query_embedding.T).toarray().ravel()

        # Use efficient top-k selection if requested
        if top_k is not None and top_k < len(similarities):
            # argpartition is O(n) vs argsort O(n log n)
            # Get indices of top_k largest values (unsorted)
            top_k_indices = np.argpartition(-similarities, top_k)[:top_k]
            # Sort only the top_k values
            sorted_top_k = top_k_indices[np.argsort(-similarities[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            # Full sort by similarity (high to low)
            ranked_order = np.argsort(-similarities)
            return candidate_indices[ranked_order]

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore

from .base_retriever import BaseRetriever


class VerbatimRetriever(BaseRetriever):
    """Naive baseline retriever that ranks by n-gram overlap (verbatim text matching).

    Uses efficient sparse matrix operations for fast computation.
    """

    def __init__(self, ngram_range: tuple[int, int] = (3, 5), min_df: int = 1):
        """
        Args:
            ngram_range: Range of n-gram sizes (min, max) for word-level n-grams
            min_df: Minimum document frequency for n-grams
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.vectorizer = CountVectorizer(
            ngram_range=ngram_range,
            lowercase=True,
            min_df=min_df,
            binary=True,  # Binary presence/absence for Jaccard-like similarity
        )
        self._is_fitted = False

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        if mask is not None:
            fit_texts = texts[mask]
        else:
            fit_texts = texts

        self.vectorizer.fit(fit_texts)
        self._is_fitted = True

    def transform(self, texts: np.ndarray) -> csr_matrix:
        if not self._is_fitted:
            raise RuntimeError("Retriever must be fitted before transform")

        return self.vectorizer.transform(texts)

    def retrieve(
        self,
        query_idx: int,
        embeddings: csr_matrix,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        query_vec = embeddings[query_idx]
        candidate_vecs = embeddings[candidate_indices]

        # Compute Jaccard similarity: intersection / union
        # For binary vectors: Jaccard = (A ∩ B) / (A ∪ B) = (A·B) / (||A||₁ + ||B||₁ - A·B)
        intersection = candidate_vecs.dot(query_vec.T).toarray().ravel()

        query_sum = query_vec.sum()
        candidate_sums = np.array(candidate_vecs.sum(axis=1)).ravel()

        union = query_sum + candidate_sums - intersection

        # Avoid division by zero
        similarities = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=float),
            where=union != 0,
        )

        # Rank candidates by similarity
        if top_k is not None and top_k < len(similarities):
            top_k_indices = np.argpartition(-similarities, top_k)[:top_k]
            sorted_top_k = top_k_indices[np.argsort(-similarities[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-similarities)
            return candidate_indices[ranked_order]

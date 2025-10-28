import numpy as np
from sklearn.feature_extraction.text import CountVectorizer  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore

from .base_retriever import BaseRetriever


class LexicalRetriever(BaseRetriever):
    """Simple lexical overlap baseline using bag-of-words Jaccard similarity.

    Ranks candidates by the proportion of shared unique words (unigrams only).
    This is even simpler than n-gram overlap - just word-level matching.
    """

    def __init__(self, lowercase: bool = True, stop_words: str | None = None):
        """
        Args:
            lowercase: Whether to lowercase text
            stop_words: Stop words to filter ('english' or None)
        """
        self.vectorizer = CountVectorizer(
            ngram_range=(1, 1),  # Unigrams only
            lowercase=lowercase,
            stop_words=stop_words,
            binary=True,  # Binary presence/absence for Jaccard
            min_df=1,
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

        # Compute Jaccard similarity at word level
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

import numpy as np
import bm25s  # type: ignore

from .base_retriever import BaseRetriever


class BM25Retriever(BaseRetriever):
    """BM25 retriever that builds index on whole corpus and filters candidates during retrieval."""

    def __init__(self, tokenizer=None, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            tokenizer: Optional function to tokenize text. If None, uses simple split.
            k1: BM25 k1 parameter (term frequency saturation)
            b: BM25 b parameter (length normalization)
        """
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer
        self.k1 = k1
        self.b = b
        self.bm25: bm25s.BM25 | None = None
        self.tokenized_corpus: list[list[str]] | None = None
        self._is_fitted = False

    def _default_tokenizer(self, text: str) -> list[str]:
        """Default tokenizer that splits on whitespace and lowercases."""
        return text.lower().split()

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        """
        Fit BM25 on the entire corpus (ignores mask for indexing).
        Index is built on all texts.
        """
        # Tokenize all texts (build index on whole corpus)
        self.tokenized_corpus = [self.tokenizer(text) for text in texts]
        self.bm25 = bm25s.BM25(k1=self.k1, b=self.b, method="bm25l")
        self.bm25.index(self.tokenized_corpus, show_progress=True)
        self._is_fitted = True

    def transform(self, texts: np.ndarray) -> np.ndarray:
        """
        BM25 doesn't produce embeddings, so this returns a dummy array.
        The retrieve method works directly with tokenized texts.
        """
        if not self._is_fitted:
            raise RuntimeError("Retriever must be fitted before transform")
        # Return dummy array - not used by retrieve method
        return np.zeros((len(texts), 1))

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """
        Retrieve and rank candidates using BM25.
        Filters to only valid candidates during retrieval.

        Args:
            query_idx: Index of the query paragraph
            embeddings: Ignored (BM25 doesn't use embeddings)
            candidate_indices: Indices of candidate paragraphs to rank
            top_k: If provided, only return top k results
        """
        if not self._is_fitted:
            raise RuntimeError("Retriever must be fitted before retrieve")
        if self.bm25 is None or self.tokenized_corpus is None:
            raise RuntimeError("BM25 not properly initialized")

        # Get query text tokenized
        query_tokens = self.tokenized_corpus[query_idx]

        # Filter to valid candidates (those in candidate_indices)
        # Get scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Filter scores to only candidates
        candidate_scores = scores[candidate_indices]

        # Use efficient top-k selection if requested
        if top_k is not None and top_k < len(candidate_scores):
            top_k_indices = np.argpartition(-candidate_scores, top_k)[:top_k]
            sorted_top_k = top_k_indices[np.argsort(-candidate_scores[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-candidate_scores)
            return candidate_indices[ranked_order]

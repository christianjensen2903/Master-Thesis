import numpy as np
from numpy.typing import NDArray
import bm25s  # type: ignore
import Stemmer  # type: ignore


class BM25Retriever:
    """BM25 retriever using bm25s for fast retrieval. Rebuilds index on each search."""

    def __init__(
        self, stemmer_language: str = "english", k1: float = 1.5, b: float = 0.75
    ):
        self.stemmer = Stemmer.Stemmer(stemmer_language)
        self.k1 = k1
        self.b = b

        # Text references (set via set_corpus)
        self._pid_to_text: NDArray | None = None
        self._query_texts: NDArray | None = None

        # Index state
        self._corpus_texts: list[str] = []
        self._index_to_original: list[int] = []
        self._bm25: bm25s.BM25 | None = None

    def set_corpus(self, pid_to_text: NDArray, query_texts: NDArray) -> None:
        """Set text references for document and query lookup."""
        self._pid_to_text = pid_to_text
        self._query_texts = query_texts

    def fit(self, texts: NDArray, mask: NDArray | None = None) -> None:
        """Fit is a no-op for BM25 - we rebuild each time."""
        pass

    def transform(
        self, texts: NDArray, paragraph_ids: list[tuple[str, int]] | None = None
    ) -> NDArray:
        """Returns indices as placeholders (BM25 works directly with texts)."""
        return np.arange(len(texts))

    def create_index(self, dim: int) -> None:
        """Reset index state."""
        self._corpus_texts = []
        self._index_to_original = []
        self._bm25 = None

    def add_to_index(self, embeddings: NDArray, indices: NDArray) -> None:
        """Add documents to the index. Uses indices to look up texts from pid_to_text."""
        if self._pid_to_text is not None:
            texts = [str(self._pid_to_text[idx]) for idx in indices]
        else:
            texts = [str(e) for e in embeddings]

        self._corpus_texts.extend(texts)
        self._index_to_original.extend(indices.tolist())

    def _build_index(self) -> None:
        """Build the BM25 index from stored texts."""
        if not self._corpus_texts:
            self._bm25 = None
            return
        corpus_tokens = bm25s.tokenize(
            self._corpus_texts, stemmer=self.stemmer, show_progress=False
        )
        self._bm25 = bm25s.BM25(k1=self.k1, b=self.b)
        self._bm25.index(corpus_tokens, show_progress=False)

    def search_index(self, query_texts: NDArray, top_k: int) -> tuple[NDArray, NDArray]:
        """Search the index. Rebuilds BM25 index before each search.

        query_texts should be an array of query text strings.
        """
        n_queries = len(query_texts)

        if not self._corpus_texts:
            return np.full((n_queries, 0), -1, dtype=np.int64), np.zeros(
                (n_queries, 0), dtype=np.float32
            )

        # Rebuild index
        self._build_index()

        # Convert to list of strings
        queries = [str(q) for q in query_texts]

        # Tokenize queries
        query_tokens = bm25s.tokenize(
            queries, stemmer=self.stemmer, show_progress=False
        )

        # Search
        k = min(top_k, len(self._corpus_texts))
        assert self._bm25 is not None
        results, scores = self._bm25.retrieve(query_tokens, k=k, show_progress=False)

        # Map to original indices
        original_indices = np.array(
            [[self._index_to_original[idx] for idx in row] for row in results],
            dtype=np.int64,
        )

        return original_indices, scores.astype(np.float32)

    def reset_index(self) -> None:
        self._corpus_texts = []
        self._index_to_original = []
        self._bm25 = None

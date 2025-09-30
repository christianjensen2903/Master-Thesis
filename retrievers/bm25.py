from __future__ import annotations

"""BM25 retriever implementation with batch querying.

Uses `rank_bm25.BM25Okapi` over a tokenized corpus and scores each query
efficiently. Mirrors the interface of `TFIDFRetriever`.
"""

from typing import Callable, Sequence
import logging

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore
from tqdm import tqdm  # type: ignore

from langchain_core.documents import Document

from .base import BaseRetriever


logger = logging.getLogger(__name__)


def _default_tokenizer(text: str) -> list[str]:
    """A simple, lowercase whitespace tokenizer.

    Parameters
    ----------
    text
        Raw text to tokenize.

    Returns
    -------
    list[str]
        List of tokens.
    """

    if not isinstance(text, str):
        return []
    return text.lower().split()


class BM25Retriever(BaseRetriever):
    """A BM25 retriever with efficient batch inference.

    Parameters
    ----------
    documents
        Candidate `Document`s to retrieve from.
    tokenizer
        Callable used to tokenize text into a list of string tokens. Defaults
        to a lowercase whitespace tokenizer.
    k1
        BM25 parameter controlling term frequency saturation.
    b
        BM25 parameter controlling length normalization.
    epsilon
        Small value added to IDF to avoid zero.
    show_progress
        If True, display a progress bar while ranking queries.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        tokenizer: Callable[[str], list[str]] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0.25,
        show_progress: bool = True,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(documents, preprocess=preprocess)
        base_tokenizer: Callable[[str], list[str]] = tokenizer or _default_tokenizer

        def _combined_tokenizer(text: str) -> list[str]:
            processed = self.preprocess(text)
            return base_tokenizer(processed)

        self._tokenizer: Callable[[str], list[str]] = _combined_tokenizer
        self._show_progress = show_progress

        corpus_tokens: list[list[str]] = [
            self._tokenizer(doc.page_content) for doc in documents
        ]
        self._bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b, epsilon=epsilon)

        # Cache BM25 internals for fast, vectorized scoring
        # Shapes
        self._num_docs: int = len(documents)
        self._doc_len: np.ndarray = np.asarray(self._bm25.doc_len, dtype=np.float32)
        self._avgdl: float = float(self._bm25.avgdl)
        # Scalar params
        self._k1: float = float(self._bm25.k1)
        self._b: float = float(self._bm25.b)
        # IDF per token
        self._idf: dict[str, float] = {t: float(v) for t, v in self._bm25.idf.items()}
        # Build an inverted index: token -> (doc_indices, term_frequencies)
        # This avoids scanning all documents per query
        postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        temp: dict[str, list[tuple[int, int]]] = {}
        for doc_idx, freq_map in enumerate(self._bm25.doc_freqs):
            # freq_map: dict[token, term_frequency_in_doc]
            for token, tf in freq_map.items():
                if tf <= 0:
                    continue
                if token not in temp:
                    temp[token] = []
                temp[token].append((doc_idx, int(tf)))

        for token, pairs in temp.items():
            if not pairs:
                continue
            doc_indices = np.fromiter((p[0] for p in pairs), dtype=np.int32)
            term_freqs = np.fromiter((p[1] for p in pairs), dtype=np.float32)
            postings[token] = (doc_indices, term_freqs)

        self._postings: dict[str, tuple[np.ndarray, np.ndarray]] = postings

        logger.info(
            "Initialized BM25Retriever with %d documents (k1=%.3f, b=%.3f, eps=%.3f)",
            len(documents),
            k1,
            b,
            epsilon,
        )

    def get_relevant_documents_batch(
        self, queries: list[str], k: int
    ) -> list[list[Document]]:
        """Retrieve ranked documents for a batch of queries.

        Parameters
        ----------
        queries
            The input queries as raw text.
        k
            The number of top documents to return per query.

        Returns
        -------
        list[list[Document]]
            Ranked lists of `Document` instances per query.
        """

        if not queries:
            return []

        k = int(k)
        num_docs = len(self.documents)
        k = min(max(k, 0), num_docs)
        if k == 0:
            return [[] for _ in queries]

        results: list[list[Document]] = []
        iterator: Sequence[int] = range(len(queries))
        progress_iter = tqdm(iterator, desc="Ranking", disable=not self._show_progress)
        for i in progress_iter:
            q_text = queries[i]
            q_tokens = self._tokenizer(q_text)
            if not q_tokens:
                results.append([])
                continue

            top_indices = self._topk_indices_vectorized(q_tokens, k)
            ranked_docs = [self.documents[j] for j in top_indices.tolist()]
            results.append(ranked_docs)

        return results

    @staticmethod
    def _stable_desc_indices(scores: np.ndarray) -> np.ndarray:
        """Return indices that sort scores descending with stable index tiebreak.

        Parameters
        ----------
        scores
            Array of scores of shape (D,).

        Returns
        -------
        np.ndarray
            Indices sorted by descending score; ties broken by ascending index.
        """

        num = scores.shape[0]
        indices = np.arange(num)
        # lexsort sorts by last key as primary; we want -scores primary then index
        order = np.lexsort((indices, -scores))
        return order

    # ---------- Internal: fast vectorized scoring ----------
    def _topk_indices_vectorized(self, q_tokens: list[str], k: int) -> np.ndarray:
        """Return top-k doc indices for the given query tokens using a
        vectorized accumulator over an inverted index.

        Parameters
        ----------
        q_tokens
            Tokenized query.
        k
            Number of documents to return.

        Returns
        -------
        np.ndarray
            Array of top-k document indices sorted by score desc with stable
            index tiebreak.
        """

        if k <= 0:
            return np.empty((0,), dtype=np.int32)

        # Use unique tokens to avoid double-counting
        # Keep order deterministic by using dict.fromkeys semantics
        unique_tokens = list(dict.fromkeys(q_tokens))

        scores = np.zeros(self._num_docs, dtype=np.float32)
        for token in unique_tokens:
            posting = self._postings.get(token)
            idf = self._idf.get(token)
            if posting is None or idf is None:
                continue
            doc_indices, term_freqs = posting
            if doc_indices.size == 0:
                continue
            dl = self._doc_len[doc_indices]
            denom = term_freqs + self._k1 * (1.0 - self._b + self._b * dl / self._avgdl)
            contrib = (term_freqs * (self._k1 + 1.0)) / denom
            # Scale by IDF
            scores[doc_indices] += np.float32(idf) * contrib

        # Select top-k with stable tiebreak
        if scores.size <= k:
            doc_indices_all = self._stable_desc_indices(scores)
            return doc_indices_all[:k]

        part = np.argpartition(-scores, k - 1)[:k]
        order = np.lexsort((part, -scores[part]))
        return part[order]

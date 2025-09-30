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
    ) -> None:
        super().__init__(documents)
        self._tokenizer: Callable[[str], list[str]] = tokenizer or _default_tokenizer
        self._show_progress = show_progress

        corpus_tokens: list[list[str]] = [
            self._tokenizer(doc.page_content) for doc in documents
        ]
        self._bm25 = BM25Okapi(corpus_tokens, k1=k1, b=b, epsilon=epsilon)

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

            scores: np.ndarray = self._bm25.get_scores(q_tokens)  # shape (D,)

            if scores.size <= k:
                # Full sort with stable tie-break on original index
                doc_indices = self._stable_desc_indices(scores)
                top_indices = doc_indices[:k]
            else:
                # Partial select, then stable sort the top-k block
                part = np.argpartition(-scores, k - 1)[:k]
                # Stable sort by (-score, index)
                order = np.lexsort((part, -scores[part]))
                top_indices = part[order]

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

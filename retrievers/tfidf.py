from __future__ import annotations

"""TF-IDF retriever implementation with batch querying.

Uses scikit-learn's `TfidfVectorizer` and sparse matrix operations to score
all documents against many queries at once.
"""

from typing import Any, Callable
import logging

import numpy as np
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

from langchain_core.documents import Document

from .base import BaseRetriever
from tqdm import tqdm  # type: ignore


logger = logging.getLogger(__name__)


class TFIDFRetriever(BaseRetriever):
    """A simple TF-IDF retriever with efficient batch inference.

    Parameters
    ----------
    documents
        Candidate `Document`s to retrieve from.
    tfidf_params
        Keyword arguments passed to `TfidfVectorizer`.
    normalize_scores
        If True, L2-normalize vectors enabling cosine similarity via dot
        product. `TfidfVectorizer` produces L2-normalized rows by default, so
        this can usually be left as True.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        tfidf_params: dict[str, Any] | None = None,
        normalize_scores: bool = True,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(documents, preprocess=preprocess)
        self._vectorizer = TfidfVectorizer(**(tfidf_params or {}))
        corpus = [self.preprocess(doc.page_content) for doc in documents]
        # Fit on the candidate corpus once
        self._doc_matrix: csr_matrix = self._vectorizer.fit_transform(corpus).tocsr()
        self._normalize_scores = normalize_scores

    def get_relevant_documents_batch(
        self, queries: list[str], k: int
    ) -> list[list[Document]]:
        if not queries:
            return []
        k = int(k)
        num_docs = self._doc_matrix.shape[0]
        k = min(max(k, 0), num_docs)
        if k == 0:
            return [[] for _ in queries]

        # Vectorize all queries at once using the fitted vocabulary
        preprocessed_queries = [self.preprocess(q) for q in queries]
        query_matrix: csr_matrix = self._vectorizer.transform(
            preprocessed_queries
        ).tocsr()

        # Scores are cosine similarities because rows are L2-normalized by default
        scores: csr_matrix = query_matrix @ self._doc_matrix.T  # shape (Q, D)

        # For each query row, select top-k efficiently
        results: list[list[Document]] = []
        for i in tqdm(range(scores.shape[0]), desc="Ranking"):
            row: csr_matrix = scores.getrow(i)
            if row.nnz == 0:
                results.append([])
                continue
            # Get indices and data of non-zeros
            cols = row.indices
            vals = row.data

            # If fewer than k non-zero, take all; else partial top-k selection
            if vals.size <= k:
                top_idx = np.argsort(vals)[::-1]  # descending
            else:
                # Argpartition for O(n) selection then sort the top-k
                part = np.argpartition(vals, -k)[-k:]
                top_idx = part[np.argsort(vals[part])[::-1]]

            doc_indices = cols[top_idx].tolist()
            ranked_docs = [self.documents[j] for j in doc_indices[:k]]
            results.append(ranked_docs)

        return results

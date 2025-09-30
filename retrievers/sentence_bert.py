from __future__ import annotations

"""Sentence-BERT retriever with batch querying.

This retriever uses `sentence-transformers` to embed documents and queries
and performs cosine-similarity search to return the top-k relevant documents.

The implementation follows the same interface as other retrievers in this
package and supports a preprocessing hook applied before embedding.
"""

from typing import Any, Callable
import logging

import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from langchain_core.documents import Document

from .base import BaseRetriever


logger = logging.getLogger(__name__)


class SentenceBERTRetriever(BaseRetriever):
    """A Sentence-BERT retriever with efficient batch inference.

    Parameters
    ----------
    documents
        Candidate `Document`s to retrieve from.
    model_name
        Hugging Face model name compatible with `sentence-transformers`.
        Defaults to ``"sentence-transformers/all-MiniLM-L6-v2"``.
    normalize_embeddings
        If True, L2-normalize embeddings and use dot product as cosine
        similarity. Recommended and enabled by default.
    encode_batch_size
        Batch size used for model encoding calls.
    show_progress
        If True, display progress bars while encoding.
    model_kwargs
        Additional keyword arguments passed to `SentenceTransformer` ctor.
    preprocess
        Optional text preprocessing function applied prior to embedding.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        normalize_embeddings: bool = True,
        encode_batch_size: int = 32,
        show_progress: bool = True,
        model_kwargs: dict[str, Any] | None = None,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        super().__init__(documents, preprocess=preprocess)
        self._normalize: bool = bool(normalize_embeddings)
        self._encode_batch_size: int = int(encode_batch_size)
        self._show_progress: bool = bool(show_progress)

        # Initialize model
        self._model = SentenceTransformer(model_name, **(model_kwargs or {}))

        # Prepare corpus for embedding
        corpus: list[str] = [self.preprocess(doc.page_content) for doc in documents]
        logger.info(
            "Encoding %d documents with model %s (normalize=%s)",
            len(corpus),
            model_name,
            self._normalize,
        )

        doc_embeddings = self._model.encode(
            corpus,
            batch_size=self._encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=self._show_progress,
        )

        # Ensure float32 for memory and speed
        self._doc_embeddings: np.ndarray = np.asarray(doc_embeddings, dtype=np.float32)

        if not self._normalize:
            # Precompute norms for cosine similarity when not normalized
            norms = np.linalg.norm(self._doc_embeddings, axis=1)
            # Avoid division by zero
            norms = np.where(norms == 0.0, 1.0, norms)
            self._doc_norms: np.ndarray = norms.astype(np.float32)

        logger.info(
            "Initialized SentenceBERTRetriever with %d documents (dim=%d)",
            self._doc_embeddings.shape[0],
            self._doc_embeddings.shape[1] if self._doc_embeddings.ndim == 2 else 0,
        )

    def _encode_queries(self, queries: list[str]) -> np.ndarray:
        """Encode a list of queries to an array of shape (Q, dim).

        Parameters
        ----------
        queries
            Raw query strings.

        Returns
        -------
        np.ndarray
            Encoded query matrix.
        """

        preprocessed = [self.preprocess(q) for q in queries]
        q_emb = self._model.encode(
            preprocessed,
            batch_size=self._encode_batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=self._show_progress,
        )
        q_emb_np = np.asarray(q_emb, dtype=np.float32)
        if not self._normalize:
            # Normalize on the fly for cosine similarity; keep original arrays unmodified
            q_norms = np.linalg.norm(q_emb_np, axis=1, keepdims=True)
            q_norms = np.where(q_norms == 0.0, 1.0, q_norms)
            q_emb_np = q_emb_np / q_norms
        return q_emb_np

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
        order = np.lexsort((indices, -scores))
        return order

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
        num_docs = self._doc_embeddings.shape[0]
        k = min(max(k, 0), num_docs)
        if k == 0:
            return [[] for _ in queries]

        q_emb: np.ndarray = self._encode_queries(queries)

        # Compute similarity matrix
        if self._normalize:
            # Dot product equals cosine similarity when vectors are L2-normalized
            sims: np.ndarray = np.matmul(q_emb, self._doc_embeddings.T)
        else:
            # Cosine similarity using precomputed doc norms and normalized queries
            numerator = np.matmul(q_emb, self._doc_embeddings.T)
            denom = self._doc_norms[np.newaxis, :]
            sims = numerator / denom

        results: list[list[Document]] = []
        for i in range(sims.shape[0]):
            row = sims[i]
            if row.size == 0:
                results.append([])
                continue

            if row.size <= k:
                top_indices = self._stable_desc_indices(row)[:k]
            else:
                # Partial top-k then stable order among the selected
                part = np.argpartition(-row, k - 1)[:k]
                order = np.lexsort((part, -row[part]))
                top_indices = part[order]

            ranked_docs = [self.documents[j] for j in top_indices.tolist()]
            results.append(ranked_docs)

        return results

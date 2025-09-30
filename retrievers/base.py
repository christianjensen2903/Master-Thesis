from __future__ import annotations

"""Base classes for retrieval models.

Defines a simple, typed interface for retrieval with support for efficient
batch querying. Concrete implementations should override the batch method for
performance and may rely on the default single-query wrapper.
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterable
import logging

from langchain_core.documents import Document


logger = logging.getLogger(__name__)


class BaseRetriever(ABC):
    """Abstract base class for retrievers.

    Implementations must return ranked `Document` lists for queries. The
    `get_relevant_documents_batch` method should be overridden for efficient
    batch inference. The default single-query method delegates to the batch
    implementation.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        preprocess: Callable[[str], str] | None = None,
    ) -> None:
        """Initialize the retriever with a fixed candidate set.

        Parameters
        ----------
        documents
            The candidate documents to retrieve from. The order is used as a
            stable tiebreaker.
        """
        if not documents:
            msg = "`documents` must be a non-empty list"
            logger.error(msg)
            raise ValueError(msg)
        self._documents: list[Document] = documents
        # Preprocessing hook applied to raw text prior to tokenization/vectorization
        # This enables custom stopword removal, stemming, normalization, etc.
        self._preprocess: Callable[[str], str] = preprocess or (lambda s: s)

    @property
    def documents(self) -> list[Document]:
        """Return the underlying candidate documents."""

        return self._documents

    def preprocess(self, text: str) -> str:
        """Apply the configured preprocessing to raw text.

        Parameters
        ----------
        text
            The input text to normalize.

        Returns
        -------
        str
            The preprocessed text.
        """

        return self._preprocess(text)

    @abstractmethod
    def get_relevant_documents_batch(
        self, queries: list[str], k: int
    ) -> list[list[Document]]:
        """Retrieve ranked documents for a batch of queries.

        Parameters
        ----------
        queries
            The input queries as raw text.
        k
            The number of top documents to return per query. If `k` is greater
            than the number of candidates, implementers should cap to the
            available candidates.

        Returns
        -------
        list[list[Document]]
            A list with one ranked list per query, each containing at most `k`
            `Document` instances.
        """

    def get_relevant_documents(self, query: str, k: int) -> list[Document]:
        """Retrieve ranked documents for a single query.

        This default implementation calls the batch method with a single
        element. Subclasses can override for performance if desired.
        """

        results = self.get_relevant_documents_batch([query], k=k)
        return results[0] if results else []

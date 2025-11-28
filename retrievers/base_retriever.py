from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray


class BaseRetriever(ABC):
    """Base class for all retriever implementations with iterative evaluation support."""

    @abstractmethod
    def fit(self, texts: NDArray, mask: NDArray | None = None) -> None:
        """Fit the retriever on a collection of texts."""
        pass

    @abstractmethod
    def transform(
        self, texts: NDArray, paragraph_ids: list[tuple[str, int]] | None = None
    ) -> NDArray:
        """Transform texts into their vector representations."""
        pass

    @abstractmethod
    def create_index(self, dim: int) -> None:
        """Create an empty index for iterative building."""
        pass

    @abstractmethod
    def add_to_index(self, embeddings: NDArray, indices: NDArray) -> None:
        """Add embeddings to the index with their original indices."""
        pass

    @abstractmethod
    def search_index(
        self, query_embeddings: NDArray, top_k: int
    ) -> tuple[NDArray, NDArray]:
        """
        Search the index for nearest neighbors.

        Returns:
            original_indices: Shape (n_queries, top_k) - original paragraph indices
            scores: Shape (n_queries, top_k) - similarity scores
        """
        pass

    @abstractmethod
    def reset_index(self) -> None:
        """Reset the index to empty state."""
        pass

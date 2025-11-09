from abc import ABC, abstractmethod
import numpy as np


class BaseRetriever(ABC):
    """Base class for all retriever implementations."""

    @abstractmethod
    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        """
        Fit the retriever on a collection of texts.

        Args:
            texts: Array of paragraph texts
            mask: Optional boolean mask indicating which texts to fit on (e.g., train set)
        """
        pass

    @abstractmethod
    def transform(self, texts: np.ndarray) -> np.ndarray:
        """
        Transform texts into their vector representations.

        Args:
            texts: Array of paragraph texts

        Returns:
            Matrix of shape (n_texts, n_features)
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """
        Retrieve and rank candidate paragraphs for a given query.

        Args:
            query_embedding: Embedding vector of the query
            embeddings: Full embedding matrix of all paragraphs
            candidate_indices: Indices of candidate paragraphs to rank
            top_k: If provided, only return top k results (faster)

        Returns:
            Array of candidate indices sorted by relevance (most relevant first)
        """
        pass

    def fit_transform(
        self, texts: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Fit on texts and transform them in one step.

        Args:
            texts: Array of paragraph texts
            mask: Optional boolean mask for fitting

        Returns:
            Transformed embeddings
        """
        self.fit(texts, mask)
        return self.transform(texts)

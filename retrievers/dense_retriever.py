import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from .base_retriever import BaseRetriever
import faiss  # type: ignore


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers for semantic similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        max_seq_length: int | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.model = SentenceTransformer(model_name)
        if max_seq_length is not None:
            self.model.max_seq_length = max_seq_length
        self._is_fitted = True  # Dense models don't need explicit fitting

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        # Dense retrievers use pre-trained models, no fitting needed
        pass

    def transform(self, texts: np.ndarray) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings

    def retrieve(
        self,
        query_embedding: np.ndarray,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """Create temporary index with only candidates - efficient for flat indices."""
        # Extract candidate embeddings
        candidate_embeddings = embeddings[candidate_indices]

        # Build temporary index
        temp_index = faiss.IndexFlatIP(embeddings.shape[1])
        temp_index.add(candidate_embeddings)

        # Search
        query_vec = query_embedding.reshape(1, -1)
        k = top_k if top_k is not None else len(candidate_indices)

        _, local_indices = temp_index.search(query_vec, k)

        # Map back to original indices
        return candidate_indices[local_indices[0]]

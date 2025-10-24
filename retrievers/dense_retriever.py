import numpy as np
from sentence_transformers import SentenceTransformer  # type: ignore

from .base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers for semantic similarity."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name)
        self._is_fitted = True  # Dense models don't need explicit fitting

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        # Dense retrievers use pre-trained models, no fitting needed
        pass

    def transform(self, texts: np.ndarray) -> np.ndarray:
        embeddings = self.model.encode(
            texts.tolist(),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings

    def retrieve(
        self, query_idx: int, embeddings: np.ndarray, candidate_indices: np.ndarray
    ) -> np.ndarray:
        query_vec = embeddings[query_idx]
        candidate_vecs = embeddings[candidate_indices]

        # Cosine similarity via dot product (if embeddings are normalized)
        if self.normalize_embeddings:
            similarities = candidate_vecs @ query_vec
        else:
            # Compute cosine similarity manually if not normalized
            query_norm = np.linalg.norm(query_vec)
            candidate_norms = np.linalg.norm(candidate_vecs, axis=1)
            similarities = (candidate_vecs @ query_vec) / (
                candidate_norms * query_norm + 1e-8
            )

        # Sort by similarity (high to low)
        ranked_order = np.argsort(-similarities)

        return candidate_indices[ranked_order]

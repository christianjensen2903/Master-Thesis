import numpy as np
from FlagEmbedding import BGEM3FlagModel  # type: ignore

from .base_retriever import BaseRetriever


class BGEM3Retriever(BaseRetriever):
    """BGE-M3 retriever supporting dense, sparse, and ColBERT hybrid retrieval."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        max_passage_length: int = 8192,
        weights_for_different_modes: list[float] | None = None,
    ):
        """
        Initialize BGE-M3 retriever.

        Args:
            model_name: HuggingFace model name for BGE-M3
            use_fp16: Use FP16 for faster computation
            batch_size: Batch size for encoding
            max_passage_length: Maximum sequence length (smaller = faster)
            weights_for_different_modes: Weights for [dense, sparse, colbert] scores.
                If None, uses equal weights [0.33, 0.33, 0.34]
        """
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_passage_length = max_passage_length
        self.weights = (
            weights_for_different_modes
            if weights_for_different_modes is not None
            else [0.33, 0.33, 0.34]
        )
        if len(self.weights) != 3:
            raise ValueError("weights_for_different_modes must have 3 elements")
        if abs(sum(self.weights) - 1.0) > 1e-6:
            raise ValueError("weights_for_different_modes must sum to 1.0")

        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self._is_fitted = True

        # Store all three types of embeddings
        self.dense_embeddings: np.ndarray | None = None
        self.sparse_embeddings: list | None = None
        self.colbert_embeddings: list | None = None
        self.texts: np.ndarray | None = None

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        # BGE-M3 uses pre-trained models, no fitting needed
        pass

    def transform(self, texts: np.ndarray) -> np.ndarray:
        """
        Transform texts into embeddings. Returns dense embeddings for compatibility.
        Also stores sparse and ColBERT embeddings internally for hybrid retrieval.
        """
        self.texts = texts

        # Encode with all three modes
        outputs = self.model.encode(
            texts.tolist(),
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=True,
            batch_size=self.batch_size,
            max_length=self.max_passage_length,
        )

        # Store all embeddings
        self.dense_embeddings = np.array(outputs["dense_vecs"])
        self.sparse_embeddings = outputs["lexical_weights"]
        self.colbert_embeddings = outputs["colbert_vecs"]

        # Return dense embeddings for compatibility with existing code
        return self.dense_embeddings

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """
        Retrieve candidates using hybrid scoring (dense + sparse + colbert).
        Uses pre-computed embeddings for efficiency.
        """
        if (
            self.dense_embeddings is None
            or self.sparse_embeddings is None
            or self.colbert_embeddings is None
        ):
            raise ValueError("Must call transform() first to generate embeddings")

        # Get query embeddings
        query_dense = self.dense_embeddings[query_idx]
        query_sparse = self.sparse_embeddings[query_idx]
        query_colbert = self.colbert_embeddings[query_idx]

        # Get candidate embeddings
        candidate_dense = self.dense_embeddings[candidate_indices]
        candidate_sparse = [self.sparse_embeddings[i] for i in candidate_indices]
        candidate_colbert = [self.colbert_embeddings[i] for i in candidate_indices]

        # Compute dense scores (cosine similarity via dot product, embeddings are normalized)
        dense_scores = candidate_dense @ query_dense

        # Compute sparse scores (lexical matching)
        sparse_scores = np.array(
            [
                self.model.compute_lexical_matching_score(query_sparse, cand_sparse)
                for cand_sparse in candidate_sparse
            ]
        )

        # Compute ColBERT scores
        colbert_scores = np.array(
            [
                self.model.colbert_score(query_colbert, cand_colbert)
                for cand_colbert in candidate_colbert
            ]
        )

        # Normalize scores using min-max normalization for fair combination
        # This ensures all scores are in [0, 1] range regardless of their original scale
        def min_max_norm(scores: np.ndarray) -> np.ndarray:
            min_val = np.min(scores)
            max_val = np.max(scores)
            if max_val - min_val < 1e-8:
                return (
                    np.ones_like(scores) * 0.5
                )  # All same values, return neutral score
            return (scores - min_val) / (max_val - min_val)

        dense_scores_norm = min_max_norm(dense_scores)
        sparse_scores_norm = min_max_norm(sparse_scores)
        colbert_scores_norm = min_max_norm(colbert_scores)

        # Weighted combination
        combined_scores = (
            self.weights[0] * dense_scores_norm
            + self.weights[1] * sparse_scores_norm
            + self.weights[2] * colbert_scores_norm
        )

        # Rank by score (highest first)
        if top_k is not None and top_k < len(combined_scores):
            top_k_indices = np.argpartition(-combined_scores, top_k)[:top_k]
            sorted_top_k = top_k_indices[np.argsort(-combined_scores[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-combined_scores)
            return candidate_indices[ranked_order]

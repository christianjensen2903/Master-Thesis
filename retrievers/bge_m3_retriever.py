import numpy as np
from FlagEmbedding import BGEM3FlagModel  # type: ignore

from .base_retriever import BaseRetriever
import faiss  # type: ignore


class BGEM3Retriever(BaseRetriever):

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        batch_size: int = 32,
        max_seq_length: int = 8192,
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_passage_length = max_seq_length

        self.model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        self._is_fitted = True

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        # BGE-M3 uses pre-trained models, no fitting needed
        pass

    def transform(self, texts: np.ndarray) -> np.ndarray:
        outputs = self.model.encode(
            texts.tolist(),
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
            batch_size=self.batch_size,
            max_length=self.max_passage_length,
        )
        return outputs["dense_vecs"]

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        # Extract candidate embeddings
        candidate_embeddings = embeddings[candidate_indices]

        # Build temporary index
        temp_index = faiss.IndexFlatIP(embeddings.shape[1])
        temp_index.add(candidate_embeddings)

        # Search
        query_vec = embeddings[query_idx : query_idx + 1]
        k = top_k if top_k is not None else len(candidate_indices)

        _, local_indices = temp_index.search(query_vec, k)

        # Map back to original indices
        return candidate_indices[local_indices[0]]

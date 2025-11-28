import os

# Fix OpenMP conflict on macOS (FAISS and PyTorch may use different OpenMP runtimes)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer  # type: ignore
import faiss  # type: ignore

# Set FAISS to single-threaded mode to avoid segmentation faults
faiss.omp_set_num_threads(1)

from .base_retriever import BaseRetriever


class DenseRetriever(BaseRetriever):
    """Dense retriever using sentence transformers with FAISS incremental index."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        normalize_embeddings: bool = True,
        max_seq_length: int | None = None,
        preprocessed_dir: str | None = None,
        save_embeddings_path: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar
        self.normalize_embeddings = normalize_embeddings
        self.max_seq_length = max_seq_length
        self.preprocessed_dir = preprocessed_dir
        self.save_embeddings_path = save_embeddings_path

        # Precomputed embeddings
        self.precomputed_doc_embeddings: NDArray | None = None
        self.precomputed_query_embeddings: NDArray | None = None
        self.par_id_to_idx: dict[str, int] | None = None
        self.par_metadata: list[dict] | None = None

        # FAISS index for iterative evaluation
        self._index: faiss.IndexFlatIP | None = None
        self._index_to_original: list[int] = []

        if preprocessed_dir:
            self._load_precomputed_embeddings()
        else:
            self.model = SentenceTransformer(model_name)
            if max_seq_length is not None:
                self.model.max_seq_length = max_seq_length

        self._is_fitted = True

    def _load_precomputed_embeddings(self) -> None:
        if self.preprocessed_dir is None:
            return

        preprocessed_path = Path(self.preprocessed_dir)

        doc_emb_path = preprocessed_path / "paragraph_embeddings_doc.npy"
        if not doc_emb_path.exists():
            raise FileNotFoundError(
                f"Precomputed document embeddings not found at {doc_emb_path}. "
                "Run precompute_embeddings.py first."
            )

        doc_embeddings = np.load(doc_emb_path)
        self.precomputed_doc_embeddings = doc_embeddings
        print(f"Loaded precomputed document embeddings: {doc_embeddings.shape}")

        query_emb_path = preprocessed_path / "paragraph_embeddings_query.npy"
        if query_emb_path.exists():
            query_embeddings = np.load(query_emb_path)
            self.precomputed_query_embeddings = query_embeddings
            print(f"Loaded precomputed query embeddings: {query_embeddings.shape}")

        metadata_path = preprocessed_path / "paragraph_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata: list[dict] = pickle.load(f)
            self.par_metadata = metadata
            self.par_id_to_idx = {m["id"]: i for i, m in enumerate(metadata)}
            print(f"Loaded metadata for {len(metadata)} paragraphs")

    def _get_paragraph_id(self, celex: str, number: int) -> str:
        return f"par:{celex}:{number}"

    def fit(self, texts: NDArray, mask: NDArray | None = None) -> None:
        if self.precomputed_doc_embeddings is not None:
            return

        if not hasattr(self, "model"):
            self.model = SentenceTransformer(self.model_name)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length

    def transform(
        self, texts: NDArray, paragraph_ids: list[tuple[str, int]] | None = None
    ) -> NDArray:
        if self.precomputed_doc_embeddings is not None and paragraph_ids is not None:
            if len(paragraph_ids) != len(texts):
                raise ValueError(
                    f"paragraph_ids length ({len(paragraph_ids)}) must match texts length ({len(texts)})"
                )

            if self.par_id_to_idx is None:
                raise ValueError("Metadata not loaded.")

            embedding_indices = []
            missing = []
            for celex, number in paragraph_ids:
                par_id = self._get_paragraph_id(celex, number)
                if par_id in self.par_id_to_idx:
                    embedding_indices.append(self.par_id_to_idx[par_id])
                else:
                    missing.append(par_id)

            if missing:
                print(
                    f"Warning: {len(missing)} paragraphs not found. Falling back to encoding."
                )
                return self._encode_texts(texts)

            return self.precomputed_doc_embeddings[embedding_indices]

        return self._encode_texts(texts)

    def _encode_texts(self, texts: NDArray) -> NDArray:
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(self.model_name)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
        )
        return embeddings

    def transform_queries(
        self,
        query_texts: NDArray,
        paragraph_ids: list[tuple[str, int]] | None = None,
    ) -> NDArray:
        if self.precomputed_query_embeddings is not None and paragraph_ids is not None:
            if len(paragraph_ids) != len(query_texts):
                raise ValueError(
                    f"paragraph_ids length ({len(paragraph_ids)}) must match query_texts length ({len(query_texts)})"
                )

            if self.par_id_to_idx is None:
                raise ValueError("Metadata not loaded.")

            embedding_indices = []
            missing = []
            for celex, number in paragraph_ids:
                par_id = self._get_paragraph_id(celex, number)
                if par_id in self.par_id_to_idx:
                    embedding_indices.append(self.par_id_to_idx[par_id])
                else:
                    missing.append(par_id)

            if missing:
                print(
                    f"Warning: {len(missing)} queries not found. Falling back to encoding."
                )
                return self._encode_texts(query_texts)

            return self.precomputed_query_embeddings[embedding_indices]

        return self._encode_texts(query_texts)

    # --- Iterative index methods ---

    def create_index(self, dim: int) -> None:
        self._index = faiss.IndexFlatIP(dim)
        self._index_to_original = []

    def add_to_index(self, embeddings: NDArray, indices: NDArray) -> None:
        if self._index is None:
            raise RuntimeError("Index not created. Call create_index first.")

        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        self._index.add(embeddings)
        self._index_to_original.extend(indices.tolist())

    def search_index(
        self, query_embeddings: NDArray, top_k: int
    ) -> tuple[NDArray, NDArray]:
        if self._index is None or self._index.ntotal == 0:
            n_queries = len(query_embeddings)
            return np.full((n_queries, 0), -1, dtype=np.int64), np.zeros(
                (n_queries, 0), dtype=np.float32
            )

        query_embeddings = np.ascontiguousarray(query_embeddings.astype(np.float32))
        k = min(top_k, self._index.ntotal)

        scores, faiss_indices = self._index.search(query_embeddings, k)

        # Map FAISS indices back to original indices
        original_indices = np.array(
            [
                [self._index_to_original[idx] if idx >= 0 else -1 for idx in row]
                for row in faiss_indices
            ],
            dtype=np.int64,
        )

        return original_indices, scores

    def reset_index(self) -> None:
        self._index = None
        self._index_to_original = []

    # --- Legacy methods for compatibility ---

    def load_embeddings(self, path: str | None = None) -> NDArray | None:
        load_path = path or self.save_embeddings_path
        if load_path is None or not os.path.exists(load_path):
            return None

        try:
            embeddings = np.load(load_path)
            print(f"Loaded embeddings from {load_path} (shape: {embeddings.shape})")
            return embeddings
        except Exception as e:
            print(f"Failed to load embeddings from {load_path}: {e}")
            return None

    def save_embeddings(self, embeddings: NDArray, path: str | None = None) -> None:
        save_path = path or self.save_embeddings_path
        if save_path is None:
            return

        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        np.save(save_path, embeddings)
        print(f"Saved embeddings to {save_path} (shape: {embeddings.shape})")

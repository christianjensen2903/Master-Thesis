import os
import pickle
from pathlib import Path
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

        # Precomputed embeddings (loaded if preprocessed_dir is provided)
        self.precomputed_doc_embeddings: np.ndarray | None = None
        self.precomputed_query_embeddings: np.ndarray | None = None
        self.par_id_to_idx: dict[str, int] | None = None
        self.par_metadata: list[dict] | None = None

        # Load precomputed embeddings if directory is provided
        if preprocessed_dir:
            self._load_precomputed_embeddings()
        else:
            # Only initialize model if not using precomputed embeddings
            self.model = SentenceTransformer(model_name)
            if max_seq_length is not None:
                self.model.max_seq_length = max_seq_length

        self._is_fitted = True  # Dense models don't need explicit fitting

    def _load_precomputed_embeddings(self) -> None:
        """Load precomputed embeddings from preprocessed directory."""
        if self.preprocessed_dir is None:
            return

        preprocessed_path = Path(self.preprocessed_dir)

        # Load document embeddings
        doc_emb_path = preprocessed_path / "paragraph_embeddings_doc.npy"
        if not doc_emb_path.exists():
            raise FileNotFoundError(
                f"Precomputed document embeddings not found at {doc_emb_path}. "
                "Run precompute_embeddings.py first."
            )

        self.precomputed_doc_embeddings = np.load(doc_emb_path)
        print(
            f"Loaded precomputed document embeddings: {self.precomputed_doc_embeddings.shape}"
        )

        # Load query embeddings
        query_emb_path = preprocessed_path / "paragraph_embeddings_query.npy"
        if query_emb_path.exists():
            self.precomputed_query_embeddings = np.load(query_emb_path)
            print(
                f"Loaded precomputed query embeddings: {self.precomputed_query_embeddings.shape}"
            )

        # Load metadata to create mapping
        metadata_path = preprocessed_path / "paragraph_metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                self.par_metadata = pickle.load(f)
            self.par_id_to_idx = {m["id"]: i for i, m in enumerate(self.par_metadata)}
            print(f"Loaded metadata for {len(self.par_metadata)} paragraphs")

    def _get_paragraph_id(self, celex: str, number: int) -> str:
        """Convert (celex, number) to paragraph ID format used in precomputed embeddings."""
        return f"par:{celex}:{number}"

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        # Dense retrievers use pre-trained models, no fitting needed
        # If using precomputed embeddings, nothing to fit
        if self.precomputed_doc_embeddings is not None:
            return

        # Only initialize model if not already initialized
        if not hasattr(self, "model"):
            self.model = SentenceTransformer(self.model_name)
            if self.max_seq_length is not None:
                self.model.max_seq_length = self.max_seq_length

    def transform(
        self, texts: np.ndarray, paragraph_ids: list[tuple[str, int]] | None = None
    ) -> np.ndarray:
        """
        Transform texts into embeddings.

        If precomputed embeddings are available and paragraph_ids are provided,
        uses precomputed embeddings. Otherwise, encodes texts using the model.

        Args:
            texts: Array of paragraph texts
            paragraph_ids: Optional list of (celex, number) tuples for each text.
                          Required if using precomputed embeddings.

        Returns:
            Matrix of shape (n_texts, n_features)
        """
        # Use precomputed embeddings if available and paragraph_ids provided
        if self.precomputed_doc_embeddings is not None and paragraph_ids is not None:
            if len(paragraph_ids) != len(texts):
                raise ValueError(
                    f"paragraph_ids length ({len(paragraph_ids)}) must match texts length ({len(texts)})"
                )

            if self.par_id_to_idx is None:
                raise ValueError(
                    "Metadata not loaded. Cannot map paragraph IDs to embeddings."
                )

            # Map paragraph IDs to embedding indices
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
                    f"Warning: {len(missing)} paragraphs not found in precomputed embeddings. "
                    f"Falling back to encoding. First missing: {missing[0]}"
                )
                # Fall back to encoding for missing paragraphs
                return self._encode_texts(texts)

            return self.precomputed_doc_embeddings[embedding_indices]

        # Fall back to encoding
        return self._encode_texts(texts)

    def _encode_texts(self, texts: np.ndarray) -> np.ndarray:
        """Encode texts using the sentence transformer model."""
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
        query_texts: np.ndarray,
        paragraph_ids: list[tuple[str, int]] | None = None,
    ) -> np.ndarray:
        """
        Transform query texts into embeddings.

        If precomputed query embeddings are available and paragraph_ids are provided,
        uses precomputed query embeddings. Otherwise, encodes texts using the model.

        Args:
            query_texts: Array of query texts
            paragraph_ids: Optional list of (celex, number) tuples for each query.
                          Required if using precomputed query embeddings.

        Returns:
            Matrix of shape (n_queries, n_features)
        """
        # Use precomputed query embeddings if available
        if self.precomputed_query_embeddings is not None and paragraph_ids is not None:
            if len(paragraph_ids) != len(query_texts):
                raise ValueError(
                    f"paragraph_ids length ({len(paragraph_ids)}) must match query_texts length ({len(query_texts)})"
                )

            if self.par_id_to_idx is None:
                raise ValueError(
                    "Metadata not loaded. Cannot map paragraph IDs to embeddings."
                )

            # Map paragraph IDs to embedding indices
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
                    f"Warning: {len(missing)} queries not found in precomputed embeddings. "
                    f"Falling back to encoding. First missing: {missing[0]}"
                )
                return self._encode_texts(query_texts)

            return self.precomputed_query_embeddings[embedding_indices]

        # Fall back to encoding
        return self._encode_texts(query_texts)

    def load_embeddings(self, path: str | None = None) -> np.ndarray | None:
        """Load embeddings from disk using numpy's load format."""
        load_path = path or self.save_embeddings_path
        if load_path is None:
            return None

        if not os.path.exists(load_path):
            return None

        try:
            embeddings = np.load(load_path)
            print(f"Loaded embeddings from {load_path} (shape: {embeddings.shape})")
            return embeddings
        except Exception as e:
            print(f"Failed to load embeddings from {load_path}: {e}")
            return None

    def save_embeddings(self, embeddings: np.ndarray, path: str | None = None) -> None:
        """Save embeddings to disk using numpy's save format."""
        save_path = path or self.save_embeddings_path
        if save_path is None:
            return

        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        np.save(save_path, embeddings)
        print(f"Saved embeddings to {save_path} (shape: {embeddings.shape})")

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

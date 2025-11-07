import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from .base_retriever import BaseRetriever


class GNNRetriever(BaseRetriever):
    def __init__(
        self,
        gnn_model: nn.Module,
        model_path: str | None = None,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.text_encoder_name = text_encoder_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # If loading from a checkpoint, try to read config first to determine text encoder
        if model_path is not None:
            try:
                checkpoint = torch.load(model_path, map_location="cpu")
                checkpoint_config = (
                    checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
                )
                if isinstance(checkpoint_config, dict):
                    te_name = checkpoint_config.get(
                        "text_encoder"
                    ) or checkpoint_config.get("text_encoder_name")
                    if isinstance(te_name, str):
                        self.text_encoder_name = te_name
            except Exception:
                pass

        # Initialize text encoder
        self.text_encoder = SentenceTransformer(self.text_encoder_name)
        input_dim = self.text_encoder.get_sentence_embedding_dimension()
        if input_dim is None:
            raise ValueError("Text encoder does not provide embedding dimension")

        self.gnn_model = gnn_model.to(self.device)

        # Load pretrained weights if provided
        if model_path is not None:
            self.load_model(model_path)

        self._is_fitted = False
        self.graph_data: Data | None = None
        self.text_to_node_id: dict[str, int] = {}
        self.idx_to_text: dict[int, str] = {}
        self.doc_embeddings: np.ndarray | None = None
        self.query_embeddings_cache: dict[str, np.ndarray] = {}

    def load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = (
            checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
        )
        if state_dict is None:
            raise ValueError("Checkpoint missing 'model_state_dict'")
        self.gnn_model.load_state_dict(state_dict)  # type: ignore[arg-type]
        cfg = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
        self.gnn_model.eval()

    def save_model(self, model_path: str) -> None:
        torch.save(
            {
                "model_state_dict": self.gnn_model.state_dict(),
                "config": {
                    "model_type": type(self.gnn_model).__name__,
                    "text_encoder": self.text_encoder_name,
                },
            },
            model_path,
        )

    def build_graph(
        self,
        texts: np.ndarray,
        citation_graph: dict[int, list[int]] | None = None,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> Data:
        # Use pre-computed embeddings if provided, otherwise encode texts
        if precomputed_embeddings is not None:
            print("Using pre-computed text embeddings...")
            text_embeddings = precomputed_embeddings
        else:
            print("Encoding texts with text encoder...")
            text_embeddings = self.text_encoder.encode(
                texts.tolist(),
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )

        x = torch.tensor(text_embeddings, dtype=torch.float32)  # type: ignore[call-overload]
        edge_list = []
        if citation_graph is not None:
            for src, targets in citation_graph.items():
                for tgt in targets:
                    edge_list.append([tgt, src])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()  # type: ignore[call-overload]

        return Data(x=x, edge_index=edge_index)

    def fit(
        self,
        texts: np.ndarray,
        mask: np.ndarray | None = None,
        citation_graph: dict[int, list[int]] | None = None,
        paragraph_dates: np.ndarray | None = None,
        precomputed_embeddings: np.ndarray | None = None,
    ) -> None:
        """
        Fit the retriever on a collection of texts.

        Args:
            texts: Array of paragraph texts
            mask: Optional boolean mask (unused for GNN, kept for compatibility)
            citation_graph: Citation graph (should be temporal DAG for causal inference)
            paragraph_dates: Optional dates for temporal masking during inference
            precomputed_embeddings: Optional pre-computed text embeddings (if None, will compute from texts)
        """
        # Build graph structure
        self.graph_data = self.build_graph(
            texts, citation_graph, precomputed_embeddings
        )
        self.text_to_node_id = {text: i for i, text in enumerate(texts)}
        self.paragraph_dates = paragraph_dates

        # Pre-compute document embeddings using full GNN
        self.gnn_model.eval()
        with torch.no_grad():
            x = self.graph_data.x.to(self.device)
            edge_index = self.graph_data.edge_index.to(self.device)
            doc_embeddings = self.gnn_model(x, edge_index)

            if self.normalize_embeddings:
                doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

            self.doc_embeddings = doc_embeddings.cpu().numpy()

        self._is_fitted = True

    def transform(self, texts: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.doc_embeddings is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Return GNN document embeddings for all texts in the graph
        result = np.zeros((len(texts), self.doc_embeddings.shape[1]), dtype=np.float32)

        # Update idx_to_text mapping for retrieve() lookups
        for i, text in enumerate(texts):
            self.idx_to_text[i] = text
            node_id = self.text_to_node_id.get(text)
            if node_id is not None and node_id < len(self.doc_embeddings):
                result[i] = self.doc_embeddings[node_id]

        return result

    def _get_query_embedding(self, text: str) -> np.ndarray:
        """Get query embedding for a text, using cache if available."""
        if text in self.query_embeddings_cache:
            return self.query_embeddings_cache[text]

        # Compute query embedding
        self.gnn_model.eval()
        with torch.no_grad():
            # Get text embedding
            text_embedding = self.text_encoder.encode(
                [text],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

            # Project through query encoder
            query_tensor = torch.tensor(  # type: ignore[call-overload]
                text_embedding, dtype=torch.float32, device=self.device
            )
            query_embedding = self.gnn_model.encode_query(query_tensor)  # type: ignore

            if self.normalize_embeddings:
                query_embedding = F.normalize(query_embedding, p=2, dim=1)

            query_embedding = query_embedding.cpu().numpy()[0]

        # Cache for future use
        self.query_embeddings_cache[text] = query_embedding
        return query_embedding

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        # Look up query text and get query embedding
        query_text = self.idx_to_text.get(query_idx)
        if query_text is not None:
            query_vec = self._get_query_embedding(query_text)
        else:
            # Fallback to document embedding if text not found
            query_vec = embeddings[query_idx]

        # Get candidate document embeddings
        candidate_vecs = embeddings[candidate_indices]

        # Cosine similarity (embeddings should be normalized)
        similarities = candidate_vecs @ query_vec

        # Use efficient top-k selection if requested
        if top_k is not None and top_k < len(similarities):
            top_k_indices = np.argpartition(-similarities, top_k)[:top_k]
            sorted_top_k = top_k_indices[np.argsort(-similarities[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-similarities)
            return candidate_indices[ranked_order]

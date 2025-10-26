import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data  # type: ignore
from sentence_transformers import SentenceTransformer  # type: ignore
from .base_retriever import BaseRetriever
from typing import cast


class GNNRetriever(BaseRetriever):
    def __init__(
        self,
        gnn_model: nn.Module,
        model_path: str | None = None,
        text_encoder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dim: int = 256,
        output_dim: int = 384,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        batch_size: int = 32,
        device: str | None = None,
        normalize_embeddings: bool = True,
    ) -> None:
        self.text_encoder_name = text_encoder_name
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.architecture = "external"

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # If loading from a checkpoint, try to read config first to determine text encoder/architecture
        checkpoint_config: dict[str, object] | None = None
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
                    arch = checkpoint_config.get("architecture")
                    if isinstance(arch, str):
                        self.architecture = arch
            except Exception:
                pass

        # Initialize text encoder
        self.text_encoder = SentenceTransformer(self.text_encoder_name)
        input_dim = self.text_encoder.get_sentence_embedding_dimension()
        if input_dim is None:
            raise ValueError("Text encoder does not provide embedding dimension")

        # Use provided GNN model
        self.gnn_model: BaseGNNEncoder
        self.gnn_model = cast(BaseGNNEncoder, gnn_model.to(self.device))

        # Validate input dimension compatibility when possible
        try:
            model_input_dim = cast(int, getattr(self.gnn_model, "input_dim"))
            if model_input_dim != input_dim:
                raise ValueError(
                    f"Provided GNN model input_dim={model_input_dim} does not match text encoder dim={input_dim}"
                )
        except AttributeError:
            pass

        # Load pretrained weights if provided
        if model_path is not None:
            self.load_model(model_path)

        self._is_fitted = False
        self.graph_data: Data | None = None
        self.text_to_node_id: dict[str, int] = {}

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
                    "architecture": "external",
                    "text_encoder": self.text_encoder_name,
                    "hidden_dim": self.hidden_dim,
                    "output_dim": self.output_dim,
                    "num_layers": self.num_layers,
                    "num_heads": self.num_heads,
                    "dropout": self.dropout,
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

        x = torch.tensor(text_embeddings, dtype=torch.float32)

        # Build edge index from citation graph
        if citation_graph is not None:
            edge_list = []
            for src, targets in citation_graph.items():
                for tgt in targets:
                    if src < len(texts) and tgt < len(texts):
                        edge_list.append([src, tgt])
                        # Add reverse edge for undirected graph
                        edge_list.append([tgt, src])

            if edge_list:
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            else:
                # Empty graph - no edges
                edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            # No citation graph provided - fully connected within batches
            # or use k-NN based on text similarity
            edge_index = self._build_knn_graph(text_embeddings, k=10)

        return Data(x=x, edge_index=edge_index)

    def _build_knn_graph(self, embeddings: np.ndarray, k: int = 10) -> torch.Tensor:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        # Build k-NN graph based on cosine similarity
        knn = NearestNeighbors(n_neighbors=min(k + 1, len(embeddings)), metric="cosine")
        knn.fit(embeddings)
        distances, indices = knn.kneighbors(embeddings)

        edge_list = []
        for i, neighbors in enumerate(indices):
            for j in neighbors[1:]:  # Skip self
                edge_list.append([i, j])

        if edge_list:
            return torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        else:
            return torch.empty((2, 0), dtype=torch.long)

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
        self._is_fitted = True

    def transform(self, texts: np.ndarray) -> np.ndarray:
        if not self._is_fitted or self.graph_data is None:
            raise ValueError("Model not fitted. Call fit() first.")

        self.gnn_model.eval()

        with torch.no_grad():
            # Move data to device
            x = self.graph_data.x.to(self.device)
            edge_index = self.graph_data.edge_index.to(self.device)

            # Forward pass through GNN
            embeddings = self.gnn_model(x, edge_index)

            # Normalize embeddings
            if self.normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            embeddings = embeddings.cpu().numpy()

        # Map requested texts to their embeddings
        result = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
        for i, text in enumerate(texts):
            node_id = self.text_to_node_id.get(text)
            if node_id is not None and node_id < len(embeddings):
                result[i] = embeddings[node_id]

        return result

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        query_vec = embeddings[query_idx]
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

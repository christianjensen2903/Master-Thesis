import numpy as np
import torch

from .base_retriever import BaseRetriever
from pylate import models


class ColBERTRetriever(BaseRetriever):
    """ColBERT retriever using LFM2-ColBERT-350M for late interaction retrieval."""

    def __init__(
        self,
        model_name: str = "LiquidAI/LFM2-ColBERT-350M",
        batch_size: int = 32,
        show_progress_bar: bool = True,
        device: str | None = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.show_progress_bar = show_progress_bar

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = models.ColBERT(model_name_or_path=model_name)
        if hasattr(self.model, "tokenizer") and hasattr(
            self.model.tokenizer, "pad_token"
        ):
            if self.model.tokenizer.pad_token is None:
                self.model.tokenizer.pad_token = self.model.tokenizer.eos_token

        self.model.to(self.device)
        self._is_fitted = False
        self._document_embeddings: torch.Tensor | None = None
        self._texts: np.ndarray | None = None

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        if mask is not None:
            fit_texts = texts[mask]
        else:
            fit_texts = texts

        self._texts = texts
        self._is_fitted = True

    def transform(self, texts: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Retriever must be fitted before transform")

        texts_list = texts.tolist() if isinstance(texts, np.ndarray) else list(texts)

        document_embeddings = self.model.encode(
            texts_list,
            batch_size=self.batch_size,
            is_query=False,
            show_progress_bar=self.show_progress_bar,
        )

        if isinstance(document_embeddings, torch.Tensor):
            self._document_embeddings = document_embeddings.to(self.device)
        else:
            self._document_embeddings = torch.tensor(document_embeddings).to(
                self.device
            )

        return np.arange(len(texts))

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        if self._document_embeddings is None or self._texts is None:
            raise RuntimeError("Must call transform() before retrieve()")

        query_text = self._texts[query_idx]
        query_text_list = [query_text] if isinstance(query_text, str) else query_text

        with torch.no_grad():
            query_emb = self.model.encode(
                query_text_list,
                batch_size=1,
                is_query=True,
                show_progress_bar=False,
            )

            if isinstance(query_emb, torch.Tensor):
                query_emb = query_emb.to(self.device)
            else:
                query_emb = torch.tensor(query_emb).to(self.device)

            query_emb = query_emb[0] if query_emb.dim() > 2 else query_emb

            candidate_docs = self._document_embeddings[candidate_indices]

            query_tokens, dim = query_emb.shape[0], query_emb.shape[1]
            num_candidates = candidate_docs.shape[0]
            doc_tokens = candidate_docs.shape[1]

            query_expanded = (
                query_emb.unsqueeze(0)
                .unsqueeze(2)
                .expand(num_candidates, query_tokens, doc_tokens, dim)
            )
            doc_expanded = candidate_docs.unsqueeze(1).expand(
                num_candidates, query_tokens, doc_tokens, dim
            )

            similarities_matrix = torch.sum(query_expanded * doc_expanded, dim=-1)
            maxsim_scores = torch.max(similarities_matrix, dim=-1)[0].sum(dim=-1)

            similarities = maxsim_scores.cpu().numpy()

        if top_k is not None and top_k < len(similarities):
            top_k_indices = np.argpartition(-similarities, top_k)[:top_k]
            sorted_top_k = top_k_indices[np.argsort(-similarities[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-similarities)
            return candidate_indices[ranked_order]

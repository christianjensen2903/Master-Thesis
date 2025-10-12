import pyterrier as pt  # type: ignore
import faiss  # type: ignore
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm  # type: ignore

# Ensure single-threaded execution to avoid segmentation faults
torch.set_num_threads(1)


class HuggingFaceRetriever(pt.Transformer):
    def __init__(
        self,
        documents_df: pd.DataFrame,
        model_name: str,
        use_gpu: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
    ) -> None:
        self.model_name = model_name
        self.documents_df = documents_df.reset_index(drop=True)
        self.batch_size = batch_size
        self.max_length = max_length
        self.index: faiss.Index

        # Set device
        self.device = torch.device(
            "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        )
        print(f"Using device: {self.device}")

        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._build_index()

    def _mean_pooling(
        self, model_output: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _encode_texts(self, texts: list[str], show_progress: bool = True) -> np.ndarray:
        all_embeddings = []

        iterator = range(0, len(texts), self.batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Encoding batches")

        with torch.no_grad():
            for i in iterator:
                batch_texts = texts[i : i + self.batch_size]

                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                model_output = self.model(**encoded)
                embeddings = self._mean_pooling(model_output, encoded["attention_mask"])

                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def _build_index(self) -> None:
        print(f"Encoding documents with {self.model_name}...")
        doc_texts = self.documents_df["text"].tolist()
        embeddings = self._encode_texts(doc_texts, show_progress=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        num_elements = len(embeddings)

        print(f"Building FAISS index with {num_elements} documents...")
        self.index = faiss.IndexFlatIP(dim)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        if self.device.type == "cuda" and faiss.get_num_gpus() > 0:
            print("Using GPU for FAISS index")
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(embeddings)
        print("FAISS index built successfully!")

    def transform(self, queries_df: pd.DataFrame) -> pd.DataFrame:
        results = []

        query_texts = queries_df["query"].tolist()
        qids = queries_df["qid"].tolist()

        print(f"Encoding {len(query_texts)} queries...")
        query_embeddings = self._encode_texts(query_texts, show_progress=True)
        query_embeddings = query_embeddings.astype("float32")

        # Normalize query embeddings for cosine similarity
        faiss.normalize_L2(query_embeddings)

        print("Retrieving documents...")
        scores, doc_indices = self.index.search(query_embeddings, k=1000)

        for qid, doc_idx_list, score_list in zip(qids, doc_indices, scores):
            for rank, (doc_idx, score) in enumerate(zip(doc_idx_list, score_list)):
                if doc_idx == -1:
                    continue

                docno = self.documents_df.loc[doc_idx, "docno"]

                results.append(
                    {
                        "qid": qid,
                        "docno": docno,
                        "score": float(score),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(results)

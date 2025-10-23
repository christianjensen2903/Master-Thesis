import faiss  # type: ignore
from sentence_transformers import SentenceTransformer
import pandas as pd  # type: ignore
import numpy as np
import sys
import os
import torch
from tqdm import tqdm

import pyterrier as pt  # type: ignore

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Document, load_candidate_documents

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# Fix for macOS segmentation fault with multiprocessing
torch.set_num_threads(1)


class DenseRetriever(pt.Transformer):
    def __init__(self, model_name: str, use_gpu: bool = False, k: int = 1000) -> None:
        device = "cuda" if use_gpu else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.index = None  # will become a faiss.IndexIDMap2
        self.id2docno: dict[int, str] = {}
        self.k = k

    def _maybe_to_gpu(self, index: faiss.Index) -> faiss.Index:
        if not self.use_gpu:
            return index
        # Move the CPU index to GPU(s)
        res = faiss.StandardGpuResources()
        return faiss.index_cpu_to_gpu(res, 0, index)  # single GPU; adjust if needed

    def build_index(
        self,
        documents: list[Document],
        show_progress: bool = True,
    ) -> None:

        embeddings = self.model.encode(
            [doc.text for doc in documents],
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

        # Build int64 ids and mapping to original docnos
        ids = np.arange(len(documents), dtype=np.int64)
        self.id2docno = {i: doc.docno for i, doc in enumerate(documents)}

        # Build an IP index wrapped with ID map
        d = embeddings.shape[1]
        base = faiss.IndexFlatIP(d)
        idmap = faiss.IndexIDMap2(base)
        idmap.add_with_ids(embeddings, ids)

        # Optionally move to GPU
        self.index = self._maybe_to_gpu(idmap)

    def transform(self, queries_df: pd.DataFrame) -> pd.DataFrame:
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() first.")

        query_texts = queries_df["query"].tolist()
        qids = queries_df["qid"].tolist()

        print(f"Encoding {len(query_texts)} queries...")
        qemb = self.model.encode(
            query_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        scores, ids = self.index.search(qemb, self.k)

        results = []
        for qid, id_list, score_list in zip(qids, ids, scores):
            for rank, (doc_id, score) in enumerate(zip(id_list, score_list)):
                if doc_id == -1:  # padding when k > number of indexed docs
                    continue
                results.append(
                    {
                        "qid": qid,
                        "docno": self.id2docno[int(doc_id)],
                        "score": float(score),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(results)


if __name__ == "__main__":
    # pt.init()  # optional depending on your pipeline usage
    print("Loading documents...")
    docs = load_candidate_documents("2018-01-01", use_all_paragraphs=True)
    print(f"Loaded {len(docs)} documents")

    print("Initializing dense retriever...")
    dense_retriever = DenseRetriever(
        model_name="sentence-transformers/all-MiniLM-L6-v2", use_gpu=False
    )

    print("Building index...")
    try:
        dense_retriever.build_index(docs)
        print("Index built successfully!")
    except Exception as e:
        print(f"Error building index: {e}")
        import traceback

        traceback.print_exc()

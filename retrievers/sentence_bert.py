import pyterrier as pt  # type: ignore
import faiss  # type: ignore
from sentence_transformers import SentenceTransformer
import pandas as pd  # type: ignore


class SentenceBERTRetriever(pt.Transformer):
    def __init__(
        self,
        documents_df: pd.DataFrame,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = False,
    ) -> None:
        self.model = SentenceTransformer(model_name)
        self.documents_df = documents_df.reset_index(drop=True)
        self.use_gpu = use_gpu
        self.index: faiss.Index
        self._build_index()

    def _build_index(self) -> None:
        print("Encoding documents with SentenceBERT...")
        doc_texts = self.documents_df["text"].tolist()
        embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        embeddings = embeddings.astype("float32")

        dim = embeddings.shape[1]
        num_elements = len(embeddings)

        print(f"Building FAISS index with {num_elements} documents...")
        # Use IndexFlatIP for inner product (cosine similarity with normalized vectors)
        self.index = faiss.IndexFlatIP(dim)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        if self.use_gpu and faiss.get_num_gpus() > 0:
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
        query_embeddings = self.model.encode(query_texts, show_progress_bar=True)
        query_embeddings = query_embeddings.astype("float32")

        # Normalize query embeddings for cosine similarity
        faiss.normalize_L2(query_embeddings)

        print("Retrieving documents...")
        # Search for top 1000 documents
        scores, doc_indices = self.index.search(query_embeddings, k=1000)

        for qid, doc_idx_list, score_list in zip(qids, doc_indices, scores):
            for rank, (doc_idx, score) in enumerate(zip(doc_idx_list, score_list)):
                if doc_idx == -1:  # FAISS returns -1 for padding
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

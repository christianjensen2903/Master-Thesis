import numpy as np
from numpy.typing import NDArray
import lightgbm as lgb  # type: ignore
from tqdm import tqdm

from retrievers.base_retriever import BaseRetriever
from ltr_feature_extractor import LTRFeatureExtractor


class LTRRetriever(BaseRetriever):
    """Learning-to-Rank retriever that reranks results using metadata features."""

    def __init__(
        self,
        base_retriever: BaseRetriever,
        model_path: str | None = None,
        judgments_path: str = "data/judgments_cleaned.json",
        paragraph_celex: NDArray[np.object_] | None = None,
        paragraph_number: NDArray[np.object_] | None = None,
        paragraph_dates: NDArray | None = None,
        rerank_top_k: int = 1000,
    ):
        """
        Initialize LTR retriever.

        Args:
            base_retriever: Initial retriever to get candidates
            model_path: Path to trained LightGBM model
            judgments_path: Path to judgments_cleaned.json
            paragraph_celex: Array mapping pid -> CELEX ID (set after data loading)
            paragraph_number: Array mapping pid -> paragraph number (set after data loading)
            paragraph_dates: Array mapping pid -> date (set after data loading)
            rerank_top_k: Only rerank top K results from base retriever
        """
        self.base_retriever = base_retriever
        self.model_path = model_path
        self.model: lgb.Booster | None = None
        self.rerank_top_k = rerank_top_k

        # Feature extractor
        self.feature_extractor = LTRFeatureExtractor(judgments_path)

        # Metadata arrays (set by evaluator after loading data)
        self.paragraph_celex = paragraph_celex
        self.paragraph_number = paragraph_number
        self.paragraph_dates = paragraph_dates

        # Load model if path provided
        if model_path:
            self.load_model(model_path)

    def load_model(self, path: str) -> None:
        """Load trained LightGBM model."""
        print(f"Loading LTR model from {path}...")
        self.model = lgb.Booster(model_file=path)
        print("Model loaded successfully")

    def set_metadata_arrays(
        self,
        paragraph_celex: NDArray[np.object_],
        paragraph_number: NDArray[np.object_],
        paragraph_dates: NDArray,
    ) -> None:
        """Set metadata arrays after data loading."""
        self.paragraph_celex = paragraph_celex
        self.paragraph_number = paragraph_number
        self.paragraph_dates = paragraph_dates

    def fit(self, texts: NDArray, mask: NDArray | None = None) -> None:
        """Fit base retriever and load feature extractor."""
        print("Fitting base retriever...")
        self.base_retriever.fit(texts, mask)

        print("Loading feature extractor...")
        self.feature_extractor.load()

    def transform(self, texts: NDArray) -> NDArray:
        """Transform texts using base retriever."""
        return self.base_retriever.transform(texts)

    def retrieve(
        self,
        query_idx: int,
        embeddings: NDArray,
        candidate_indices: NDArray,
        top_k: int | None = None,
    ) -> NDArray:
        """
        Retrieve and rerank candidates using LTR model.

        Process:
        1. Use base retriever to get initial ranking
        2. Take top rerank_top_k candidates
        3. Extract features for each candidate
        4. Rerank using LTR model
        5. Return reranked results
        """
        if self.model is None:
            # Fallback to base retriever if no model loaded
            return self.base_retriever.retrieve(
                query_idx, embeddings, candidate_indices, top_k
            )

        # Get initial ranking from base retriever
        # Retrieve more than needed for reranking
        initial_k = min(self.rerank_top_k, len(candidate_indices))
        initial_ranking = self.base_retriever.retrieve(
            query_idx, embeddings, candidate_indices, top_k=initial_k
        )

        if len(initial_ranking) == 0:
            return initial_ranking

        # Extract features for reranking
        features_list = []
        valid_candidates = []

        # Get query metadata
        query_celex = str(self.paragraph_celex[query_idx])
        query_par_num = int(self.paragraph_number[query_idx])
        query_date = (
            self.paragraph_dates[query_idx]
            if self.paragraph_dates is not None
            else None
        )

        # Compute dense similarities once
        query_emb = embeddings[query_idx]
        cand_embs = embeddings[initial_ranking]

        # Normalize embeddings
        query_norm = np.linalg.norm(query_emb)
        cand_norms = np.linalg.norm(cand_embs, axis=1)

        if query_norm > 0 and np.all(cand_norms > 0):
            similarities = (query_emb @ cand_embs.T) / (query_norm * cand_norms)
        else:
            similarities = np.zeros(len(initial_ranking))

        for idx, cand_pid in enumerate(initial_ranking):
            try:
                cand_celex = str(self.paragraph_celex[cand_pid])
                cand_par_num = int(self.paragraph_number[cand_pid])
                cand_date = (
                    self.paragraph_dates[cand_pid]
                    if self.paragraph_dates is not None
                    else None
                )

                # Extract features
                feature_dict = self.feature_extractor.extract_features(
                    query_celex=query_celex,
                    query_par_num=query_par_num,
                    cand_celex=cand_celex,
                    cand_par_num=cand_par_num,
                    dense_similarity=float(similarities[idx]),
                    query_date=query_date,
                    cand_date=cand_date,
                )

                # Convert to feature vector in consistent order
                feature_names = self.feature_extractor.get_feature_names()
                feature_vec = [feature_dict.get(name, 0.0) for name in feature_names]

                features_list.append(feature_vec)
                valid_candidates.append(cand_pid)
            except Exception as e:
                # Skip candidates with missing metadata
                continue

        if not features_list:
            return initial_ranking[:top_k] if top_k else initial_ranking

        # Predict scores using LTR model
        features_matrix = np.array(features_list)
        scores = self.model.predict(features_matrix)

        # Rerank by scores (descending)
        reranked_indices = np.argsort(-scores)
        reranked_candidates = np.array(valid_candidates)[reranked_indices]

        # Return top_k if specified
        if top_k:
            return reranked_candidates[:top_k]
        return reranked_candidates

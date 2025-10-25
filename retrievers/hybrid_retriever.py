import numpy as np
from collections import defaultdict

from .base_retriever import BaseRetriever


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining case-level link prediction with paragraph-level semantic similarity.

    Implements two methods:
    1. Two-stage filtering: Use case-level model to filter top-k cases,
       then apply paragraph-level retrieval within those cases.
    2. Re-ranking by weighting: Multiply paragraph similarity by case citation probability.
    """

    def __init__(
        self,
        paragraph_retriever: BaseRetriever,
        case_retriever: BaseRetriever | None = None,
        paragraph_celex: np.ndarray | None = None,
        method: str = "two_stage",
        top_k_cases: int = 100,
    ):
        """
        Args:
            paragraph_retriever: Retriever for paragraph-level semantic similarity
            case_retriever: Optional retriever for case-level link prediction
            paragraph_celex: Array mapping paragraph id to CELEX case id
            method: Either "two_stage" or "rerank"
            top_k_cases: Number of top cases to consider in two-stage method
        """
        self.paragraph_retriever = paragraph_retriever
        self.case_retriever = case_retriever
        self.paragraph_celex = paragraph_celex
        self.method = method
        self.top_k_cases = top_k_cases

        if method not in ["two_stage", "rerank"]:
            raise ValueError(f"Method must be 'two_stage' or 'rerank', got {method}")

        if case_retriever is not None and paragraph_celex is None:
            raise ValueError("paragraph_celex required when using case_retriever")

        # Build mapping from case to paragraph indices
        self.case_to_paragraphs: dict[str, list[int]] | None = (
            self._build_case_to_paragraphs(paragraph_celex)
            if paragraph_celex is not None
            else None
        )

    def _build_case_to_paragraphs(
        self, paragraph_celex: np.ndarray
    ) -> dict[str, list[int]]:
        """Build mapping from CELEX case id to list of paragraph indices."""
        case_to_pars = defaultdict(list)
        for pid, celex in enumerate(paragraph_celex):
            if celex is not None:
                case_to_pars[celex].append(pid)
        return dict(case_to_pars)

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        """
        Fit both paragraph and case retrievers.

        Args:
            texts: Array of paragraph texts
            mask: Optional boolean mask indicating training paragraphs
        """
        # Fit paragraph-level retriever
        print("Fitting paragraph-level retriever...")
        self.paragraph_retriever.fit(texts, mask)

        # Fit case-level retriever if provided
        if self.case_retriever is not None:
            print("Fitting case-level retriever...")
            self.case_retriever.fit(texts, mask)

    def transform(self, texts: np.ndarray) -> np.ndarray:
        """Transform texts using paragraph-level retriever."""
        return self.paragraph_retriever.transform(texts)

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """
        Retrieve and rank candidate paragraphs using hybrid approach.

        Args:
            query_idx: Index of the query paragraph
            embeddings: Embedding matrix
            candidate_indices: Indices of candidate paragraphs to rank
            top_k: If provided, only return top k results

        Returns:
            Array of candidate indices sorted by relevance
        """
        if self.case_retriever is None or self.paragraph_celex is None:
            # Fall back to pure paragraph-level retrieval
            return self.paragraph_retriever.retrieve(
                query_idx, embeddings, candidate_indices, top_k
            )

        if self.method == "two_stage":
            return self._retrieve_two_stage(
                query_idx, embeddings, candidate_indices, top_k
            )
        else:  # rerank
            return self._retrieve_rerank(
                query_idx, embeddings, candidate_indices, top_k
            )

    def _retrieve_two_stage(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """
        Two-stage retrieval:
        1. Use case-level model to filter top-k cases
        2. Apply paragraph-level retrieval within selected cases
        """
        assert self.case_retriever is not None
        assert self.paragraph_celex is not None
        assert self.case_to_paragraphs is not None

        # Get query case
        query_case = self.paragraph_celex[query_idx]

        # Get unique cases from candidates
        candidate_cases = list(set(self.paragraph_celex[candidate_indices]))

        # Map case names to indices (use first paragraph from each case as representative)
        case_indices_list: list[int] = []
        case_names: list[str] = []
        for case in candidate_cases:
            if case in self.case_to_paragraphs:
                # Use first paragraph from case as representative
                case_indices_list.append(self.case_to_paragraphs[case][0])
                case_names.append(case)

        case_indices_arr: np.ndarray = np.array(case_indices_list, dtype=np.int64)

        # Stage 1: Rank cases using case-level retriever
        ranked_case_indices: np.ndarray = self.case_retriever.retrieve(
            query_idx, embeddings, case_indices_arr, top_k=self.top_k_cases
        )

        # Get top-k case names
        top_cases = set(self.paragraph_celex[ranked_case_indices])

        # Stage 2: Filter candidates to only those in top cases
        filtered_list: list[int] = [
            int(cand)
            for cand in candidate_indices
            if self.paragraph_celex[cand] in top_cases
        ]

        filtered_candidates: np.ndarray
        if len(filtered_list) == 0:
            # Fallback to all candidates if filtering is too aggressive
            filtered_candidates = candidate_indices
        else:
            filtered_candidates = np.array(filtered_list, dtype=np.int64)

        # Apply paragraph-level retrieval within filtered candidates
        return self.paragraph_retriever.retrieve(
            query_idx, embeddings, filtered_candidates, top_k
        )

    def _retrieve_rerank(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """
        Re-ranking by weighting:
        Multiply paragraph similarity by case citation probability.
        """
        assert self.case_retriever is not None
        assert self.paragraph_celex is not None
        assert self.case_to_paragraphs is not None

        # Get paragraph-level scores
        para_ranked_arr = self.paragraph_retriever.retrieve(
            query_idx, embeddings, candidate_indices, top_k=None
        )

        # Create mapping from candidate to rank
        para_ranks = {int(cand): rank for rank, cand in enumerate(para_ranked_arr)}

        # Get case-level scores for each candidate
        # Group candidates by case
        case_candidates = defaultdict(list)
        for cand in candidate_indices:
            case = self.paragraph_celex[cand]
            if case is not None:
                case_candidates[case].append(cand)

        # Get representative paragraph for each case and compute case scores
        case_scores = {}
        for case, pars in case_candidates.items():
            # Use first paragraph as representative
            if case in self.case_to_paragraphs:
                rep_idx = self.case_to_paragraphs[case][0]
                # Get case score (we'll approximate with paragraph score)
                case_ranked = self.case_retriever.retrieve(
                    query_idx, embeddings, np.array([rep_idx]), top_k=None
                )
                # Use inverse rank as score
                case_scores[case] = 1.0 / (1.0 + len(case_candidates))

        # Compute hybrid scores
        hybrid_scores_list: list[float] = []
        for cand in candidate_indices:
            # Paragraph score (inverse rank)
            para_score = 1.0 / (1.0 + para_ranks.get(int(cand), len(candidate_indices)))

            # Case score
            case = self.paragraph_celex[cand]
            case_score = case_scores.get(case, 0.5)  # Default to neutral

            # Hybrid score
            hybrid_score = para_score * case_score
            hybrid_scores_list.append(hybrid_score)

        hybrid_scores_arr = np.array(hybrid_scores_list)

        # Sort by hybrid score
        if top_k is not None and top_k < len(hybrid_scores_arr):
            top_k_indices = np.argpartition(-hybrid_scores_arr, top_k)[:top_k]
            sorted_top_k = top_k_indices[np.argsort(-hybrid_scores_arr[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-hybrid_scores_arr)
            return candidate_indices[ranked_order]

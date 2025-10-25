import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from tqdm import tqdm  # type: ignore

from .base_retriever import BaseRetriever


class GraphRetriever(BaseRetriever):
    """
    Graph-based retriever using network features for link prediction.

    Implements the approach from combining case-level network analysis
    with paragraph-level semantic similarity.

    Features:
    1. Time difference
    2. TF-IDF similarity
    3. Preferential attachment
    4. Adamic-Adar
    5. Common neighbors
    6. Common referrers
    """

    def __init__(
        self,
        citation_graph: dict[int, list[int]],
        paragraph_dates: np.ndarray,
        paragraph_celex: np.ndarray,
        tfidf_embeddings: np.ndarray | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        """
        Args:
            citation_graph: Dictionary mapping paragraph id -> list of cited paragraph ids
            paragraph_dates: Array of publication dates for each paragraph
            paragraph_celex: Array of CELEX IDs for each paragraph
            tfidf_embeddings: Optional precomputed TF-IDF embeddings
            n_estimators: Number of trees in random forest
            max_depth: Maximum depth of trees
            random_state: Random seed
            n_jobs: Number of parallel jobs
        """
        self.citation_graph = citation_graph
        self.paragraph_dates = paragraph_dates
        self.paragraph_celex = paragraph_celex
        self.tfidf_embeddings = tfidf_embeddings

        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.model: RandomForestClassifier | None = None
        self._is_fitted = False

        # Build reverse citation graph (who cites me)
        self.cited_by_graph = self._build_reverse_graph()

        # Precompute in-degrees and out-degrees
        self.in_degrees = self._compute_in_degrees()
        self.out_degrees = self._compute_out_degrees()

    def _build_reverse_graph(self) -> dict[int, set[int]]:
        """Build reverse citation graph (cited -> citers)."""
        reverse_graph = defaultdict(set)
        for src, targets in self.citation_graph.items():
            for tgt in targets:
                reverse_graph[tgt].add(src)
        return dict(reverse_graph)

    def _compute_in_degrees(self) -> dict[int, int]:
        """Compute in-degree (number of times cited) for each paragraph."""
        in_degrees: dict[int, int] = defaultdict(int)
        for src, targets in self.citation_graph.items():
            for tgt in targets:
                in_degrees[tgt] += 1
        return dict(in_degrees)

    def _compute_out_degrees(self) -> dict[int, int]:
        """Compute out-degree (number of citations made) for each paragraph."""
        return {pid: len(citations) for pid, citations in self.citation_graph.items()}

    def _compute_time_difference(self, src_pid: int, tgt_pid: int) -> float:
        """
        Compute time difference in days between source and target paragraphs.
        Returns negative if target is newer (which shouldn't happen in citations).
        """
        src_date = self.paragraph_dates[src_pid]
        tgt_date = self.paragraph_dates[tgt_pid]

        # Convert numpy datetime64 to days
        diff = (src_date - tgt_date) / np.timedelta64(1, "D")
        return float(diff)

    def _compute_tfidf_similarity(self, src_pid: int, tgt_pid: int) -> float:
        """Compute TF-IDF cosine similarity between paragraphs."""
        if self.tfidf_embeddings is None:
            return 0.0

        src_vec = self.tfidf_embeddings[src_pid]
        tgt_vec = self.tfidf_embeddings[tgt_pid]

        # Handle sparse matrices
        if isinstance(src_vec, csr_matrix):
            similarity = cosine_similarity(src_vec, tgt_vec)[0, 0]
        else:
            similarity = cosine_similarity(
                src_vec.reshape(1, -1), tgt_vec.reshape(1, -1)
            )[0, 0]

        return float(similarity)

    def _compute_preferential_attachment(
        self, src_pid: int, tgt_pid: int, exclude_edge: bool = False
    ) -> float:
        """
        Preferential attachment: product of in-degree of target and out-degree of source.
        Nodes with many connections are more likely to get new connections.
        """
        src_out = self.out_degrees.get(src_pid, 0)
        tgt_in = self.in_degrees.get(tgt_pid, 0)

        # Exclude the edge src->tgt if it exists and exclude_edge is True
        if exclude_edge and tgt_pid in self.citation_graph.get(src_pid, []):
            src_out = max(0, src_out - 1)
            tgt_in = max(0, tgt_in - 1)

        return float(src_out * tgt_in)

    def _compute_adamic_adar(
        self, src_pid: int, tgt_pid: int, exclude_edge: bool = False
    ) -> float:
        """
        Adamic-Adar index: sum of 1/log(degree) over common neighbors.
        Measures how likely two nodes are to be connected based on shared neighbors.
        """
        # Get common neighbors (paragraphs both cite)
        src_citations = set(self.citation_graph.get(src_pid, []))
        tgt_citations = set(self.citation_graph.get(tgt_pid, []))

        # Exclude the edge src->tgt if needed
        if exclude_edge and tgt_pid in src_citations:
            src_citations = src_citations - {tgt_pid}

        common = src_citations.intersection(tgt_citations)

        if not common:
            return 0.0

        score = 0.0
        for neighbor in common:
            degree = self.in_degrees.get(neighbor, 0)
            # Adjust degree if we're excluding the edge and neighbor is the target
            if exclude_edge and neighbor == tgt_pid:
                degree = max(1, degree - 1)
            if degree > 1:
                score += 1.0 / np.log(degree)

        return float(score)

    def _compute_common_neighbors(
        self, src_pid: int, tgt_pid: int, exclude_edge: bool = False
    ) -> int:
        """
        Common neighbors: number of paragraphs that both source and target cite.
        """
        src_citations = set(self.citation_graph.get(src_pid, []))
        tgt_citations = set(self.citation_graph.get(tgt_pid, []))

        # Exclude the edge src->tgt if needed
        if exclude_edge and tgt_pid in src_citations:
            src_citations = src_citations - {tgt_pid}

        return len(src_citations.intersection(tgt_citations))

    def _compute_common_referrers(
        self, src_pid: int, tgt_pid: int, exclude_edge: bool = False
    ) -> int:
        """
        Common referrers: number of paragraphs that cite both source and target.
        """
        src_citers = self.cited_by_graph.get(src_pid, set())
        tgt_citers = self.cited_by_graph.get(tgt_pid, set())

        # Exclude the edge src->tgt if needed: src should not be counted as citing tgt
        if exclude_edge:
            # If src cites tgt, remove src from tgt's citers when computing common referrers
            tgt_citers = tgt_citers - {src_pid}

        return len(src_citers.intersection(tgt_citers))

    def _compute_features(
        self, src_pid: int, tgt_pid: int, exclude_edge: bool = False
    ) -> np.ndarray:
        """
        Compute all 6 features for a pair of paragraphs.

        Args:
            src_pid: Source paragraph ID
            tgt_pid: Target paragraph ID
            exclude_edge: If True, exclude the edge src->tgt from graph statistics
                         to avoid bias in feature computation
        """
        features = np.array(
            [
                self._compute_time_difference(src_pid, tgt_pid),
                self._compute_tfidf_similarity(src_pid, tgt_pid),
                self._compute_preferential_attachment(src_pid, tgt_pid, exclude_edge),
                self._compute_adamic_adar(src_pid, tgt_pid, exclude_edge),
                self._compute_common_neighbors(src_pid, tgt_pid, exclude_edge),
                self._compute_common_referrers(src_pid, tgt_pid, exclude_edge),
            ]
        )
        return features

    def fit(self, texts: np.ndarray, mask: np.ndarray | None = None) -> None:
        """
        Fit the random forest model on training data.

        Args:
            texts: Not used (features computed from graph structure)
            mask: Boolean mask indicating training paragraphs
        """
        if mask is None:
            raise ValueError("Graph retriever requires a training mask")

        print("Generating training samples from citation graph...")
        train_pids = np.where(mask)[0]

        X_train: list[np.ndarray] = []
        y_train: list[int] = []

        # Sample positive examples (actual citations)
        positive_samples: list[tuple[int, int]] = []
        for src_pid in tqdm(train_pids, desc="Collecting positive samples"):
            cited = self.citation_graph.get(src_pid, [])
            for tgt_pid in cited:
                # Only include if target is also in training set
                if mask[tgt_pid]:
                    positive_samples.append((src_pid, tgt_pid))

        print(f"Found {len(positive_samples)} positive samples")

        # Sample negative examples (non-citations)
        # Sample same number as positive examples
        print("Sampling negative examples...")
        negative_samples: list[tuple[int, int]] = []
        n_negative_needed = len(positive_samples)

        for src_pid in tqdm(train_pids, desc="Sampling negatives"):
            if len(negative_samples) >= n_negative_needed:
                break

            cited_list = self.citation_graph.get(src_pid, [])
            cited_set = set(cited_list)
            src_date = self.paragraph_dates[src_pid]

            # Sample candidates older than source
            candidates = train_pids[self.paragraph_dates[train_pids] < src_date]

            # Remove actual citations
            candidates = [c for c in candidates if c not in cited_set and c != src_pid]

            if len(candidates) > 0:
                # Sample a few negatives per source
                n_sample = min(
                    10, len(candidates), n_negative_needed - len(negative_samples)
                )
                sampled = np.random.choice(candidates, size=n_sample, replace=False)
                for tgt_pid in sampled:
                    negative_samples.append((src_pid, tgt_pid))

        print(f"Sampled {len(negative_samples)} negative samples")

        # Compute features for all samples
        print("Computing features for positive samples...")
        for src_pid, tgt_pid in tqdm(positive_samples):
            # Exclude the edge to avoid bias in feature computation
            features = self._compute_features(src_pid, tgt_pid, exclude_edge=True)
            X_train.append(features)
            y_train.append(1)

        print("Computing features for negative samples...")
        for src_pid, tgt_pid in tqdm(negative_samples):
            # No edge to exclude for negative samples
            features = self._compute_features(src_pid, tgt_pid, exclude_edge=False)
            X_train.append(features)
            y_train.append(0)

        X_train_arr = np.vstack(X_train)
        y_train_arr = np.array(y_train)

        print(f"Training Random Forest on {len(X_train_arr)} samples...")
        print(f"Feature shape: {X_train_arr.shape}")
        print(
            f"Positive samples: {np.sum(y_train_arr)}, Negative samples: {len(y_train_arr) - np.sum(y_train_arr)}"
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight="balanced",
        )

        self.model.fit(X_train_arr, y_train_arr)
        self._is_fitted = True

        print("Training complete!")

    def transform(self, texts: np.ndarray) -> np.ndarray:
        """
        Not used for graph-based retrieval (features computed on-demand).
        Returns identity embedding.
        """
        return np.arange(len(texts)).reshape(-1, 1)

    def retrieve(
        self,
        query_idx: int,
        embeddings: np.ndarray,
        candidate_indices: np.ndarray,
        top_k: int | None = None,
    ) -> np.ndarray:
        """
        Retrieve and rank candidate paragraphs using graph-based link prediction.

        Args:
            query_idx: Index of the query paragraph
            embeddings: Not used (features computed from graph)
            candidate_indices: Indices of candidate paragraphs to rank
            top_k: If provided, only return top k results

        Returns:
            Array of candidate indices sorted by link prediction score
        """
        if not self._is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before retrieval")

        # Compute features for all candidates
        features_list: list[np.ndarray] = []
        for cand_idx in candidate_indices:
            feat = self._compute_features(query_idx, cand_idx)
            features_list.append(feat)

        features = np.vstack(features_list)

        # Predict link probabilities
        scores = self.model.predict_proba(features)[:, 1]  # Probability of link

        # Sort by score (high to low)
        if top_k is not None and top_k < len(scores):
            top_k_indices = np.argpartition(-scores, top_k)[:top_k]
            sorted_top_k = top_k_indices[np.argsort(-scores[top_k_indices])]
            return candidate_indices[sorted_top_k]
        else:
            ranked_order = np.argsort(-scores)
            return candidate_indices[ranked_order]

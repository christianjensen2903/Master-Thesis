from __future__ import annotations

"""Random Forest link-prediction retriever.

Ranks candidate documents by the predicted probability that a query paragraph
would cite each candidate, using a RandomForestClassifier trained on
pre-cutoff historical citations with features:
- Time difference (days)
- TF-IDF cosine similarity
- Preferential attachment
- Adamic-Adar
- Common neighbors
- Common referrers (directed)

Artifacts expected in an `artifacts_dir` (created by the training script):
- vectorizer.joblib    : fitted `TfidfVectorizer`
- model.joblib         : trained `RandomForestClassifier`
- g_undirected.gpickle : pre-cutoff undirected graph
- g_directed.gpickle   : pre-cutoff directed graph
- config.json          : includes at least the `cutoff_date`
"""

from typing import Callable, Iterable, Tuple
import json
from pathlib import Path
import pickle

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from joblib import load  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from tqdm import tqdm  # type: ignore

from langchain_core.documents import Document

from .base import BaseRetriever
from features.link_features import (
    build_pair_features,
    features_to_array,
)


QueryInfoProvider = Callable[[str], tuple[str | None, pd.Timestamp | None]]


class RandomForestLinkRetriever(BaseRetriever):
    """Retriever that uses a trained RandomForest link model for ranking.

    Parameters
    ----------
    documents
        Candidate `Document`s to retrieve from. Each document's metadata must
        include keys `id` (str) and `date` (datetime-like).
    artifacts_dir
        Directory containing the serialized vectorizer/model/graphs/config.
    preprocess
        Optional text preprocessing callable applied prior to vectorization.
    query_info_provider
        Optional callable that maps a query text to a tuple `(from_id, from_date)`.
        If omitted or if it returns `(None, None)`, graph and time features will
        degrade to zeros for that query.
    """

    def __init__(
        self,
        documents: list[Document],
        *,
        artifacts_dir: str | Path,
        preprocess: Callable[[str], str] | None = None,
        query_info_provider: QueryInfoProvider | None = None,
    ) -> None:
        super().__init__(documents, preprocess=preprocess)

        self._artifacts_dir: Path = Path(artifacts_dir)
        self._query_info_provider: QueryInfoProvider | None = query_info_provider

        # Load artifacts
        self._vectorizer: TfidfVectorizer = load(
            self._artifacts_dir / "vectorizer.joblib"
        )
        self._model: RandomForestClassifier = load(self._artifacts_dir / "model.joblib")
        # Ensure inference is quiet (avoid joblib Parallel verbose spam)
        self._model.set_params(verbose=0)
        with open(self._artifacts_dir / "g_undirected.gpickle", "rb") as f:
            self._g_undirected: nx.Graph = pickle.load(f)
        with open(self._artifacts_dir / "g_directed.gpickle", "rb") as f:
            self._g_directed: nx.DiGraph = pickle.load(f)

        config_path = self._artifacts_dir / "config.json"
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as f:
                self._config = json.load(f)
        else:
            self._config = {"cutoff_date": None}

        # Precompute candidate representations
        self._cand_ids: list[str] = [str(doc.metadata["id"]) for doc in self.documents]
        self._cand_dates: list[pd.Timestamp | None] = [
            (
                pd.to_datetime(doc.metadata.get("date"))
                if doc.metadata.get("date") is not None
                else None
            )
            for doc in self.documents
        ]
        cand_texts = [self.preprocess(doc.page_content) for doc in self.documents]
        self._cand_matrix: csr_matrix = self._vectorizer.transform(cand_texts).tocsr()

    def _resolve_query_info(
        self, query_text: str
    ) -> tuple[str | None, pd.Timestamp | None]:
        """Resolve `(from_id, from_date)` for a query text using the provider if set."""

        if self._query_info_provider is None:
            return None, None
        try:
            return self._query_info_provider(query_text)
        except Exception:
            return None, None

    def get_relevant_documents_batch(
        self, queries: list[str], k: int
    ) -> list[list[Document]]:
        if not queries:
            return []
        k = int(k)
        num_docs = len(self.documents)
        k = min(max(k, 0), num_docs)
        if k == 0:
            return [[] for _ in queries]

        # Transform all queries
        preprocessed = [self.preprocess(q) for q in queries]
        q_matrix: csr_matrix = self._vectorizer.transform(preprocessed).tocsr()

        results: list[list[Document]] = []
        for i in tqdm(range(len(queries)), desc="Ranking (RF)", leave=False):
            q_text = queries[i]
            q_vec = q_matrix.getrow(i)
            from_id, from_date = self._resolve_query_info(q_text)

            # Build features against all candidates
            pair_feats = []
            for cand_idx, to_id in enumerate(self._cand_ids):
                to_date = self._cand_dates[cand_idx]
                to_vec = self._cand_matrix.getrow(cand_idx)
                feats = build_pair_features(
                    from_id=str(from_id) if from_id is not None else "",
                    to_id=str(to_id),
                    from_date=from_date,
                    to_date=to_date,
                    from_vec=q_vec,
                    to_vec=to_vec,
                    g_undirected=self._g_undirected,
                    g_directed=self._g_directed,
                )
                pair_feats.append(feats)

            X = features_to_array(pair_feats)
            # Predict probabilities of class 1 (edge exists)
            if hasattr(self._model, "predict_proba"):
                probs = self._model.predict_proba(X)[:, 1]
            else:
                # Fallback: use decision_function if available
                if hasattr(self._model, "decision_function"):
                    scores = self._model.decision_function(X)
                    # Min-max normalize to [0,1]
                    mn, mx = float(np.min(scores)), float(np.max(scores))
                    probs = (scores - mn) / (mx - mn + 1e-12)
                else:
                    probs = np.zeros(X.shape[0], dtype=np.float32)

            # Rank candidates by probability
            order = np.argsort(probs)[::-1][:k]
            ranked_docs = [self.documents[j] for j in order]
            results.append(ranked_docs)

        return results

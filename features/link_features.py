from __future__ import annotations

"""Utilities for computing link prediction features.

This module centralizes feature engineering shared by training and inference:
- Temporal features (e.g., time difference)
- Textual similarity features (TF-IDF cosine similarity)
- Graph-based features (preferential attachment, Adamic-Adar, common neighbors,
  and common referrers for directed graphs)

Graphs are expected to be constructed from the historical (pre-cutoff) citation
edges to avoid temporal leakage.
"""

from typing import Iterable, Tuple
from dataclasses import dataclass
import math

import networkx as nx  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from tqdm.auto import tqdm  # type: ignore


@dataclass(frozen=True)
class PairFeatures:
    """Feature vector for a single (source, target) pair.

    Attributes
    ----------
    time_diff_days
        Number of days between the source publication and target publication
        (non-negative). If unknown, set to 0.
    tfidf_cosine
        Cosine similarity between source and target TF-IDF vectors in [0, 1].
    pref_attachment
        Preferential attachment score deg(u) * deg(v) on the undirected graph.
    adamic_adar
        Adamic–Adar score on the undirected graph.
    common_neighbors
        The number of common neighbors on the undirected graph.
    common_referrers
        The number of common predecessors on the directed graph.
    """

    time_diff_days: float
    tfidf_cosine: float
    pref_attachment: float
    adamic_adar: float
    common_neighbors: int
    common_referrers: int


def build_temporally_valid_edge_mask(df: pd.DataFrame) -> pd.Series:
    """Return a boolean mask for edges where target date is strictly earlier than source.

    Parameters
    ----------
    df
        DataFrame with columns `DATE_FROM` and `DATE_TO` as datetime-like.

    Returns
    -------
    pd.Series
        Boolean mask of the same length as `df`.
    """

    return pd.to_datetime(df["DATE_TO"]) < pd.to_datetime(df["DATE_FROM"])


def build_pre_cutoff_graphs(
    df: pd.DataFrame, cutoff: pd.Timestamp
) -> Tuple[nx.Graph, nx.DiGraph]:
    """Construct undirected and directed graphs using edges strictly before the cutoff.

    The graphs are constructed over paragraph identifiers `FROM_ID` and `TO_ID`.

    Parameters
    ----------
    df
        DataFrame containing citation pairs with columns `FROM_ID`, `TO_ID`,
        `DATE_FROM`, `DATE_TO`.
    cutoff
        Cutoff date; only edges with `DATE_FROM` strictly earlier than `cutoff`
        and `DATE_TO` strictly earlier than their corresponding `DATE_FROM` are
        included.

    Returns
    -------
    (nx.Graph, nx.DiGraph)
        The undirected and directed graphs, respectively.
    """

    df = df.copy()
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])  # type: ignore[assignment]
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])  # type: ignore[assignment]
    mask_time = build_temporally_valid_edge_mask(df)
    mask_cut = df["DATE_FROM"] < pd.to_datetime(cutoff)
    sub = df.loc[mask_time & mask_cut, ["FROM_ID", "TO_ID"]].dropna()

    g_undirected = nx.Graph()
    g_directed = nx.DiGraph()
    for _, row in tqdm(
        sub.iterrows(), total=len(sub), desc="Building pre-cutoff graphs"
    ):
        u = str(row["FROM_ID"])  # source paragraph id
        v = str(row["TO_ID"])  # target paragraph id
        g_undirected.add_edge(u, v)
        g_directed.add_edge(u, v)

    return g_undirected, g_directed


def fit_tfidf(
    texts: list[str], *, tfidf_params: dict[str, object] | None = None
) -> Tuple[TfidfVectorizer, csr_matrix]:
    """Fit a TF-IDF vectorizer and return the document-term matrix.

    Parameters
    ----------
    texts
        Corpus to fit the vectorizer on.
    tfidf_params
        Optional keyword parameters to initialize `TfidfVectorizer`.

    Returns
    -------
    (TfidfVectorizer, csr_matrix)
        The fitted vectorizer and the sparse matrix of shape (N, V).
    """

    vectorizer = TfidfVectorizer(**(tfidf_params or {}))
    matrix = vectorizer.fit_transform(texts).tocsr()
    return vectorizer, matrix


def cosine_similarity_sparse(a: csr_matrix, b: csr_matrix) -> np.ndarray:
    """Compute cosine similarity between rows of `a` and rows of `b`.

    Assumes rows are L2-normalized, as is the default for `TfidfVectorizer`.

    Parameters
    ----------
    a
        Matrix of shape (N, V)
    b
        Matrix of shape (M, V)

    Returns
    -------
    np.ndarray
        Dense similarity matrix of shape (N, M) with values in [0, 1].
    """

    return (a @ b.T).toarray()


def compute_graph_features(
    g_undirected: nx.Graph,
    g_directed: nx.DiGraph,
    u: str,
    v: str,
) -> Tuple[float, float, int, int]:
    """Compute graph-based features for (u, v).

    Parameters
    ----------
    g_undirected
        Undirected graph built from pre-cutoff edges.
    g_directed
        Directed graph built from pre-cutoff edges.
    u
        Source node identifier (FROM_ID).
    v
        Target node identifier (TO_ID).

    Returns
    -------
    (pref_attachment, adamic_adar, common_neighbors, common_referrers)
        Tuple of graph feature values.
    """

    # Preferential attachment
    deg_u = g_undirected.degree(u) if g_undirected.has_node(u) else 0
    deg_v = g_undirected.degree(v) if g_undirected.has_node(v) else 0
    pref_attachment = float(deg_u * deg_v)

    # Adamic-Adar (sum over common neighbors of 1/log(deg))
    adamic_adar = 0.0
    if g_undirected.has_node(u) and g_undirected.has_node(v):
        try:
            for _, _, score in nx.adamic_adar_index(g_undirected, [(u, v)]):
                adamic_adar = float(score)
        except ZeroDivisionError:
            adamic_adar = 0.0

    # Common neighbors
    common_neighbors = 0
    if g_undirected.has_node(u) and g_undirected.has_node(v):
        common_neighbors = len(list(nx.common_neighbors(g_undirected, u, v)))

    # Common referrers = common predecessors in directed pre-cutoff graph
    common_referrers = 0
    if g_directed.has_node(u) and g_directed.has_node(v):
        preds_u = set(g_directed.predecessors(u))
        preds_v = set(g_directed.predecessors(v))
        common_referrers = len(preds_u.intersection(preds_v))

    return pref_attachment, adamic_adar, int(common_neighbors), int(common_referrers)


def build_pair_features(
    *,
    from_id: str,
    to_id: str,
    from_date: pd.Timestamp | None,
    to_date: pd.Timestamp | None,
    from_vec: csr_matrix | None,
    to_vec: csr_matrix | None,
    g_undirected: nx.Graph,
    g_directed: nx.DiGraph,
) -> PairFeatures:
    """Assemble the full feature vector for a (FROM_ID, TO_ID) pair.

    Parameters
    ----------
    from_id
        Paragraph identifier of the source.
    to_id
        Paragraph identifier of the target.
    from_date
        Publication date of the source paragraph, or None if unknown.
    to_date
        Publication date of the target paragraph, or None if unknown.
    from_vec
        TF-IDF row-vector for the source text.
    to_vec
        TF-IDF row-vector for the target text.
    g_undirected
        Pre-cutoff undirected graph.
    g_directed
        Pre-cutoff directed graph.

    Returns
    -------
    PairFeatures
        The computed feature values.
    """

    # Time difference in days (non-negative)
    time_diff_days = 0.0
    if from_date is not None and to_date is not None:
        delta = (pd.to_datetime(from_date) - pd.to_datetime(to_date)).days
        time_diff_days = float(max(delta, 0))

    # TF-IDF cosine similarity
    tfidf_cosine = 0.0
    if from_vec is not None and to_vec is not None:
        num = float(from_vec.multiply(to_vec).sum())
        # Rows are L2-normalized by default when using TfidfVectorizer
        tfidf_cosine = num

    # Graph features
    pref_attach, aa, cn, cref = compute_graph_features(
        g_undirected=g_undirected, g_directed=g_directed, u=from_id, v=to_id
    )

    return PairFeatures(
        time_diff_days=time_diff_days,
        tfidf_cosine=tfidf_cosine,
        pref_attachment=pref_attach,
        adamic_adar=aa,
        common_neighbors=cn,
        common_referrers=cref,
    )


def features_to_array(features: Iterable[PairFeatures]) -> np.ndarray:
    """Convert an iterable of `PairFeatures` into a dense array.

    The order of columns is:
    [time_diff_days, tfidf_cosine, pref_attachment, adamic_adar, common_neighbors, common_referrers]

    Parameters
    ----------
    features
        Iterable of feature objects.

    Returns
    -------
    np.ndarray
        Array of shape (N, 6).
    """

    rows: list[list[float]] = []
    for f in features:
        rows.append(
            [
                float(f.time_diff_days),
                float(f.tfidf_cosine),
                float(f.pref_attachment),
                float(f.adamic_adar),
                float(f.common_neighbors),
                float(f.common_referrers),
            ]
        )
    return np.asarray(rows, dtype=np.float32)

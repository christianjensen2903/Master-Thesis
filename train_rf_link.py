from __future__ import annotations

"""Train a Random Forest link prediction model on pre-2018 citations.

The script:
- Loads `data/clean_data.csv` containing paragraph-to-paragraph citations
- Filters candidates to `DATE_TO` < 2018-01-01 and sources with `DATE_FROM` < 2018-01-01
- Builds TF-IDF over unique paragraph texts (both FROM and TO)
- Constructs pre-cutoff graphs from historical edges
- Samples positive and negative pairs and computes features
- Trains a `RandomForestClassifier`
- Saves artifacts (vectorizer, model, graphs, config)
"""

from typing import Tuple
from pathlib import Path
import argparse
import json
import pickle
import numpy as np
import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.model_selection import train_test_split  # type: ignore
from sklearn.metrics import roc_auc_score  # type: ignore
from joblib import dump  # type: ignore
import networkx as nx  # type: ignore
from scipy.sparse import csr_matrix  # type: ignore
from tqdm.auto import tqdm  # type: ignore

from features.link_features import (
    build_pre_cutoff_graphs,
    build_pair_features,
    features_to_array,
    fit_tfidf,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RF link prediction model")
    p.add_argument("--csv", type=Path, default=Path("data/clean_data.csv"))
    p.add_argument("--artifacts-dir", type=Path, default=Path("artifacts/rf_link"))
    p.add_argument("--cutoff", type=str, default="2018-01-01")
    p.add_argument(
        "--neg-pos-ratio", type=float, default=1.0, help="Negatives per positive"
    )
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--max-neg-per-source", type=int, default=100)
    return p.parse_args()


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["DATE_FROM"] = pd.to_datetime(df["DATE_FROM"])  # type: ignore[assignment]
    df["DATE_TO"] = pd.to_datetime(df["DATE_TO"])  # type: ignore[assignment]
    df["FROM_ID"] = df["CELEX_FROM"].astype(str) + "::" + df["NUMBER_FROM"].astype(str)
    df["TO_ID"] = df["CELEX_TO"].astype(str) + "::" + df["NUMBER_TO"].astype(str)
    return df


def build_text_index(df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str]]:
    """Return FROM_ID->TEXT_FROM and TO_ID->TEXT_TO dictionaries."""
    from_map = (
        df[["FROM_ID", "TEXT_FROM"]]
        .drop_duplicates("FROM_ID")
        .set_index("FROM_ID")["TEXT_FROM"]
        .to_dict()
    )
    to_map = (
        df[["TO_ID", "TEXT_TO"]]
        .drop_duplicates("TO_ID")
        .set_index("TO_ID")["TEXT_TO"]
        .to_dict()
    )
    return from_map, to_map


def main() -> None:
    args = parse_args()
    cutoff = pd.Timestamp(args.cutoff)
    out_dir: Path = args.artifacts_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = prepare_dataframe(df)

    # Keep only temporally valid edges and those strictly before cutoff for training
    temporal_mask = pd.to_datetime(df["DATE_TO"]) < pd.to_datetime(
        df["DATE_FROM"]
    )  # target older than source
    train_edge_mask = temporal_mask & (df["DATE_FROM"] < cutoff)
    train_edges = df.loc[train_edge_mask].copy()

    # Build graphs from pre-cutoff edges
    g_undirected, g_directed = build_pre_cutoff_graphs(df, cutoff=cutoff)

    # Text corpora for TF-IDF (unique texts)
    from_map, to_map = build_text_index(df)
    all_ids = list(set(list(from_map.keys()) + list(to_map.keys())))
    id_to_text: dict[str, str] = {}
    for pid in all_ids:
        if pid in to_map:
            id_to_text[pid] = str(to_map[pid])
        elif pid in from_map:
            id_to_text[pid] = str(from_map[pid])
        else:
            id_to_text[pid] = ""

    id_list = list(id_to_text.keys())
    texts = [id_to_text[i] for i in id_list]
    vectorizer, matrix = fit_tfidf(texts)
    pid_to_row = {pid: idx for idx, pid in enumerate(id_list)}

    # Date maps
    from_date_map = (
        df[["FROM_ID", "DATE_FROM"]]
        .drop_duplicates("FROM_ID")
        .set_index("FROM_ID")["DATE_FROM"]
        .to_dict()
    )
    to_date_map = (
        df[["TO_ID", "DATE_TO"]]
        .drop_duplicates("TO_ID")
        .set_index("TO_ID")["DATE_TO"]
        .to_dict()
    )

    # Positive samples
    pos_pairs = list(
        zip(train_edges["FROM_ID"].astype(str), train_edges["TO_ID"].astype(str))
    )

    # Negative sampling: for each source, sample targets that existed before the source date but are not linked
    rng = np.random.default_rng(args.random_state)
    neg_pairs: list[tuple[str, str]] = []
    grouped = train_edges.groupby("FROM_ID")
    existing_edges = set(pos_pairs)

    # Precompute candidate TO_IDs older than each source's date
    to_ids_by_date = df[["TO_ID", "DATE_TO"]].drop_duplicates("TO_ID")
    to_ids_by_date["DATE_TO"] = pd.to_datetime(to_ids_by_date["DATE_TO"])  # type: ignore
    to_ids_by_date = to_ids_by_date.set_index("TO_ID")["DATE_TO"].to_dict()

    for from_id, group in tqdm(grouped, total=len(grouped), desc="Sampling negatives"):
        from_date = from_date_map.get(from_id)
        if from_date is None:
            continue
        # valid negatives: TO older than FROM and not already linked
        valid_targets = [
            tid
            for tid, tdate in to_ids_by_date.items()
            if pd.to_datetime(tdate) < pd.to_datetime(from_date)
        ]
        # exclude positives
        linked_targets = set(group["TO_ID"].astype(str).tolist())
        pool = [
            t
            for t in valid_targets
            if (from_id, t) not in existing_edges and t not in linked_targets
        ]
        if not pool:
            continue
        num_pos = len(linked_targets)
        num_neg = min(
            int(args.neg_pos_ratio * num_pos), len(pool), args.max_neg_per_source
        )
        if num_neg <= 0:
            continue
        sampled = rng.choice(pool, size=num_neg, replace=False)
        for t in sampled:
            neg_pairs.append((from_id, str(t)))

    # Build feature matrices
    def vec_or_none(pid: str) -> Tuple[int | None, csr_matrix | None]:
        idx = pid_to_row.get(pid)
        if idx is None:
            return None, None
        return idx, matrix.getrow(idx)

    feats_pos = []
    for u, v in tqdm(pos_pairs, desc="Building positive pair features"):
        _, u_vec = vec_or_none(u)
        _, v_vec = vec_or_none(v)
        feats_pos.append(
            build_pair_features(
                from_id=u,
                to_id=v,
                from_date=from_date_map.get(u),
                to_date=to_date_map.get(v),
                from_vec=u_vec,
                to_vec=v_vec,
                g_undirected=g_undirected,
                g_directed=g_directed,
            )
        )

    feats_neg = []
    for u, v in tqdm(neg_pairs, desc="Building negative pair features"):
        _, u_vec = vec_or_none(u)
        _, v_vec = vec_or_none(v)
        feats_neg.append(
            build_pair_features(
                from_id=u,
                to_id=v,
                from_date=from_date_map.get(u),
                to_date=to_date_map.get(v),
                from_vec=u_vec,
                to_vec=v_vec,
                g_undirected=g_undirected,
                g_directed=g_directed,
            )
        )

    X_pos = features_to_array(feats_pos)
    X_neg = features_to_array(feats_neg)
    y_pos = np.ones(X_pos.shape[0], dtype=np.int64)
    y_neg = np.zeros(X_neg.shape[0], dtype=np.int64)

    X = np.vstack([X_pos, X_neg]) if X_neg.size else X_pos
    y = np.concatenate([y_pos, y_neg]) if X_neg.size else y_pos

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=args.random_state,
        class_weight="balanced_subsample",
        verbose=1,
    )
    model.fit(X_train, y_train)

    # Simple validation metric
    if hasattr(model, "predict_proba"):
        val_probs = model.predict_proba(X_val)[:, 1]
    else:
        val_probs = model.decision_function(X_val)  # type: ignore[arg-type]
    auc = roc_auc_score(y_val, val_probs)
    print(f"Validation ROC-AUC: {auc:.4f}")

    # Save artifacts
    dump(vectorizer, out_dir / "vectorizer.joblib")
    dump(model, out_dir / "model.joblib")
    with (out_dir / "g_undirected.gpickle").open("wb") as f:
        pickle.dump(g_undirected, f, pickle.HIGHEST_PROTOCOL)
    with (out_dir / "g_directed.gpickle").open("wb") as f:
        pickle.dump(g_directed, f, pickle.HIGHEST_PROTOCOL)
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump({"cutoff_date": args.cutoff}, f, ensure_ascii=False, indent=2)

    print(f"Saved artifacts to {out_dir}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import (  # type: ignore
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split  # type: ignore
import joblib  # type: ignore


RANDOM_SEED: int = 42


FEATURE_COLUMNS: list[str] = [
    "days_diff",
    "adamic_adar",
    "pref_attach",
    "common_neigh",
    "common_in_neigh",
    "common_out_neigh",
    "cos_sim",
]


@dataclass
class TrainConfig:
    data_path: str = "data/case_to_case_training2.csv"
    artifacts_dir: str = "artifacts"
    model_name: str = "random_forest_case_to_case.joblib"
    metrics_name: str = "random_forest_metrics.json"
    test_size: float = 0.2
    n_estimators: int = 400
    max_depth: int | None = None
    n_jobs: int = -1


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def prepare_features_labels(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    for col in FEATURE_COLUMNS + ["label"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    X = df[FEATURE_COLUMNS].astype(float).to_numpy()
    y = df["label"].astype(int).to_numpy()
    return X, y, FEATURE_COLUMNS


def train_random_forest(X: np.ndarray, y: np.ndarray, config: TrainConfig) -> dict:
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    clf = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        n_jobs=config.n_jobs,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        verbose=1,
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": float(roc_auc_score(y_val, y_prob)),
        "average_precision": float(average_precision_score(y_val, y_prob)),
        "report": classification_report(y_val, y_pred, output_dict=True),
    }

    return {"model": clf, "metrics": metrics}


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_artifacts(
    model: RandomForestClassifier,
    metrics: dict,
    feature_names: Iterable[str],
    config: TrainConfig,
) -> None:
    ensure_dir(config.artifacts_dir)
    model_path = os.path.join(config.artifacts_dir, config.model_name)
    metrics_path = os.path.join(config.artifacts_dir, config.metrics_name)
    features_path = os.path.join(config.artifacts_dir, "random_forest_features.json")

    joblib.dump(model, model_path)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump(list(feature_names), f, indent=2)


def main() -> None:
    config = TrainConfig()
    df = load_dataset(config.data_path)
    X, y, feature_names = prepare_features_labels(df)
    result = train_random_forest(X, y, config)

    # Print results instead of saving
    print("Random Forest Training Results:")
    print("=" * 50)
    print(f"ROC AUC Score: {result['metrics']['roc_auc']:.4f}")
    print(f"Average Precision: {result['metrics']['average_precision']:.4f}")
    print("\nClassification Report:")
    print("-" * 30)
    report = result["metrics"]["report"]
    for class_label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"Class {class_label}:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")

    print(f"\nFeature Names: {feature_names}")
    print(f"Model trained with {result['model'].n_estimators} estimators")


if __name__ == "__main__":
    main()

"""
src/train.py

Trains a retention-risk model and saves artifacts for downstream scoring + dashboarding.

What changed vs your original version
-------------------------------------
1) Uses your realistic synthetic dataset by default:
      data/student_retention_synthetic.csv
2) Saves more portfolio-grade artifacts:
      artifacts/model.pkl
      artifacts/features.json
      artifacts/metrics.json        (includes confusion matrix + precision/recall/f1)
      artifacts/config.json         (training run settings)
      artifacts/data_schema.json    (basic info about columns)
      artifacts/sample.csv          (for demo scoring in Streamlit)

Run (default)
-------------
python -m src.train

Run (custom)
------------
python -m src.train --data data/student_retention_synthetic.csv --label-col label --id-col student_id
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


ARTIFACT_DIR = Path("artifacts")
DEFAULT_DATA_PATH = Path("data/student_retention_synthetic.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train retention-risk model.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA_PATH),
        help="Path to CSV training dataset.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="label",
        help="Name of binary target column (1 = at risk / not retained).",
    )
    parser.add_argument(
        "--id-col",
        type=str,
        default="student_id",
        help="Optional ID column to exclude from training features.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Fraction of data used for test set.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Default threshold for classification metrics.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=400,
        help="Number of trees in RandomForest.",
    )
    return parser.parse_args()


def infer_feature_columns(df: pd.DataFrame, label_col: str, id_col: Optional[str]) -> List[str]:
    """
    Determine which columns will be used as model features.
    We exclude label_col and (optionally) id_col.
    """
    excluded = {label_col}
    if id_col and id_col in df.columns:
        excluded.add(id_col)

    features = [c for c in df.columns if c not in excluded]
    return features


def build_data_schema(df: pd.DataFrame, label_col: str, id_col: Optional[str]) -> Dict:
    """
    Create a simple schema artifact: column names, dtypes, and basic row counts.
    """
    schema = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "label_col": label_col,
        "id_col": id_col if (id_col and id_col in df.columns) else None,
        "columns": [
            {
                "name": c,
                "dtype": str(df[c].dtype),
                "missing_rate": float(df[c].isna().mean()),
            }
            for c in df.columns
        ],
    }
    return schema


def compute_threshold_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict:
    """
    Compute confusion-matrix-based metrics at a given probability threshold.
    """
    y_pred = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # precision_recall_fscore_support returns per-class; average='binary' uses positive class=1
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )

    return {
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        },
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    args = parse_args()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {data_path}. "
            f"Generate it first: python -m src.make_synthetic_data"
        )

    # -----------------------------
    # 1) Load data
    # -----------------------------
    df = pd.read_csv(data_path)

    if args.label_col not in df.columns:
        raise ValueError(f"label-col '{args.label_col}' not found in CSV columns.")

    # Basic target sanity: must be 0/1
    y = df[args.label_col].astype(int).to_numpy()
    unique = set(np.unique(y).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"Label column must be binary 0/1. Found values: {sorted(unique)}")

    # -----------------------------
    # 2) Define features
    # -----------------------------
    features = infer_feature_columns(df, args.label_col, args.id_col)

    # Ensure features exist and are numeric/coercible
    X = df[features].copy()
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # Impute missing values simply (median) — good baseline for tabular data
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    # -----------------------------
    # 3) Train/test split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    # -----------------------------
    # 4) Train model
    # -----------------------------
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # -----------------------------
    # 5) Evaluate
    # -----------------------------
    proba_test = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba_test)),
        "pr_auc": float(average_precision_score(y_test, proba_test)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(np.mean(y_train)),
        "positive_rate_test": float(np.mean(y_test)),
    }
    metrics.update(compute_threshold_metrics(y_test, proba_test, args.threshold))

    # A very useful “business-like” metric: precision in the top 10% risk bucket
    k = max(1, int(0.10 * len(proba_test)))
    top_idx = np.argsort(-proba_test)[:k]
    precision_top10 = float(np.mean(y_test[top_idx]))  # actual positive rate in top bucket
    metrics["precision_top_10pct"] = precision_top10
    metrics["top_10pct_size"] = int(k)

    # -----------------------------
    # 6) Save artifacts
    # -----------------------------
    joblib.dump(model, ARTIFACT_DIR / "model.pkl")
    (ARTIFACT_DIR / "features.json").write_text(json.dumps(features, indent=2))
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    config = {
        "data_path": str(data_path),
        "label_col": args.label_col,
        "id_col": args.id_col if args.id_col in df.columns else None,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "threshold": args.threshold,
        "model_type": "RandomForestClassifier",
        "n_estimators": args.n_estimators,
    }
    (ARTIFACT_DIR / "config.json").write_text(json.dumps(config, indent=2))

    schema = build_data_schema(df, args.label_col, args.id_col)
    (ARTIFACT_DIR / "data_schema.json").write_text(json.dumps(schema, indent=2))

    # Save a realistic sample for demo scoring in the Streamlit app
    # (Keep ID + label here so the app can show them, but scoring will ignore non-feature cols.)
    df.sample(n=min(300, len(df)), random_state=args.random_state).to_csv(
        ARTIFACT_DIR / "sample.csv", index=False
    )

    print("Training complete.")
    print(f"Saved artifacts to: {ARTIFACT_DIR.resolve()}")
    print(f"ROC-AUC={metrics['roc_auc']:.3f} | PR-AUC={metrics['pr_auc']:.3f} | Top10% Precision={metrics['precision_top_10pct']:.3f}")
    print(f"Threshold={metrics['threshold']:.2f} | Precision={metrics['precision']:.3f} Recall={metrics['recall']:.3f} F1={metrics['f1']:.3f}")


if __name__ == "__main__":
    main()

# src/inference.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd


ARTIFACT_DIR = Path("artifacts")


@dataclass(frozen=True)
class ModelBundle:
    model: object
    features: List[str]


def load_bundle(artifact_dir: Path = ARTIFACT_DIR) -> ModelBundle:
    model_path = artifact_dir / "model.pkl"
    feat_path = artifact_dir / "features.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}. Run training first.")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}. Re-train to generate it.")

    model = joblib.load(model_path)
    features = json.loads(feat_path.read_text())
    return ModelBundle(model=model, features=features)


def validate_features(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return (clean_df, warnings). Raises on hard errors."""
    warnings = []

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    extra = [c for c in df.columns if c not in features]
    if extra:
        warnings.append(f"Ignoring extra columns: {extra}")

    X = df[features].copy()

    # Coerce numeric
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    n_missing = int(X.isna().sum().sum())
    if n_missing > 0:
        warnings.append(f"Found {n_missing} missing/non-numeric values; imputing with column median.")
        X = X.fillna(X.median(numeric_only=True))

    return X, warnings


def score(df: pd.DataFrame, bundle: ModelBundle) -> pd.Series:
    X, _ = validate_features(df, bundle.features)
    proba = bundle.model.predict_proba(X)[:, 1]
    return pd.Series(proba, index=df.index, name="risk_score")

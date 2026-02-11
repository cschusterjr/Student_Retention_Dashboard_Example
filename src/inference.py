"""
src/inference.py

Purpose
-------
Centralizes "inference-time" logic (loading artifacts, validating inputs, scoring).
This separation is a strong portfolio signal because it shows:
  - UI concerns (Streamlit) are separated from business logic (ML scoring)
  - Reusable scoring code for future: API service, batch scoring, notebooks, etc.

Artifacts expected in artifacts/:
  - model.pkl        : trained scikit-learn classifier
  - features.json    : list of feature column names used during training
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import joblib
import pandas as pd


# Default folder where training saves model + metadata.
ARTIFACT_DIR = Path("artifacts")


@dataclass(frozen=True)
class ModelBundle:
    """
    A small immutable container that keeps everything needed for scoring together.

    Why bundle?
    ----------
    - Avoids passing around multiple separate objects everywhere.
    - Makes it easy to extend later (e.g., add preprocessor, label map, threshold, etc.)
    """
    model: object
    features: List[str]


def load_bundle(artifact_dir: Path = ARTIFACT_DIR) -> ModelBundle:
    """
    Load model + feature metadata from disk.

    Parameters
    ----------
    artifact_dir : Path
        Directory containing trained model artifacts.

    Returns
    -------
    ModelBundle
        Contains the trained model and the expected feature list.

    Raises
    ------
    FileNotFoundError
        If required artifact files are missing.

    Notes
    -----
    We intentionally fail fast if artifacts are missing because inference without
    a known feature schema is unsafe (silent mis-scoring is worse than a crash).
    """
    model_path = artifact_dir / "model.pkl"
    feat_path = artifact_dir / "features.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {model_path}. Train first (python -m src.train)."
        )

    if not feat_path.exists():
        raise FileNotFoundError(
            f"Missing {feat_path}. Re-train to generate it (python -m src.train)."
        )

    model = joblib.load(model_path)
    features = json.loads(feat_path.read_text())

    # Defensive check: ensure features is a list of strings
    if not isinstance(features, list) or not all(isinstance(x, str) for x in features):
        raise ValueError("features.json is not a valid list of strings.")

    return ModelBundle(model=model, features=features)


def validate_features(df: pd.DataFrame, features: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate and prepare a dataframe for scoring.

    This function:
      1) Ensures required columns are present (hard error if missing).
      2) Drops/ignores unexpected columns (warning only).
      3) Coerces required columns to numeric (non-numeric becomes NaN).
      4) Imputes missing values with median (warning).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe from user upload or other source.
    features : List[str]
        Ordered list of feature columns expected by the model.

    Returns
    -------
    X : pd.DataFrame
        Clean feature matrix containing ONLY model-required features in correct order.
    warnings : List[str]
        Human-readable warnings describing non-fatal issues.

    Raises
    ------
    ValueError
        If required columns are missing.

    Why this matters
    ----------------
    Portfolio-grade scoring requires predictable behavior:
    - Bad inputs should be caught.
    - Non-fatal issues should be communicated.
    - Feature ordering must match training.
    """
    warnings: List[str] = []

    # --- 1) Check required columns exist
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- 2) Identify extra columns (OK to ignore; tell the user)
    extra = [c for c in df.columns if c not in features]
    if extra:
        warnings.append(f"Ignoring extra columns: {extra}")

    # --- 3) Select and order features exactly as expected by the model
    X = df[features].copy()

    # --- 4) Coerce to numeric; non-numeric values become NaN
    for c in features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    # --- 5) Impute missing values
    n_missing = int(X.isna().sum().sum())
    if n_missing > 0:
        warnings.append(
            f"Found {n_missing} missing/non-numeric values; imputing with column median."
        )
        X = X.fillna(X.median(numeric_only=True))

    return X, warnings


def score(df: pd.DataFrame, bundle: ModelBundle) -> pd.Series:
    """
    Compute predicted probability of the positive class (retention risk).

    Parameters
    ----------
    df : pd.DataFrame
        Raw input dataframe that includes required features.
    bundle : ModelBundle
        Loaded model and expected feature list.

    Returns
    -------
    pd.Series
        Risk scores between 0 and 1, aligned to df.index.

    Notes
    -----
    For scikit-learn classifiers with predict_proba:
      - predict_proba(X) returns shape (n_samples, n_classes)
      - [:, 1] corresponds to probability of the positive class
    """
    X, _ = validate_features(df, bundle.features)

    # scikit-learn convention: column 1 is positive class probability
    proba = bundle.model.predict_proba(X)[:, 1]

    return pd.Series(proba, index=df.index, name="risk_score")

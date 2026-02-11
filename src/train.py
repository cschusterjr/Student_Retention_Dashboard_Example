"""
src/train.py

Purpose
-------
Trains a baseline retention-risk model and saves artifacts to artifacts/.

This script is intentionally designed as a simple starting point that can evolve into:
  - training on a real CSV dataset
  - ML pipeline with feature engineering
  - hyperparameter tuning / experiment tracking
  - model registry / deployment

Current behavior
----------------
- Generates synthetic classification data (make_classification)
- Trains a RandomForestClassifier
- Computes evaluation metrics (ROC-AUC, PR-AUC)
- Saves:
    artifacts/model.pkl
    artifacts/features.json
    artifacts/metrics.json
    artifacts/sample.csv

Run
---
python -m src.train
"""

import json
import pathlib

import joblib
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split


# Output folder for artifacts.
OUT = pathlib.Path("artifacts")
OUT.mkdir(exist_ok=True, parents=True)


def main() -> None:
    """
    Main training entrypoint.

    Best practice notes
    -------------------
    - Putting work inside a function makes the module import-safe
      (useful for tests or future refactors).
    - Later you can add argparse for CLI options (e.g., train from CSV).
    """
    # ---------------------------------------------------------------------
    # 1) Create training data
    # ---------------------------------------------------------------------
    # We use synthetic data for demonstration, but we keep the structure
    # similar to a real retention dataset: multiple numeric features + binary label.
    X, y = make_classification(
        n_samples=2000,
        n_features=10,
        n_informative=6,
        class_sep=1.2,
        random_state=42,
    )

    # Feature names: in a real project these would be meaningful (e.g., "days_active_30d")
    cols = [f"f{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df["label"] = y  # convention: 1 = "high risk", 0 = "low risk"

    # ---------------------------------------------------------------------
    # 2) Split into train/test
    # ---------------------------------------------------------------------
    # Use stratify so class proportions are maintained across splits.
    # This is important when classes are imbalanced (common in retention problems).
    X_train, X_test, y_train, y_test = train_test_split(
        df[cols],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    # ---------------------------------------------------------------------
    # 3) Train model
    # ---------------------------------------------------------------------
    # Random Forest is a good baseline:
    # - handles non-linear relationships
    # - requires minimal preprocessing
    # - works well on tabular data
    clf = RandomForestClassifier(
        n_estimators=300,     # number of trees
        random_state=42,      # reproducibility
        n_jobs=-1,            # use all CPU cores to speed up training
    )
    clf.fit(X_train, y_train)

    # ---------------------------------------------------------------------
    # 4) Evaluate model
    # ---------------------------------------------------------------------
    # For risk scoring, the quality of predicted probabilities matters.
    # ROC-AUC: ranking quality across thresholds
    # PR-AUC: useful for imbalanced problems (retention is often imbalanced)
    proba = clf.predict_proba(X_test)[:, 1]
    metrics = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
    }

    # ---------------------------------------------------------------------
    # 5) Save artifacts
    # ---------------------------------------------------------------------
    # Save model
    joblib.dump(clf, OUT / "model.pkl")

    # Save feature schema (critical for safe inference)
    (OUT / "features.json").write_text(json.dumps(cols, indent=2))

    # Save a sample file for demo scoring in the Streamlit app
    df.iloc[:200].to_csv(OUT / "sample.csv", index=False)

    # Save metrics in proper JSON format
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(
        f"Trained.\n"
        f"ROC-AUC={metrics['roc_auc']:.3f} | PR-AUC={metrics['pr_auc']:.3f}\n"
        f"Artifacts saved to: {OUT.resolve()}"
    )


if __name__ == "__main__":
    main()

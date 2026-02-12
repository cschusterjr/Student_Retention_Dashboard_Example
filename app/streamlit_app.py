"""
app/streamlit_app.py

Portfolio-grade Streamlit app with:
- Overview tab (scoring, distribution, risk bands)
- Record Explorer tab (select a student / row and explain locally)
- Explainability tab (global SHAP)
- Model Quality tab (reads artifacts/metrics.json + confusion matrix)

This app assumes you've run:
  python -m src.make_synthetic_data
  python -m src.train
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

from src.inference import load_bundle, score, validate_features
from src.data_dictionary import DATA_DICTIONARY


ART = Path("artifacts")

st.set_page_config(page_title="Retention Risk Dashboard", layout="wide")
st.title("📈 Student Retention Risk Dashboard")


# -----------------------------
# Caching
# -----------------------------
@st.cache_resource
def get_bundle():
    return load_bundle(ART)

@st.cache_resource
def get_explainer(model):
    return shap.TreeExplainer(model)

@st.cache_data
def compute_shap_values(explainer, X: pd.DataFrame):
    return explainer.shap_values(X)

@st.cache_data
def load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def plot_confusion_matrix(cm: Dict[str, int]) -> plt.Figure:
    """
    Plot a simple 2x2 confusion matrix using matplotlib.
    """
    tn, fp, fn, tp = cm["tn"], cm["fp"], cm["fn"], cm["tp"]
    mat = np.array([[tn, fp], [fn, tp]])

    fig = plt.figure()
    plt.imshow(mat)
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    plt.title("Confusion Matrix")
    for (i, j), val in np.ndenumerate(mat):
        plt.text(j, i, str(val), ha="center", va="center")
    plt.xlabel("Prediction")
    plt.ylabel("Actual")
    plt.colorbar()
    return fig


# -----------------------------
# Load model bundle
# -----------------------------
try:
    bundle = get_bundle()
    st.success("✅ Model loaded.")
except Exception as e:
    st.warning("No model bundle found. Train first: `python -m src.train`.")
    st.code(str(e))
    st.stop()


# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.50, 0.01)
max_rows = st.sidebar.slider("Max rows to score", 50, 5000, 500, 50)

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Use `artifacts/sample.csv` for a quick demo if you don’t upload anything.")


# -----------------------------
# Data load
# -----------------------------
st.subheader("Data input")
st.caption(f"Model expects these features: {bundle.features}")

file = st.file_uploader("Upload a CSV (optional)", type="csv")

if file:
    df = pd.read_csv(file)
else:
    sample_path = ART / "sample.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
        st.info("Using artifacts/sample.csv (generated during training).")
    else:
        st.warning("No uploaded file and no artifacts/sample.csv found.")
        st.stop()

df = df.head(max_rows)


# -----------------------------
# Validate + score
# -----------------------------
try:
    X, warnings = validate_features(df, bundle.features)
    for w in warnings:
        st.warning(w)
except Exception as e:
    st.error("Input data failed validation.")
    st.code(str(e))
    st.stop()

risk = score(df, bundle)
out = df.copy()
out["risk_score"] = risk

# Risk band labels (simple but interpretable)
out["risk_band"] = pd.cut(
    out["risk_score"],
    bins=[-0.0001, threshold, 0.75, 1.0],
    labels=[f"Low (<{threshold:.2f})", "Medium", "High"],
)


# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_explorer, tab_explain, tab_quality = st.tabs(
    ["Overview", "Record Explorer", "Explainability", "Model Quality"]
)

# -----------------------------
# Overview tab
# -----------------------------
with tab_overview:
    c1, c2 = st.columns([2, 1])

    with c1:
        st.subheader("Top 50 Highest Risk Records")
        st.dataframe(out.sort_values("risk_score", ascending=False).head(50), use_container_width=True)

        csv = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download scored CSV", data=csv, file_name="scored.csv", mime="text/csv")

    with c2:
        st.subheader("Risk Distribution")
        fig = plt.figure()
        plt.hist(out["risk_score"], bins=30)
        plt.title("Risk Score Distribution")
        plt.xlabel("risk_score")
        plt.ylabel("count")
        st.pyplot(fig)

        st.subheader("Risk Bands")
        st.write(out["risk_band"].value_counts(dropna=False))

    st.markdown("---")
    st.subheader("Data Dictionary (what the columns mean)")
    # Show only the columns that exist in the data dictionary
    rows = []
    for col in out.columns:
        if col in DATA_DICTIONARY:
            rows.append({"column": col, "description": DATA_DICTIONARY[col]})
    if rows:
        st.table(pd.DataFrame(rows))
    else:
        st.info("No data dictionary entries matched your current columns.")

# -----------------------------
# Record Explorer tab (local explainability)
# -----------------------------
with tab_explorer:
    st.subheader("Pick a record and explain its risk score")

    # Allow user to select a row by index or by student_id if present
    if "student_id" in out.columns:
        selected_id = st.selectbox("Select student_id", out["student_id"].astype(str).tolist())
        row_idx = int(out.index[out["student_id"].astype(str) == selected_id][0])
    else:
        row_idx = st.number_input("Row index", min_value=0, max_value=len(out) - 1, value=0, step=1)

    st.write("Selected record:")
    st.dataframe(out.loc[[row_idx]], use_container_width=True)

    st.markdown("---")
    st.subheader("Local SHAP explanation (why this score is high/low)")

    explainer = get_explainer(bundle.model)

    # For speed, compute SHAP for a small batch (e.g., 200 rows) but still explain the chosen row
    X_small = X.head(min(200, len(X)))
    sv = compute_shap_values(explainer, X_small)

    if isinstance(sv, list) and len(sv) == 2:
        sv_pos = sv[1]
    else:
        sv_pos = sv

    # If chosen row isn't in the first 200, explain the closest approach:
    # (For a full solution we'd compute SHAP for that specific row, but this keeps it simple/fast.)
    explain_idx = row_idx if row_idx < len(X_small) else 0
    if row_idx >= len(X_small):
        st.info("Selected row is outside the SHAP sample window; showing explanation for row 0 for speed.")

    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value
    sv_row = sv_pos[explain_idx]
    x_row = X_small.iloc[[explain_idx]]

    fig = plt.figure()
    shap.plots._waterfall.waterfall_legacy(base_value, sv_row, x_row.iloc[0], show=False)
    st.pyplot(fig)

# -----------------------------
# Explainability tab (global)
# -----------------------------
with tab_explain:
    st.subheader("Global explanation (SHAP summary)")

    explainer = get_explainer(bundle.model)
    X_small = X.head(min(500, len(X)))
    sv = compute_shap_values(explainer, X_small)

    if isinstance(sv, list) and len(sv) == 2:
        sv_pos = sv[1]
    else:
        sv_pos = sv

    st.caption("SHAP summary plot (positive class: at-risk)")
    fig2 = plt.figure()
    shap.summary_plot(sv_pos, X_small, show=False)
    st.pyplot(fig2)

# -----------------------------
# Model Quality tab
# -----------------------------
with tab_quality:
    st.subheader("Model Quality (from artifacts/metrics.json)")

    metrics = load_json(ART / "metrics.json")
    config = load_json(ART / "config.json")

    if not metrics:
        st.warning("No metrics found. Train first: python -m src.train")
        st.stop()

    # Show key metrics in a friendly way
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    m2.metric("PR-AUC", f"{metrics.get('pr_auc', 0):.3f}")
    m3.metric("Top 10% Precision", f"{metrics.get('precision_top_10pct', 0):.3f}")
    m4.metric("Threshold", f"{metrics.get('threshold', 0.5):.2f}")

    st.markdown("---")
    st.write("**Threshold-based metrics**")
    st.write(
        {
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
        }
    )

    cm = metrics.get("confusion_matrix")
    if cm:
        st.pyplot(plot_confusion_matrix(cm))

    st.markdown("---")
    st.write("**Training configuration**")
    st.json(config)

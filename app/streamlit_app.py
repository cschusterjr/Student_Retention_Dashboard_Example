"""
app/streamlit_app.py

Purpose
-------
A Streamlit dashboard that:
  - loads a trained model bundle (model + feature schema)
  - accepts a CSV upload for scoring
  - produces risk scores and simple analytics
  - provides SHAP global + local explanations

Why this is portfolio-grade
---------------------------
- Uses cached resources (model + SHAP explainer) for performance
- Validates input schema (prevents silent scoring failures)
- Separates inference logic into src/inference.py (clean architecture)
- Provides both global and local explainability views
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from src.inference import load_bundle, score, validate_features


# ---------------------------------------------------------------------
# Streamlit page configuration
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Retention Risk Dashboard",
    layout="wide",  # wide layout is better for dashboards
)
st.title("📈 Retention Risk Dashboard")

ART = Path("artifacts")


# ---------------------------------------------------------------------
# Caching helpers
# ---------------------------------------------------------------------
# Streamlit reruns the script top-to-bottom on many interactions.
# Caching prevents expensive operations (loading model, SHAP explainer, etc.)
# from being repeated unnecessarily.

@st.cache_resource
def get_bundle():
    """Load model + features once per session (unless code changes)."""
    return load_bundle(ART)

@st.cache_resource
def get_explainer(model):
    """Create SHAP explainer once per session."""
    return shap.TreeExplainer(model)

@st.cache_data
def compute_shap_values(explainer, X: pd.DataFrame):
    """
    Compute SHAP values for a dataset (cached by contents).
    SHAP can be expensive; caching helps keep the app responsive.
    """
    return explainer.shap_values(X)


# ---------------------------------------------------------------------
# Load model bundle (fail fast with useful instructions)
# ---------------------------------------------------------------------
try:
    bundle = get_bundle()
    st.success("✅ Model bundle loaded.")
except Exception as e:
    st.warning("No model bundle found. Train first: `python -m src.train`.")
    st.code(str(e))
    st.stop()


# ---------------------------------------------------------------------
# Sidebar controls: make the dashboard interactive and user-friendly
# ---------------------------------------------------------------------
st.sidebar.header("Controls")

# Threshold lets users define what counts as "high risk".
threshold = st.sidebar.slider(
    "Risk threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.01,
    help="Scores above this value will be labeled as higher risk."
)

# Cap number of rows scored to keep the app fast when users upload big files.
max_rows = st.sidebar.slider(
    "Max rows to score",
    min_value=50,
    max_value=5000,
    value=500,
    step=50,
    help="Limits the number of rows processed for performance."
)


# ---------------------------------------------------------------------
# Data input section
# ---------------------------------------------------------------------
st.subheader("Upload CSV for scoring")

# Communicate what the model expects; reduces user confusion and errors.
st.caption(f"Required columns: {bundle.features}")

file = st.file_uploader("Upload a CSV", type="csv")

if file:
    # User-provided CSV
    df = pd.read_csv(file)
else:
    # If no upload, try to use sample.csv from artifacts (generated at train time).
    st.info("No file uploaded. Using artifacts/sample.csv if available.")
    sample_path = ART / "sample.csv"
    if sample_path.exists():
        df = pd.read_csv(sample_path)
    else:
        # If no sample exists, create an empty dataframe with correct columns.
        # This prevents the app from crashing and provides a helpful signal.
        df = pd.DataFrame({c: pd.Series(dtype="float") for c in bundle.features})

if len(df) == 0:
    st.warning("No rows available to score.")
    st.stop()

# Keep runtime predictable by limiting row count.
df = df.head(max_rows)


# ---------------------------------------------------------------------
# Validate + score
# ---------------------------------------------------------------------
# Validation ensures:
# - required columns exist
# - values are numeric (or coerced)
# - missing values are handled consistently
try:
    X, warnings = validate_features(df, bundle.features)
    for w in warnings:
        st.warning(w)
except Exception as e:
    st.error("Input data failed validation.")
    st.code(str(e))
    st.stop()

# Compute risk scores
risk = score(df, bundle)

# Build an output dataframe that includes scores + bands
out = df.copy()
out["risk_score"] = risk

# Risk banding is a simple but useful way to interpret scores.
out["risk_band"] = pd.cut(
    out["risk_score"],
    bins=[-0.0001, threshold, 1.0],
    labels=[f"Below {threshold:.2f}", f"Above {threshold:.2f}"],
)


# ---------------------------------------------------------------------
# Dashboard outputs (tables + distribution)
# ---------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Scored Records (Top 50 by risk)")
    st.dataframe(out.sort_values("risk_score", ascending=False).head(50), use_container_width=True)

    # Provide downloadable results (important usability feature).
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored CSV",
        data=csv,
        file_name="scored.csv",
        mime="text/csv",
        help="Download the scored dataset including risk_score and risk_band."
    )

with col2:
    st.subheader("Risk Distribution")
    fig = plt.figure()
    plt.hist(out["risk_score"], bins=30)
    plt.title("Risk Score Distribution")
    plt.xlabel("risk_score")
    plt.ylabel("count")
    st.pyplot(fig)

    st.subheader("Risk Band Counts")
    st.write(out["risk_band"].value_counts(dropna=False))


# ---------------------------------------------------------------------
# Explainability (SHAP)
# ---------------------------------------------------------------------
st.subheader("Explainability (SHAP)")

# SHAP explainer is cached; quick after first run.
explainer = get_explainer(bundle.model)

# Compute SHAP values (cached). For very large datasets, consider sampling.
sv = compute_shap_values(explainer, X)

# SHAP for binary classification sometimes returns:
#  - a list of arrays [class0_shap, class1_shap]
# or a single array depending on version/model.
if isinstance(sv, list) and len(sv) == 2:
    sv_pos = sv[1]  # positive class contributions
else:
    sv_pos = sv

st.caption("Global explanation: SHAP summary plot (positive class)")

# SHAP uses matplotlib; summary_plot creates its own figure unless we manage it.
fig2 = plt.figure()
shap.summary_plot(sv_pos, X, show=False)
st.pyplot(fig2)

st.caption("Local explanation: explain a single row")

row_idx = st.number_input(
    "Row index to explain",
    min_value=0,
    max_value=len(X) - 1,
    value=0,
    step=1,
    help="Select which row (0-based index) to explain."
)

x_row = X.iloc[[row_idx]]

# Waterfall plot shows which features increased/decreased predicted risk.
# expected_value can be a list for binary classifiers.
base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, tuple)) else explainer.expected_value
sv_row = sv_pos[row_idx]

st.caption("SHAP waterfall plot for the selected row")
fig3 = plt.figure()
shap.plots._waterfall.waterfall_legacy(base_value, sv_row, x_row.iloc[0], show=False)
st.pyplot(fig3)

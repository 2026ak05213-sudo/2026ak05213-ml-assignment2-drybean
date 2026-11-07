"""Dry Bean classifier evaluation app (pre-trained models, no retraining)."""
import json
import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import compute_metrics

MODEL_DIR = "model"

st.set_page_config(page_title="Dry Bean Classifier Dashboard", page_icon="🌱", layout="wide")

st.markdown(
    """
    <style>
    .banner {background: linear-gradient(90deg, #0f766e, #14b8a6); color: white;
             padding: 1.1rem 1.5rem; border-radius: 10px; margin-bottom: 1rem;}
    .banner h1 {margin: 0; font-size: 1.9rem; color: white;}
    .banner p {margin: 0.3rem 0 0 0; opacity: 0.92; font-size: 0.95rem;}
    div[data-testid="stMetric"] {background: #f1f5f9; border-left: 4px solid #0f766e;
             padding: 0.6rem 0.8rem; border-radius: 6px;}
    div[data-testid="stMetric"] label, div[data-testid="stMetricValue"] {color: #0f172a;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    with open(os.path.join(MODEL_DIR, "meta.json")) as f:
        meta = json.load(f)
    models = {n: joblib.load(os.path.join(MODEL_DIR, p)) for n, p in meta["models"].items()}
    return meta, models


meta, models = load_artifacts()
classes = meta["classes"]
target_col = meta["target_col"]
class_to_idx = {c: i for i, c in enumerate(classes)}

# ----------------------------- Sidebar --------------------------------
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type="csv")
model_name = st.sidebar.selectbox("Select model", list(models.keys()))
st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Logesh S**  \nStudent ID: 2026AK05213  \nAIMLZG565 | M.Tech (AIML)  \nBITS Pilani"
)

# ------------------------------ Main ----------------------------------
st.markdown(
    """
    <div class="banner">
      <h1>Dry Bean Classifier Dashboard</h1>
      <p>ML Assignment 2 &nbsp;|&nbsp; Logesh S &nbsp;|&nbsp; 2026AK05213</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(
    f"Target column: **{target_col}** | Classes: {', '.join(classes)} | "
    "Models are pre-trained; uploaded data is used only for evaluation."
)

if uploaded is None:
    st.info("Upload a CSV containing all feature columns and the target column "
            "(use `test_data.csv` from the repository).")
    st.stop()

df = pd.read_csv(uploaded)

missing = [c for c in meta["feature_cols"] + [target_col] if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()

y_str = df[target_col].astype(str)
unknown = set(y_str) - set(classes)
if unknown:
    st.error(f"Target column has labels unseen during training: {sorted(unknown)}")
    st.stop()

X = df[meta["feature_cols"]]
y_true = y_str.map(class_to_idx).values

with st.expander("Preview uploaded data"):
    st.dataframe(df.head(20))
    st.write(f"{df.shape[0]} rows, {df.shape[1]} columns")

# --------------------------- Selected model ---------------------------
pipe = models[model_name]
y_pred = pipe.predict(X)
y_proba = pipe.predict_proba(X)
scores = compute_metrics(y_true, y_pred, y_proba, len(classes))

st.subheader(f"Evaluation metrics - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, scores.items()):
    col.metric(k, "n/a" if pd.isna(v) else f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with right:
    st.subheader("Classification report")
    report = classification_report(
        y_true, y_pred, labels=list(range(len(classes))),
        target_names=classes, output_dict=True, zero_division=0,
    )
    st.dataframe(pd.DataFrame(report).T.round(3))

# --------------------- All-model comparison on this file ---------------
st.subheader("All models compared on the uploaded data")
rows = {}
for name, m in models.items():
    rows[name] = compute_metrics(y_true, m.predict(X), m.predict_proba(X), len(classes))
comparison = pd.DataFrame(rows).T.round(4)
st.dataframe(comparison.style.highlight_max(axis=0, color="#cfe8cf"))

st.markdown("---")
st.caption("Logesh S | 2026AK05213 | AIMLZG565 - ML Assignment 2")

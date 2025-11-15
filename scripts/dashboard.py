# dashboard_cicids_chunked.py
import os
import io
import math
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_fscore_support
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title="CICIDS-2018 Chunked Dashboard", layout="wide")


 
# CONFIG
 

MODEL_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_rf_model_A.joblib"
SCALER_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_scaler_A.joblib"

DEFAULT_CHUNK_SIZE = 40000  # rows per chunk (RAM safe)


 
# HELPERS
 

def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace("[^a-z0-9_]+", "_", regex=True)
        .str.replace("__+", "_", regex=True)
        .str.strip("_")
    )
    return df


def safe_to_numeric(df):
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)
    return df


  --
# RISK CLASSIFICATION LOGIC
  --
def classify_risk(pred_label, iso_score, high_thresh=-0.2, med_thresh=0.0):
    """
    HIGH:
        - model predicts known attack label
        - OR anomaly score < high_thresh
    MEDIUM:
        - anomaly score < med_thresh
    LOW:
        - benign and normal score
    """
    text = str(pred_label).lower()

    attack_keywords = [
        "attack", "dos", "ddos", "bruteforce", "botnet", "injection",
        "xss", "portscan", "sql", "infiltration", "malware"
    ]
    if any(k in text for k in attack_keywords):
        return "HIGH"

    if iso_score < high_thresh:
        return "HIGH"
    elif iso_score < med_thresh:
        return "MEDIUM"

    return "LOW"


  --
# PLOTS
  --

def plot_confusion_heatmap(y_true, y_pred, classes, figsize=(8, 7), cmap="magma"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig


def plot_pr_bars(y_true, y_pred, classes):
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=classes, zero_division=0
    )
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, pr, width, label="Precision")
    ax.bar(x + width / 2, rc, width, label="Recall")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel("Score")
    ax.set_title("Precision & Recall per class")
    ax.legend()
    plt.tight_layout()
    return fig


def plot_roc_multiclass(y_true, y_score, classes):
    y_bin = label_binarize(y_true, classes=classes)
    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = plt.cm.get_cmap("tab10", n_classes)
    aucs = {}

    for i, cls in enumerate(classes):
        if y_score.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls] = roc_auc
        ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})", color=colors(i))

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    return fig, aucs


 
# LOAD MODEL
 

@st.cache_resource
def load_bundle(mpath, spath):
    bundle = joblib.load(mpath)
    clf = bundle["classifier"]
    iso = bundle.get("isolation_forest", None)
    scaler = joblib.load(spath)
    features = bundle.get("feature_names", [])
    label_encoder = bundle.get("label_encoder", None)
    return clf, iso, scaler, label_encoder, features


st.sidebar.title("Model Loader")
MODEL_PATH = st.sidebar.text_input("Model path", MODEL_PATH)
SCALER_PATH = st.sidebar.text_input("Scaler path", SCALER_PATH)

try:
    classifier, iso_forest, scaler, label_encoder, trained_features = load_bundle(
        MODEL_PATH, SCALER_PATH
    )
    st.sidebar.success("Model & Scaler loaded successfully")
except Exception as e:
    st.sidebar.error(f"Load error: {e}")
    st.stop()


 
# USER INPUT SECTION
 
st.title("CICIDS-2018 Chunked Dashboard (Full Version + Risk Analysis)")

upload = st.file_uploader("Upload CSV", type=["csv"])
path_input = st.text_input("Or enter local CSV path:")
chunk_size = st.number_input("Chunk size", value=DEFAULT_CHUNK_SIZE, min_value=5000)
start = st.button("Start Predicting")

if not start:
    st.stop()


# Determine input source
if path_input:
    if not os.path.exists(path_input):
        st.error("Invalid path")
        st.stop()
    csv_source = ("path", path_input)
elif upload:
    csv_source = ("upload", upload)
else:
    st.error("Provide a dataset!")
    st.stop()


 
# CHUNKED PROCESSING
 

status = st.empty()
progress = st.progress(0)

predictions = []
iso_scores_all = []
true_labels_all = []
ts_all = []
probs_list = []

label_col = None
timestamp_col = None
first_chunk = True
total_rows_seen = 0

# Open CSV in chunks
try:
    if csv_source[0] == "path":
        reader = pd.read_csv(csv_source[1], chunksize=chunk_size, low_memory=True)
    else:
        reader = pd.read_csv(csv_source[1], chunksize=chunk_size, low_memory=True)
except Exception as e:
    st.error(f"CSV load failed: {e}")
    st.stop()

chunk_idx = 0

for chunk in reader:
    chunk_idx += 1
    status.info(f"Processing chunk #{chunk_idx}")
    chunk = normalize_columns(chunk)

    # detect label / timestamp
    if first_chunk:
        cols = list(chunk.columns)
        label_candidates = ["label", "attack_cat", "attack_type", "attack", "type"]
        ts_candidates = [c for c in cols if "time" in c or "timestamp" in c]

        label_col = next((c for c in cols if c in label_candidates), None)
        timestamp_col = ts_candidates[0] if ts_candidates else None
        first_chunk = False

    # store labels
    true_part = chunk[label_col].astype(str).values if label_col else None
    ts_part = chunk[timestamp_col].values if timestamp_col else None

    # build feature chunk
    feat_chunk = chunk.drop(columns=[c for c in [label_col, timestamp_col] if c])
    feat_chunk = safe_to_numeric(feat_chunk)

    # align
    for f in trained_features:
        if f not in feat_chunk.columns:
            feat_chunk[f] = 0

    feat_chunk = feat_chunk[trained_features]

    # scale
    X_scaled = scaler.transform(feat_chunk)

    # predict
    preds_enc = classifier.predict(X_scaled)

    if label_encoder:
        try:
            preds = label_encoder.inverse_transform(preds_enc)
        except:
            preds = preds_enc
    else:
        preds = preds_enc

    # probability
    try:
        probs = classifier.predict_proba(X_scaled)
    except:
        probs = None

    # anomaly
    if iso_forest:
        iso_score = iso_forest.decision_function(X_scaled)
    else:
        iso_score = np.zeros(X_scaled.shape[0])

    predictions.extend(list(preds))
    iso_scores_all.extend(list(iso_score))
    if true_part is not None:
        true_labels_all.extend(list(true_part))
    if ts_part is not None:
        ts_all.extend(list(ts_part))
    if probs is not None:
        probs_list.append(probs)

    total_rows_seen += len(preds)
    progress.progress(min(1.0, chunk_idx * 0.05))

status.success("Processing Complete!")


 
# BUILD OUTPUT DATAFRAME (with risk level)
 

risk_levels = [classify_risk(p, s) for p, s in zip(predictions, iso_scores_all)]

out_df = pd.DataFrame({
    "prediction": predictions,
    "iso_score": iso_scores_all,
    "risk_level": risk_levels
})

if true_labels_all:
    out_df["true_label"] = true_labels_all
if ts_all:
    out_df["timestamp"] = ts_all

# Combine probability matrix
prob_matrix = None
if probs_list:
    prob_matrix = np.vstack(probs_list)


 
# SUMMARY
 

st.header("Summary")
st.write(f"Total rows processed: **{len(out_df):,}**")

st.subheader("Risk Level Distribution")
st.bar_chart(out_df["risk_level"].value_counts())


 
# PDF REPORT GENERATION (NOW WITH RISK SUMMARY)
 

st.header("Export PDF Report")

if st.button("Generate PDF"):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

          
        # PAGE 1 – SUMMARY + RISK DISTRIBUTION
          
        fig1 = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")

        risk_counts = out_df["risk_level"].value_counts()
        summary_text = [
            "CICIDS2018 Chunked Inference Report",
            "",
            f"Rows Processed: {len(out_df):,}",
            f"Unique Predictions: {out_df['prediction'].nunique()}",
            "",
            "Risk Summary:",
        ] + [f"{lvl}: {count:,}" for lvl, count in risk_counts.items()]

        plt.text(0.02, 0.98, "\n".join(summary_text),
                 va="top", fontsize=10, family="monospace")

        # Add bar chart
        ax = fig1.add_axes([0.15, 0.1, 0.7, 0.35])
        ax.bar(risk_counts.index, risk_counts.values, color=["green", "orange", "red"])
        ax.set_title("Risk Level Distribution")
        pdf.savefig(fig1); plt.close(fig1)

          
        # CONFUSION MATRIX
          
        if "true_label" in out_df.columns:
            classes = sorted(out_df["true_label"].unique())
            fig_cm = plot_confusion_heatmap(
                out_df["true_label"], out_df["prediction"], classes
            )
            pdf.savefig(fig_cm); plt.close(fig_cm)

          
        # CLASSIFICATION REPORT
          
        if "true_label" in out_df.columns:
            fig_cr = plt.figure(figsize=(8.27, 11.69))
            plt.axis("off")
            cr_text = classification_report(
                out_df["true_label"], out_df["prediction"],
                zero_division=0
            )
            plt.text(0.02, 0.98, cr_text, va="top",
                     fontsize=9, family="monospace")
            pdf.savefig(fig_cr); plt.close(fig_cr)

          
        # ROC CURVES
          
        if prob_matrix is not None and "true_label" in out_df.columns:
            fig_roc, _ = plot_roc_multiclass(
                out_df["true_label"], prob_matrix, classes
            )
            pdf.savefig(fig_roc); plt.close(fig_roc)

          
        # ANOMALY SCORE HISTOGRAM
          
        fig_iso = plt.figure(figsize=(8, 4))
        plt.hist(out_df["iso_score"], bins=60)
        plt.title("Anomaly Score Distribution")
        pdf.savefig(fig_iso); plt.close(fig_iso)

    buf.seek(0)
    st.download_button("Download PDF Report",
                       data=buf,
                       file_name="cicids_risk_report.pdf",
                       mime="application/pdf")


 
# SAVE CSV
 

csv_path = "cicids_chunked_predictions_with_risk.csv"
out_df.to_csv(csv_path, index=False)
st.success(f"CSV saved: {csv_path}")

with open(csv_path, "rb") as f:
    st.download_button("Download CSV", f, file_name=csv_path)

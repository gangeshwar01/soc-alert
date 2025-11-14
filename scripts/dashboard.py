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

# --------------------
# CONFIG - edit if needed
# --------------------
MODEL_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_rf_model_A.joblib"
SCALER_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_scaler_A.joblib"

DEFAULT_CHUNK_SIZE = 40000  # rows per chunk (tune to your RAM)
# --------------------

# --------------------
# Helper utilities
# --------------------
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
    # replace inf with NaN, then fill numeric cols with mean and rest with 0
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)
    return df

def plot_confusion_heatmap(y_true, y_pred, classes, figsize=(8,7), cmap="magma"):
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title("Confusion Matrix")
    ax.set_xticks(np.arange(len(classes))); ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right'); ax.set_yticklabels(classes)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    return fig

def plot_pr_bars(y_true, y_pred, classes):
    pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=classes, zero_division=0)
    x = np.arange(len(classes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x - width/2, pr, width, label="Precision")
    ax.bar(x + width/2, rc, width, label="Recall")
    ax.set_xticks(x); ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_ylabel("Score"); ax.set_title("Precision & Recall per class"); ax.legend()
    plt.tight_layout()
    return fig

def plot_roc_multiclass(y_true, y_score, classes):
    # y_score: (n_samples, n_classes)
    try:
        y_bin = label_binarize(y_true, classes=classes)
    except Exception as e:
        raise RuntimeError(f"ROC: unable to binarize labels: {e}")

    n_classes = len(classes)
    fig, ax = plt.subplots(figsize=(8,6))
    colors = plt.cm.get_cmap("tab10", n_classes)
    aucs = {}
    for i, cls in enumerate(classes):
        if y_score.shape[1] <= i:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        aucs[cls] = roc_auc
        ax.plot(fpr, tpr, lw=2, label=f"{cls} (AUC={roc_auc:.2f})", color=colors(i))
    ax.plot([0,1],[0,1],"k--", lw=1)
    ax.set_xlim([0,1]); ax.set_ylim([0,1.05])
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Multi-class ROC (One-vs-Rest)")
    ax.legend(loc="lower right", fontsize="small")
    plt.tight_layout()
    return fig, aucs

# --------------------
# Load model & scaler
# --------------------
@st.cache_resource
def load_bundle(mpath, spath):
    bundle = joblib.load(mpath)
    scaler = joblib.load(spath)
    clf = bundle.get("classifier")
    iso = bundle.get("isolation_forest", None)
    label_enc = bundle.get("label_encoder", None)
    features = bundle.get("feature_names", None) or []
    return clf, iso, scaler, label_enc, features

st.sidebar.title("Model")
MODEL_PATH = st.sidebar.text_input("Model path", MODEL_PATH)
SCALER_PATH = st.sidebar.text_input("Scaler path", SCALER_PATH)

try:
    classifier, iso_forest, scaler, label_encoder, trained_features = load_bundle(MODEL_PATH, SCALER_PATH)
    st.sidebar.success("Loaded model + scaler")
except Exception as e:
    st.sidebar.error(f"Failed to load model/scaler: {e}")
    st.stop()

# --------------------
# Inputs
# --------------------
st.title("CICIDS-2018 Chunked Dashboard (ROC / Confusion / Timeline / PDF)")

st.markdown("**Select dataset** (local path recommended for very large CSVs). If uploading through the browser, large files may load into RAM.")
upload = st.file_uploader("Upload CSV (or choose file path below)", type=["csv"], accept_multiple_files=False)
path_input = st.text_input("Or enter local CSV path (preferred for large files):", "")

chunk_size = st.number_input("Chunk size (rows)", value=DEFAULT_CHUNK_SIZE, min_value=5000, step=5000)
start = st.button("Start predictions")

if not start:
    st.stop()

# choose source
if path_input:
    if not os.path.exists(path_input):
        st.error("Local file path does not exist.")
        st.stop()
    csv_source = ("path", path_input)
elif upload is not None:
    # Warning: streaming chunk read from uploaded file may still load entire file into memory in some envs
    csv_source = ("upload", upload)
else:
    st.error("Provide a file (upload) or local path.")
    st.stop()

# --------------------
# Chunked processing
# --------------------
status = st.empty()
progress = st.progress(0)
pbar_text = st.empty()

chunk_iter = None
total_rows_seen = 0
predictions = []
true_labels_all = []
probs_list = []
iso_scores_all = []
ts_all = []

# function to detect label and timestamp columns from first chunk
label_col = None
timestamp_col = None
first_chunk = True
estimated_total_rows = None

# We'll attempt to estimate number of chunks (if local path)
if csv_source[0] == "path":
    try:
        # quick estimate of total lines (could be slow for huge files, so optional)
        with open(csv_source[1], "rb") as fh:
            total_lines = 0
            for i, _ in enumerate(fh, 1):
                total_lines = i
                if i > 2000000:
                    break
        # subtract 1 for header
        estimated_total_rows = max(0, total_lines - 1)
    except Exception:
        estimated_total_rows = None

try:
    if csv_source[0] == "path":
        reader = pd.read_csv(csv_source[1], chunksize=chunk_size, iterator=True, low_memory=True)
    else:
        # uploaded file (streamlit UploadFile)
        upload_file = csv_source[1]
        upload_file.seek(0)
        reader = pd.read_csv(upload_file, chunksize=chunk_size, iterator=True, low_memory=True)
except Exception as e:
    st.error(f"Failed to open CSV in chunked mode: {e}")
    st.stop()

chunk_idx = 0
# We'll compute progress based on estimated_total_rows when available
try:
    for chunk in reader:
        chunk_idx += 1
        status.info(f"Processing chunk #{chunk_idx} (rows ~ {len(chunk):,})")
        pbar_text.text(f"Chunk {chunk_idx}")

        # normalize column names
        chunk = normalize_columns(chunk)

        # detect label, timestamp columns on first chunk
        if first_chunk:
            cols = list(chunk.columns)
            # detect label
            possible_label_cols = [c for c in cols if c in ("label", "attack_cat", "attack_type", "attack", "type")]
            label_col = possible_label_cols[0] if possible_label_cols else None
            # detect timestamp
            possible_ts = [c for c in cols if "time" in c or "timestamp" in c or "date" in c or "ts" == c]
            timestamp_col = possible_ts[0] if possible_ts else None
            first_chunk = False
            status.write(f"Detected label column: {label_col} — timestamp column: {timestamp_col}")

        # Save true label and timestamp (if present) BEFORE we drop/convert
        if label_col and label_col in chunk.columns:
            true_part = chunk[label_col].astype(str).values
        else:
            true_part = None

        if timestamp_col and timestamp_col in chunk.columns:
            ts_part = chunk[timestamp_col].values
        else:
            ts_part = None

        # Drop label/timestamp from features so alignment matches training features
        feat_chunk = chunk.drop(columns=[c for c in (label_col, timestamp_col) if c and c in chunk.columns])

        # convert to numeric safely
        feat_chunk = safe_to_numeric(feat_chunk)

        # align features to trained_features (if model saved feature list)
        if trained_features:
            # ensure all trained features exist
            for f in trained_features:
                if f not in feat_chunk.columns:
                    feat_chunk[f] = 0
            # keep only trained features (order)
            feat_chunk = feat_chunk[trained_features]
        else:
            # if no trained features list available, keep the chunk as-is
            pass

        # scale chunk with scaler (safe)
        try:
            X_scaled = scaler.transform(feat_chunk)
        except Exception as e:
            # attempt robust clean fallback
            feat_chunk = feat_chunk.replace([np.inf, -np.inf], np.nan).fillna(0)
            X_scaled = scaler.transform(feat_chunk)

        # predict
        try:
            preds_enc = classifier.predict(X_scaled)
        except Exception as e:
            st.error(f"Classifier predict failed on chunk {chunk_idx}: {e}")
            preds_enc = np.array(["error"] * X_scaled.shape[0])

        # convert predictions back to labels using label_encoder if available and fitted
        if label_encoder is not None:
            try:
                preds = label_encoder.inverse_transform(preds_enc)
            except Exception:
                # either encoder not fitted or mismatch, just use preds_enc as-is
                preds = preds_enc
        else:
            preds = preds_enc

        # probabilities (for ROC)
        chunk_probs = None
        try:
            chunk_probs = classifier.predict_proba(X_scaled)
        except Exception:
            # try decision_function -> softmax
            try:
                dec = classifier.decision_function(X_scaled)
                # if 1d, make 2d
                if dec.ndim == 1:
                    # binary decision function; create probabilities for two classes
                    exp = np.exp(dec - np.max(dec))
                    probs0 = 1 / (1 + np.exp(dec))
                    chunk_probs = np.vstack([1-probs0, probs0]).T
                else:
                    exp = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                    chunk_probs = exp / np.sum(exp, axis=1, keepdims=True)
            except Exception:
                chunk_probs = None

        # IsolationForest predictions + scores
        if iso_forest is not None:
            try:
                iso_pred = iso_forest.predict(X_scaled)
                iso_score = iso_forest.decision_function(X_scaled)
            except Exception:
                iso_pred = np.ones(X_scaled.shape[0])
                iso_score = np.zeros(X_scaled.shape[0])
        else:
            iso_pred = np.ones(X_scaled.shape[0])
            iso_score = np.zeros(X_scaled.shape[0])

        # append outputs
        predictions.extend(list(preds))
        iso_scores_all.extend(list(iso_score))
        if true_part is not None:
            true_labels_all.extend(list(true_part))
        if ts_part is not None:
            ts_all.extend(list(ts_part))
        if chunk_probs is not None:
            probs_list.append(chunk_probs)

        total_rows_seen += X_scaled.shape[0]

        # update progress
        if estimated_total_rows:
            progress_val = min(1.0, total_rows_seen / estimated_total_rows)
            progress.progress(progress_val)
        else:
            # rough visual progress: advance a little
            progress.progress(min(1.0, min(0.99, total_rows_seen / (chunk_size * 50))))

    status.success("All chunks processed.")
except StopIteration:
    pass
except Exception as e:
    st.error(f"Error while reading chunks: {e}")
    st.stop()

# build results dataframe
out_df = pd.DataFrame({"prediction": predictions, "iso_score": iso_scores_all})
if true_labels_all:
    out_df["true_label"] = true_labels_all
if ts_all:
    out_df["timestamp"] = ts_all

# combine probs (stack)
prob_matrix = None
if probs_list:
    try:
        prob_matrix = np.vstack(probs_list)
    except Exception:
        prob_matrix = None

st.subheader("Summary")
st.markdown(f"- Rows processed: **{len(out_df):,}**")
st.markdown(f"- Unique predicted labels: **{out_df['prediction'].nunique()}**")
if "true_label" in out_df.columns:
    st.markdown(f"- Unique true labels: **{out_df['true_label'].nunique()}**")

# show distribution
st.write("### Prediction distribution")
st.dataframe(out_df["prediction"].value_counts().rename_axis("label").reset_index(name="count"))

# --------------------
# Confusion matrix & classification report
# --------------------
if "true_label" in out_df.columns:
    st.header("Confusion matrix & classification report")
    classes = sorted(list(pd.unique(pd.concat([out_df["true_label"], out_df["prediction"]]).astype(str))))
    try:
        fig_cm = plot_confusion_heatmap(out_df["true_label"].astype(str), out_df["prediction"].astype(str), classes, figsize=(10,8))
        st.pyplot(fig_cm)
    except Exception as e:
        st.error(f"Failed to draw confusion matrix: {e}")

    # classification report
    try:
        report = classification_report(out_df["true_label"].astype(str), out_df["prediction"].astype(str), labels=classes, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report).T
        st.write(report_df)
    except Exception as e:
        st.error(f"Failed to create classification report: {e}")
else:
    st.info("No true labels found in dataset -> confusion and classification report cannot be computed.")

# --------------------
# ROC curves
# --------------------
st.header("ROC Curves (One-vs-Rest)")
if ("true_label" in out_df.columns) and (prob_matrix is not None):
    classes = sorted(list(pd.unique(pd.concat([out_df["true_label"], out_df["prediction"]]).astype(str))))
    # ensure prob_matrix has correct cols
    if prob_matrix.shape[1] < len(classes):
        st.warning("Probability matrix columns fewer than unique labels — ROC may be unreliable.")
    try:
        fig_roc, aucs = plot_roc_multiclass(out_df["true_label"].astype(str).values, prob_matrix, classes)
        st.pyplot(fig_roc)
        st.write("AUC per class:", aucs)
    except Exception as e:
        st.error(f"ROC failed: {e}")
else:
    st.info("ROC requires ground truth labels AND classifier probability scores (predict_proba).")

# --------------------
# Anomaly histogram & timeline
# --------------------
st.header("Anomaly score distribution & Attack timeline")
fig_iso, ax_iso = plt.subplots(figsize=(8,4))
ax_iso.hist(out_df["iso_score"], bins=80)
ax_iso.set_title("IsolationForest score distribution")
ax_iso.set_xlabel("iso_score"); ax_iso.set_ylabel("count")
st.pyplot(fig_iso)

if "timestamp" in out_df.columns:
    st.subheader("Attack frequency timeline")
    # try parse timestamps
    try:
        out_df["__ts"] = pd.to_datetime(out_df["timestamp"], errors="coerce")
        out_df["__ts_floor"] = out_df["__ts"].dt.floor("min")
        line_df = out_df.groupby(["__ts_floor", "prediction"]).size().unstack(fill_value=0)
        st.line_chart(line_df)
    except Exception as e:
        st.error(f"Timeline plotting failed: {e}")
else:
    st.info("No timestamp column detected -> timeline unavailable.")

# --------------------
# Export PDF report
# --------------------
st.header("Export PDF report")
if st.button("Generate PDF report"):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        # Page 1: summary text
        fig_txt = plt.figure(figsize=(8.27, 11.69))
        plt.axis("off")
        lines = [
            "CICIDS-2018 Chunked Dashboard Report",
            f"Rows processed: {len(out_df):,}",
            f"Unique predicted labels: {out_df['prediction'].nunique()}",
            f"Generated: {pd.Timestamp.now()}",
            ""
        ]
        if "true_label" in out_df.columns:
            lines.append(f"Unique true labels: {out_df['true_label'].nunique()}")
        lines += ["", "Top predictions:"] + [f"  {lab}: {cnt:,}" for lab, cnt in out_df["prediction"].value_counts().items()]
        plt.text(0.01, 0.99, "\n".join(lines), va="top", fontsize=10, family="monospace")
        pdf.savefig(fig_txt); plt.close(fig_txt)

        # Confusion matrix
        if "true_label" in out_df.columns:
            fig_cm = plot_confusion_heatmap(out_df["true_label"].astype(str), out_df["prediction"].astype(str), classes)
            pdf.savefig(fig_cm); plt.close(fig_cm)

            # classification report as text
            fig_cr = plt.figure(figsize=(8.27, 11.69))
            plt.axis("off")
            cr_text = classification_report(out_df["true_label"].astype(str), out_df["prediction"].astype(str), labels=classes, zero_division=0)
            plt.text(0.01, 0.99, cr_text, va="top", fontsize=8, family="monospace")
            pdf.savefig(fig_cr); plt.close(fig_cr)

        # ROC
        if ("true_label" in out_df.columns) and (prob_matrix is not None):
            try:
                fig_roc, aucs = plot_roc_multiclass(out_df["true_label"].astype(str).values, prob_matrix, classes)
                pdf.savefig(fig_roc); plt.close(fig_roc)
            except Exception:
                pass

        # PR bars
        if "true_label" in out_df.columns:
            try:
                fig_pr = plot_pr_bars(out_df["true_label"].astype(str), out_df["prediction"].astype(str), classes)
                pdf.savefig(fig_pr); plt.close(fig_pr)
            except Exception:
                pass

        # anomaly histogram
        fig_iso = plt.figure(figsize=(8,4))
        plt.hist(out_df["iso_score"], bins=80)
        plt.title("IsolationForest score distribution")
        pdf.savefig(fig_iso); plt.close(fig_iso)

        # timeline snapshot
        if "timestamp" in out_df.columns:
            try:
                fig_tl = plt.figure(figsize=(10,4))
                line_df.plot(ax=plt.gca(), legend=False)
                plt.title("Attack timeline (per minute)")
                pdf.savefig(fig_tl); plt.close(fig_tl)
            except Exception:
                pass

    buf.seek(0)
    st.success("PDF report ready.")
    st.download_button("Download PDF report", data=buf, file_name="cicids_report.pdf", mime="application/pdf")

# --------------------
# Save predictions CSV
# --------------------
save_path = os.path.join(os.getcwd(), "cicids_chunked_predictions.csv")
out_df.to_csv(save_path, index=False)
st.success(f"Predictions CSV saved: {save_path}")
with open(save_path, "rb") as f:
    st.download_button("Download predictions CSV", f, file_name=os.path.basename(save_path))

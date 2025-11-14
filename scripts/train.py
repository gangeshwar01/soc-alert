# scripts/train_cicids2018_ram.py
"""
RAM-efficient trainer for CICIDS2018 (Option A: train only on available CSVs).
Generates classification report as PNG.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# ----------------- CONFIG -----------------
DATA_DIR = r"D:\RealTime_Alert_Analysis\dataset"
OUT_DIR  = r"D:\RealTime_Alert_Analysis\model"
REPORT_DIR = r"D:\RealTime_Alert_Analysis\report\training_report"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(OUT_DIR, "cicids2018_rf_model_A.joblib")
SCALER_PATH = os.path.join(OUT_DIR, "cicids2018_scaler_A.joblib")
FEATURES_PATH = os.path.join(OUT_DIR, "cicids2018_features_A.joblib")

CHUNK_SIZE = 60000
BALANCE_PER_CLASS = 3000
RANDOM_STATE = 42
LABEL_CANDIDATES = ["label", "attack_type", "attack", "attack_cat", "flowlabel"]


# ----------------- HELPERS -----------------
def save_classification_report_png(report_text, out_path):
    plt.figure(figsize=(12, 8))
    plt.text(0.01, 0.99, report_text, fontsize=12, family="monospace", va="top")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
                  .str.strip()
                  .str.replace('[^a-zA-Z0-9_]+', '_', regex=True)
                  .str.replace('__+', '_', regex=True)
                  .str.strip('_')
                  .str.lower()
    )
    return df


def find_label_column(cols):
    for candidate in LABEL_CANDIDATES:
        for c in cols:
            if c.lower() == candidate:
                return c
    for c in cols:
        low = c.lower()
        if "label" in low or "attack" in low:
            return c
    return None


def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)


# ----------------- MAIN -----------------
def main():
    print("🚀 RAM-efficient CICIDS2018 Trainer (Option A) starting...")

    csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        raise SystemExit("❌ No CSV files found in DATA_DIR")

    print(f"📦 Found {len(csv_files)} CSV files.")
    for f in csv_files:
        print("  -", os.path.basename(f))

    sample_cols = None
    for f in csv_files:
        try:
            df_head = pd.read_csv(f, nrows=5)
            df_head = normalize_columns(df_head)
            sample_cols = df_head.columns.tolist()
            break
        except Exception as e:
            print(f"⚠ couldn't read header of {f}: {e}")

    if sample_cols is None:
        raise SystemExit("❌ Could not read headers from CSV files")

    label_col = find_label_column(sample_cols)
    if not label_col:
        raise SystemExit("❌ No label/attack column detected in CSV headers.")

    print(f"🏷 Found label column: '{label_col}'")

    numeric_candidates = []
    try:
        sample = pd.read_csv(csv_files[0], nrows=1000)
        sample = normalize_columns(sample)
        if label_col not in sample.columns:
            label_col = find_label_column(sample.columns)
            if not label_col:
                raise SystemExit("❌ Label column disappeared after normalization.")
        for c in sample.columns:
            if c == label_col:
                continue
            if is_numeric_series(sample[c]) or pd.to_numeric(sample[c], errors="coerce").notna().any():
                numeric_candidates.append(c)
    except Exception as e:
        print("⚠ Warning scanning numeric candidates:", e)

    if not numeric_candidates:
        numeric_candidates = [c for c in sample_cols if c != label_col]

    print(f"🔢 Numeric column candidates: {len(numeric_candidates)}")

    common_cols = set(numeric_candidates)
    for f in csv_files[1:]:
        try:
            h = normalize_columns(pd.read_csv(f, nrows=2)).columns.tolist()
            common_cols &= set(h)
        except:
            pass
    numeric_cols = sorted(list(common_cols))
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)

    print(f"🔎 Final numeric columns used: {len(numeric_cols)}")

    pool_by_label = defaultdict(list)
    counts = defaultdict(int)

    def clean_numeric_chunk(df_chunk):
        df_chunk = normalize_columns(df_chunk)
        if label_col not in df_chunk.columns and label_col.lower() in df_chunk.columns:
            actual = [c for c in df_chunk.columns if "label" in c or "attack" in c]
            if actual:
                df_chunk = df_chunk.rename(columns={actual[0]: label_col})

        use_cols = [c for c in numeric_cols if c in df_chunk.columns]
        df_num = df_chunk[use_cols].apply(pd.to_numeric, errors="coerce")
        df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_num = df_num.fillna(df_num.median(numeric_only=True)).fillna(0)

        if label_col in df_chunk.columns:
            lab = df_chunk[label_col].astype(str).str.strip()
        else:
            cand = [c for c in df_chunk.columns if "label" in c or "attack" in c]
            lab = df_chunk[cand[0]].astype(str).str.strip() if cand else pd.Series(["Unknown"] * len(df_num))

        return df_num, lab

    print("\n🧩 Building balanced sample pool...")
    for f in csv_files:
        print(f"\n📥 Processing file: {os.path.basename(f)}")
        try:
            for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=True):
                try:
                    df_chunk, lab = clean_numeric_chunk(chunk)
                except Exception as e:
                    print("⚠ chunk cleaning failed:", e)
                    continue

                for lbl, group in df_chunk.groupby(lab):
                    lbl = str(lbl)
                    need = BALANCE_PER_CLASS - counts[lbl]
                    if need <= 0:
                        continue

                    if len(group) <= need:
                        selected = group
                    else:
                        selected = group.sample(n=need, random_state=RANDOM_STATE)

                    pool_by_label[lbl].append(selected)
                    counts[lbl] += len(selected)

                if all(v >= BALANCE_PER_CLASS for v in counts.values() if v > 0):
                    print("✅ Quota reached. Stopping early.")
                    break
        except Exception as e:
            print("⚠ Error iterating file:", e)

    parts = []
    for lbl, dfs in pool_by_label.items():
        if dfs:
            parts.append(pd.concat(dfs, ignore_index=True).assign(_label=lbl))

    if not parts:
        raise SystemExit("❌ Balanced pool empty.")

    balanced_df = pd.concat(parts, ignore_index=True)
    balanced_df = balanced_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    y = balanced_df["_label"].astype(str)
    X = balanced_df.drop(columns=["_label"])

    print(f"\n🎯 Balanced pool: {X.shape[0]} rows, {X.shape[1]} features")
    print("🔢 Label counts:")
    print(y.value_counts())

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    print("\n🌲 Training RandomForest...")
    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    clf.fit(X_train_scaled, y_train)
    print("✅ Training complete.")

    val_preds = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, val_preds)
    print(f"\n📊 Validation accuracy: {acc*100:.2f}%")

    report_text = classification_report(y_val, val_preds, zero_division=0)
    print("\n📄 Classification Report:")
    print(report_text)

    # ---------- SAVE REPORT PNG ----------
    png_path = os.path.join(REPORT_DIR, "classification_report.png")
    save_classification_report_png(report_text, png_path)
    print(f"🖼 Classification Report PNG saved at: {png_path}")
    # --------------------------------------

    X_scaled_all = scaler.transform(X)
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_scaled_all)
    print("✅ IsolationForest trained.")

    bundle = {
        "classifier": clf,
        "isolation_forest": iso,
        "scaler": scaler,
        "feature_names": list(X.columns),
        "label_classes": list(y.unique())
    }
    joblib.dump(bundle, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(list(X.columns), FEATURES_PATH)

    print(f"\n💾 Model saved: {MODEL_PATH}")
    print(f"💾 Scaler saved: {SCALER_PATH}")
    print(f"💾 Feature list saved: {FEATURES_PATH}")
    print("🎉 Training finished.")


if __name__ == "__main__":
    main()

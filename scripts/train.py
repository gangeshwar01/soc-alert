# scripts/train_cicids2018_ram.py
"""
RAM-efficient trainer for CICIDS2018 (Option A: train only on available CSVs).
- Reads CSV files in DATA_DIR (non-recursive) that match '*.csv'
- Streams each CSV by chunks, extracts numeric columns, and builds a balanced
  sample pool with up to BALANCE_PER_CLASS rows per label (keeps memory small).
- Trains a RandomForest on the balanced pool and saves model + scaler + features.

Usage:
    (venv) python scripts/train_cicids2018_ram.py
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

# ----------------- CONFIG -----------------
DATA_DIR = r"D:\RealTime_Alert_Analysis\dataset"          # folder containing your 6 CSVs
OUT_DIR  = r"D:\RealTime_Alert_Analysis\model"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_PATH  = os.path.join(OUT_DIR, "cicids2018_rf_model_A.joblib")
SCALER_PATH = os.path.join(OUT_DIR, "cicids2018_scaler_A.joblib")
FEATURES_PATH = os.path.join(OUT_DIR, "cicids2018_features_A.joblib")

CHUNK_SIZE = 60000         # rows per read_csv chunk (tune smaller if needed)
BALANCE_PER_CLASS = 3000   # target rows per label (total memory ~ classes * this)
RANDOM_STATE = 42
LABEL_CANDIDATES = ["label", "attack_type", "attack", "attack_cat", "flowlabel"]

# ----------------- HELPERS -----------------
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
    # fallback: any column containing 'label' or 'attack'
    for c in cols:
        low = c.lower()
        if "label" in low or "attack" in low:
            return c
    return None

def is_numeric_series(s):
    # consider numeric-like columns (floats and ints); avoid object/string columns
    return pd.api.types.is_numeric_dtype(s)

# ----------------- MAIN -----------------
def main():
    print("🚀 RAM-efficient CICIDS2018 Trainer (Option A) starting...")
    csv_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv")]
    if not csv_files:
        raise SystemExit("❌ No CSV files found in DATA_DIR")

    print(f"📦 Found {len(csv_files)} CSV files (training will use only these).")
    for f in csv_files:
        print("  -", os.path.basename(f))

    # Stage 1: Scan first file header to locate label & numeric column candidates
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
    print(f"🏷 Found label column (guessed): '{label_col}'")

    # We'll discover numeric columns dynamically per chunk, but prefer stable set:
    # Determine numeric candidates from the first CSV by reading small sample
    numeric_candidates = []
    try:
        sample = pd.read_csv(csv_files[0], nrows=1000)
        sample = normalize_columns(sample)
        if label_col not in sample.columns:
            # try lowercased label name if normalization changed it
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
        # fallback: use all columns except label and obvious strings
        numeric_candidates = [c for c in sample_cols if c != label_col]

    print(f"🔢 Numeric column candidates (count): {len(numeric_candidates)}")
    # keep only columns that exist in every CSV header (safe superset)
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

    # Stage 2: Streaming pass to build balanced small pool
    pool_by_label = defaultdict(list)   # store list of DataFrames per label (not huge)
    counts = defaultdict(int)

    # Helper to try to extract and clean numeric portion from a chunk
    def clean_numeric_chunk(df_chunk):
        df_chunk = normalize_columns(df_chunk)
        # ensure label exists
        if label_col not in df_chunk.columns and label_col.lower() in df_chunk.columns:
            # support label normalization
            actual = [c for c in df_chunk.columns if "label" in c or "attack" in c]
            if actual:
                df_chunk = df_chunk.rename(columns={actual[0]: label_col})
        # select numeric columns intersection
        use_cols = [c for c in numeric_cols if c in df_chunk.columns]
        df_num = df_chunk[use_cols].apply(pd.to_numeric, errors="coerce")
        # convert infinities & huge values to NaN
        df_num.replace([np.inf, -np.inf], np.nan, inplace=True)
        # fill with column median (robust) to avoid bias
        df_num = df_num.fillna(df_num.median(numeric_only=True)).fillna(0)
        # label series
        lab = None
        if label_col in df_chunk.columns:
            lab = df_chunk[label_col].astype(str).str.strip()
        else:
            # fallback: try to find likely label col
            cand = [c for c in df_chunk.columns if "label" in c or "attack" in c]
            lab = df_chunk[cand[0]].astype(str).str.strip() if cand else pd.Series(["Unknown"] * len(df_num))
        return df_num, lab

    print("\n🧩 Building balanced sample pool from chunks (this may take a few minutes)...")
    # iterate files and chunks
    for f in csv_files:
        print(f"\n📥 Processing file: {os.path.basename(f)}")
        try:
            for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=True):
                try:
                    df_chunk, lab = clean_numeric_chunk(chunk)
                except Exception as e:
                    print("⚠ chunk cleaning failed:", e)
                    continue

                # for each label in the chunk, pull up to remaining quota
                for lbl, group in df_chunk.groupby(lab):
                    lbl = str(lbl) if pd.notna(lbl) else "Unknown"
                    need = BALANCE_PER_CLASS - counts[lbl]
                    if need <= 0:
                        continue
                    # sample up to 'need' rows from this group
                    grp_df = group
                    if len(grp_df) <= need:
                        selected = grp_df
                    else:
                        selected = grp_df.sample(n=need, random_state=RANDOM_STATE)
                    pool_by_label[lbl].append(selected)
                    counts[lbl] += len(selected)

                # fast-stop if all labels reached quota
                if all(v >= BALANCE_PER_CLASS for v in counts.values() if v > 0):
                    print("✅ All observed labels reached quota. Breaking early.")
                    break
        except Exception as e:
            print("⚠ Error iterating file:", e)

    # Build final balanced DataFrame
    parts = []
    for lbl, dfs in pool_by_label.items():
        if not dfs:
            continue
        parts.append(pd.concat(dfs, ignore_index=True).assign(_label=lbl))
    if not parts:
        raise SystemExit("❌ No data collected into balanced pool (empty).")
    balanced_df = pd.concat(parts, ignore_index=True)
    # if some labels have >BALANCE_PER_CLASS due to sampling overlap, clip
    # shuffle
    balanced_df = balanced_df.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    # label column is _label
    y = balanced_df["_label"].astype(str)
    X = balanced_df.drop(columns=["_label"])
    print(f"\n🎯 Balanced pool built: {X.shape[0]} rows, {X.shape[1]} features")
    print("🔢 Label counts (sample pool):")
    print(y.value_counts())

    # Stage 3: Train on the small balanced set
    # Split for quick validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Scale (fit on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # RandomForest training (keep moderate size)
    print("\n🌲 Training RandomForestClassifier (on balanced pool)...")
    clf = RandomForestClassifier(
        n_estimators=200,
        n_jobs=-1,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
    clf.fit(X_train_scaled, y_train)
    print("✅ RandomForest trained.")

    # Evaluate quickly
    val_preds = clf.predict(X_val_scaled)
    acc = accuracy_score(y_val, val_preds)
    print(f"\n📊 Validation accuracy on held-out balanced pool: {acc*100:.2f}%")
    print("\n📄 Classification report (balanced pool validation):")
    print(classification_report(y_val, val_preds, zero_division=0))

    # Train isolation forest on whole scaled balanced data for anomaly detection
    X_scaled_all = scaler.transform(X)
    iso = IsolationForest(contamination=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    iso.fit(X_scaled_all)
    print("✅ IsolationForest trained on balanced pool.")

    # Persist model bundle
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

    print("\n🎉 Training finished (Option A). Notes:")
    print(" - This model was trained only on the CSV files found in DATA_DIR (you said you had 6 files).")
    print(" - Accuracy above is on a small balanced pool (representative), not the full production distribution.")
    print(" - For better generalization consider training on the full CICIDS2018 / UNSW datasets later.")

if __name__ == "__main__":
    main()

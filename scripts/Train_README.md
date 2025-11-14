# ✅ **README.md for `train.py`**

```markdown
# 🧠 UNSW-NB15 Model Training (train.py)

This document explains how the `train.py` script works and how to run it.  
The script trains a **RandomForest attack classifier** and an **IsolationForest anomaly detector** using the **UNSW-NB15 dataset**, and saves:

- A preprocessing **StandardScaler**
- A combined **model bundle** (`RandomForest + IsolationForest + LabelEncoder + Feature Names`)
- Accuracy charts and reports (generated externally)

---

## 📌 Overview

`train.py` performs the complete machine learning pipeline:

1. Load UNSW-NB15 training and testing datasets  
2. Preprocess features  
3. One-hot encode categorical columns  
4. Balance classes using resampling  
5. Scale numeric features  
6. Train:
   - **RandomForestClassifier** for attack classification  
   - **IsolationForest** for anomaly detection  
7. Save the final model bundle and scaler

The trained models are stored in:

```

report/training_report/

```

---

## 📂 Input & Output Paths

### **Input Parquet Files**
```

UNSW_NB15_training-set.parquet
UNSW_NB15_testing-set.parquet

```

### **Output (Auto-created)**
```

unsw15_model_v1.joblib
unsw15_scaler_v1.joblib

```

Both stored in:

```

report/training_report/

````

These files should be tracked with **Git LFS**.

---

## 🔧 How to Run the Training Script

### **1. Activate your virtual environment**
```bash
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac
````

### **2. Run the script**

```bash
python scripts/train.py
```

### ✔ Output Logs

You will see logs like:

```
📥 Loading UNSW-NB15 datasets...
🔤 One-Hot Encoding categorical columns...
⚖️ Balancing dataset...
📏 Scaling features...
🌲 Training RandomForest...
🤖 Training IsolationForest...
💾 Model saved: unsw15_model_v1.joblib
```

---

## 🏗️ Pipeline Details

### **1. Load Data**

The script loads:

```python
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
```

Missing attack labels are replaced with `"Normal"`.

---

### **2. Remove Unused Columns**

Dropped columns:

* `attack_cat` (label)
* `label` (binary flag)

---

### **3. One-Hot Encode Categorical Features**

Target columns:

```python
categorical_cols = ["proto", "service", "state"]
```

Encoding:

```python
pd.get_dummies(..., dummy_na=True)
```

The test set is aligned to train columns to avoid shape mismatch.

---

### **4. Balance the Dataset**

Each attack category is resampled to **5000 records**:

* If class > 5000 → undersample
* If class < 5000 → oversample

This ensures balanced training.

---

### **5. Standardization**

Uses **StandardScaler** to normalize numeric features.

Saved as:

```
unsw15_scaler_v1.joblib
```

---

### **6. Encode Labels**

LabelEncoder converts string attack categories into integers.

---

### **7. Train Models**

#### **RandomForest Classifier**

```python
RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    n_jobs=-1
)
```

#### **IsolationForest**

```python
IsolationForest(contamination=0.05)
```

This detects anomalies independent of attack categories.

---

### **8. Save Model Bundle**

The final `joblib` file contains:

```python
{
    "classifier": rf,
    "isolation_forest": iso,
    "label_encoder": le,
    "feature_names": list(X_bal.columns),
}
```

Saved to:

```
unsw15_model_v1.joblib
```

---

## 📦 Model Files

| File                      | Description                                                    |
| ------------------------- | -------------------------------------------------------------- |
| `unsw15_model_v1.joblib`  | RandomForest + IsolationForest + LabelEncoder + column mapping |
| `unsw15_scaler_v1.joblib` | StandardScaler used on all numerical features                  |


# 🔐 **CICIDS2018 Threat Detection Dashboard**

### Streamlit-Based Real-Time Attack Classification, Anomaly Detection & Risk Scoring

This dashboard delivers a complete solution for loading large CICIDS2018 datasets in **chunked streaming mode**, preprocessing them, running RandomForest-based attack predictions, generating IsolationForest anomaly scores, computing **risk levels**, exporting enhanced CSV outputs, and producing an optional multi-page PDF analytics report.

The dashboard combines prediction, anomaly detection, risk scoring, visualization, and reporting — optimized for real-time security analysis and large-file performance.

---

## 🚀 **Features**

### ✔ Model-Based Predictions (CICIDS2018)

Automatically loads your trained components:

| File                           | Purpose                                     |
| ------------------------------ | ------------------------------------------- |
| `cicids2018_rf_model_A.joblib` | RandomForest attack classifier              |
| `cicids2018_scaler_A.joblib`   | StandardScaler used during training         |
| `cicids2018_features_A.joblib` | Final numeric feature list used in training |

Supports:

* `predict()` – attack/benign classification
* `predict_proba()` – probability distribution
* IsolationForest anomaly detection
* **Risk-Level scoring (LOW / MEDIUM / HIGH)**

Risk is computed using:

* Model-predicted label
* IsolationForest anomaly score thresholds

---

## 🎛 **Dashboard Panels & Analytics**

### **1. Dataset Overview**

* Load input through file upload or local system path
* Chunk-based streaming (RAM-safe)
* Cleaned/normalized column names
* Column and label detection
* Preview, missing values, class distribution

---

### **2. Classification & Confusion Matrix**

* Per-row predictions
* Probability scores (if available)
* Confusion matrix heatmap
* Classification report (precision, recall, F1)
* Precision/Recall/F1 bar charts

---

### **3. Anomaly Detection**

* IsolationForest anomaly score
* Histogram of anomaly distribution
* Outlier frequency summary

---

### **4. Timeline & Trends**

If a timestamp column exists (or synthetic timestamps generated):

* Attack frequency per minute
* Line chart for temporal attack visualization

---

### **5. Risk-Level Analysis (NEW)**

* Risk classification: **LOW**, **MEDIUM**, **HIGH**
* Risk summary
* Risk-level bar chart
* Included in CSV and PDF outputs

---

### **6. Export Options**

#### ✔ Enhanced Predictions CSV

Exports:

```
prediction, iso_score, risk_level, true_label, timestamp
```

#### ✔ Multi-Page PDF Report (Optional)

Includes:

* Summary page
* Risk summary + bar chart
* Confusion matrix
* Classification report
* ROC curves
* PR bar charts
* Anomaly score histogram
* Timeline (if timestamps available)

---

## 📥 **Supported Input Formats**

* `.csv` (recommended)
* File upload or local filesystem
* Very large datasets processed in streaming chunks
* Automatic numeric conversion
* Auto-detection of label and timestamp columns

If timestamp missing → synthetic timestamps can be generated for visualization only.

---

## 📂 **Required Model Files**

Default paths:

```python
MODEL_PATH  = r"D:\RealTime_Alert_Analysis\model\cicids2018_rf_model_A.joblib"
SCALER_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_scaler_A.joblib"
FEATURES_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_features_A.joblib"
```

Generated using:

```
scripts/train.py
```

Model bundle includes:

* RandomForest classifier
* IsolationForest model
* StandardScaler
* Full training feature list
* Label encoder (if used)

---

## 🧠 **Internal Pipeline – How It Works**

### 1️⃣ Load Model & Scaler

Cached using:

```python
@st.cache_resource
```

### 2️⃣ Chunked Dataset Loading

* Fully RAM-efficient
* Handles multi-GB CSV files
* Processes chunk-by-chunk until complete

### 3️⃣ Normalize Column Names

* Lowercase
* Remove symbols
* Standardized for stability

### 4️⃣ Auto-Detect Key Columns

* Label column (`label`, `attack_cat`, etc.)
* Timestamp column (if present)

### 5️⃣ Feature Extraction & Alignment

* Align cols to training feature list
* Add missing cols with 0
* Remove unknown/unnecessary cols

### 6️⃣ Preprocessing & Prediction

* Scale using saved StandardScaler
* RandomForest → predicted labels
* IsolationForest → anomaly score
* Compute risk level

### 7️⃣ Construct Final Output

DataFrame contains:

| Column     | Meaning                         |
| ---------- | ------------------------------- |
| prediction | Final class label               |
| iso_score  | IsolationForest anomaly score   |
| risk_level | LOW / MEDIUM / HIGH             |
| true_label | (optional) dataset ground truth |
| timestamp  | (optional) real/synthetic time  |

### 8️⃣ Visualizations

* Confusion Matrix
* ROC Curves (one-vs-rest)
* Precision/Recall/F1
* Anomaly histogram
* Timeline chart
* Risk-level graph

---

## ▶ **Running the Dashboard**

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Run Streamlit:

```bash
streamlit run scripts/dashboard.py
```

Opens at:

```
http://localhost:8501
```

---

## 📊 **Generated Output Files**

| File                                       | Description                               |
| ------------------------------------------ | ----------------------------------------- |
| `cicids_chunked_predictions_with_risk.csv` | With predictions + risk + anomaly scores  |
| `classification_report.png`                | Training report generated during training |
| `plots/*.png`                              | (Optional) visualization exports          |
| `cicids_risk_report.pdf`                   | Multi-page detailed analytics PDF         |

---

## 📌 Important Notes

* Incoming data is automatically normalized and aligned.
* If true labels are missing, evaluation panels (confusion/ROC) are disabled.
* Synthetic timestamps are only used for visualization, not saved or modified.
* Optimized for **very large datasets**, avoiding RAM spikes.
* Models must be generated using your provided `train.py` script to ensure compatibility.

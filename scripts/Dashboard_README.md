````markdown
# 🔐 CICIDS2018 Threat Detection Dashboard  
### Streamlit-Based Real-Time Attack Classification & Visualization

This dashboard provides an interactive interface for loading datasets, preprocessing them, generating predictions using the CICIDS2018 trained model bundle, visualizing metrics, and exporting analytic reports.

The application integrates classification, anomaly scoring, visualization panels, and dataset inspection — all in a unified web dashboard.

---

## 🚀 Features

### ✔ Model-Based Predictions (CICIDS2018)
The dashboard automatically loads your saved training outputs:

| File | Purpose |
|------|---------|
| `cicids2018_rf_model_A.joblib` | RandomForest classifier |
| `cicids2018_scaler_A.joblib` | StandardScaler used during training |
| `cicids2018_features_A.joblib` | List of numeric feature names |

Supports:
- `predict()`
- `predict_proba()` (probabilities per label)
- IsolationForest anomaly detection (from model bundle)

---

## 🎛 Dashboard Tabs

The UI includes multiple analytics tabs (depending on your implementation):

### **1. Dataset Overview**
- Uploaded dataset summary  
- Preview first rows  
- Missing value stats  
- Number of features  
- Detected label column  
- Class distribution plot (if label exists)

### **2. Classification & Confusion Matrix**
- Predicted labels
- Probability scores
- Confusion matrix heatmap  
- Classification report table  
- Metrics (precision / recall / f1-score)

### **3. Anomaly Detection**
- IsolationForest anomaly scores  
- Score histogram  
- Outlier distribution  

### **4. Timeline & Trends (If timestamp exists)**
- Attack count per minute  
- Synthetic timestamp generator if missing  
- Line/bar charts of attack frequency  

### **5. Export**
- Export predictions as CSV  
- Download confusion matrix / plots  
- Generate PDF report (if implemented)

---

## 📥 Supported Input Formats

You can load:

- `.csv` datasets  
- `.parquet` datasets  
- File uploads via Streamlit  
- Local dataset paths  

If dataset has no timestamp column, the dashboard can generate a **synthetic timeline** for visual charts.

---

## 📂 Required Model Files

Expected model bundle paths (default):

```python
MODEL_PATH  = r"D:\RealTime_Alert_Analysis\model\cicids2018_rf_model_A.joblib"
SCALER_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_scaler_A.joblib"
FEATURES_PATH = r"D:\RealTime_Alert_Analysis\model\cicids2018_features_A.joblib"
````

These match the output of your `scripts/train.py`.

Model bundle contains:

* RandomForest model
* IsolationForest model
* StandardScaler
* Feature list for alignment
* Class label list

---

## 🧠 How It Works (Pipeline Summary)

### **1. Load Model & Scaler**

Cached using:

```python
@st.cache_resource
```

### **2. Load Dataset**

Supports:

* Upload
* File path
* CSV or Parquet

### **3. Normalize Column Names**

Lowercasing
Remove spaces
Remove special characters

### **4. Detect Label Column**

Supports:

* `label`
* `attack_type`
* `flowlabel`
* `attack_cat`
* “label-like” columns

### **5. Extract Feature Columns**

Using:

```
cicids2018_features_A.joblib
```

Dashboard:

* Aligns user dataset to training features
* Drops extra columns
* Adds missing columns (filled with 0)

### **6. Scaling & Prediction**

* StandardScaler → `transform()`
* RandomForest → predicted labels
* Probability vector via `predict_proba()`
* IsolationForest anomaly score

### **7. Generate Final Result DataFrame**

Includes:

* Predicted labels
* Probabilities
* Anomaly scores
* Timestamp (real or synthetic)
* Input features

### **8. Visualize & Export**

Charts & metrics available under multiple tabs.

---

## ▶ Running the Dashboard

### **1. Install dependencies**

```bash
pip install -r requirements.txt
```

### **2. Run Streamlit**

```bash
streamlit run scripts/dashboard.py
```

This launches the UI at:

```
http://localhost:8501
```

---

## 📊 Output Files

| File                        | Description                                    |
| --------------------------- | ---------------------------------------------- |
| `predictions.csv`           | Output predictions with anomaly scores         |
| `classification_report.png` | Training report stored in repo/training_report |
| `plots/*.png` (optional)    | Exported visualizations                        |
| `report.pdf` (optional)     | Multi-page analytics PDF                       |

---

## 📌 Notes

* The dashboard automatically standardizes user data to match training features.
* If the dataset contains no labels, evaluation tabs are disabled.
* If timestamps are missing, synthetic timestamps are generated only for visualization.
* Works exclusively with CICIDS2018-trained models produced by your `train.py`.

---

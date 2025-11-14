Below is a **professional, clear, and complete README.md specifically for your `dashboard.py` (dashboard_unsw_full.py) Streamlit dashboard**.
It explains the purpose, features, input/output workflow, configuration, and usage.

Fully tailored to your code.

---

# ✅ **README.md for `dashboard.py` (UNSW-NB15 Dashboard)**

````markdown
# 🔐 UNSW-NB15 Full Analysis Dashboard  
### Streamlit-based Real-Time Threat Analysis & Visualization

This dashboard provides a full interactive interface for analyzing, visualizing, and validating predictions from the UNSW-NB15 machine learning model.  
It integrates classification, anomaly detection, ROC curves, confusion matrices, PDF reporting, and real-time visualization in one unified tool.

---

## 🚀 Features

### ✔ Model-Based Predictions
- Loads the trained model bundle (`unsw15_model_v1.joblib`)
- Loads the feature scaler (`unsw15_scaler_v1.joblib`)
- Supports:
  - RandomForest prediction
  - IsolationForest anomaly scoring
  - Probability-based predictions (`predict_proba`)
  - Auto-softmax fallback when `predict_proba` is unavailable

---

### ✔ Interactive Visual Analytics
The dashboard includes 5 fully interactive tabs:

1. **Overview**
   - Dataset summary  
   - Feature count  
   - Class distribution visualization  
   - Anomaly ratio  

2. **Confusion Matrix & Heatmap**
   - Beautiful heatmap with annotations  
   - Classification report table  

3. **ROC & Metrics**
   - Multi-class ROC (One-vs-Rest)
   - AUC values per class
   - Precision/Recall/F1 Bar Charts  

4. **Anomaly & Timeline**
   - IsolationForest anomaly score histogram  
   - Attack timeline grouped by minute  
   - Auto-generated synthetic timestamps if missing  

5. **Export / Report**
   - Generates an automated PDF containing:
     - Summary
     - Confusion matrix
     - ROC curves
     - Precision/Recall graphs
     - Anomaly histograms
     - Timeline graphs  

---

## 📥 Supported Input Formats

The dashboard accepts:

- `.csv` files  
- `.parquet` files  
- Dataset paths from local filesystem  
- Uploaded files via Streamlit UI  

If no timestamp exists, the dashboard offers to generate a **synthetic timestamp** for timeline charts.

---

## 📂 File Requirements

### **Required model files**

| File | Purpose |
|------|---------|
| `unsw15_model_v1.joblib` | Contains RandomForest, IsolationForest, LabelEncoder & trained feature list |
| `unsw15_scaler_v1.joblib` | StandardScaler used during training |

These must match your `train.py` output.

---

## ⚙ Configuration

The code uses the following editable defaults:

```python
MODEL_PATH  = r"D:\RealTime_Alert_Analysis\report\training_report\unsw15_model_v1.joblib"
SCALER_PATH = r"D:\RealTime_Alert_Analysis\report\training_report\unsw15_scaler_v1.joblib"
````

Both paths can be changed via **Streamlit sidebar**.

### **Categorical columns**

The dashboard must use the same categorical columns as training:

```python
CATEGORICAL_COLS = ["proto", "service", "state"]
```

---

## 🧠 How It Works (Pipeline Summary)

### **1. Load Model & Scaler**

cached using:

```python
@st.cache_resource
```

### **2. Load Dataset**

Accepts:

* uploaded file
* or path from text input

### **3. Normalize column names**

Lowercase, remove illegal characters, fix spacing, etc.

### **4. Detect and remove label column**

Supports:

```
label, attack_cat, attack_type, attack
```

### **5. One-hot encode categorical features**

* Uses `pd.get_dummies()`
* Aligns to training feature list
* Adds missing columns (default = 0)
* Removes extra columns

### **6. Ensure numerical format**

Converts all values to numbers and fills missing values.

### **7. Scale & Predict**

* Scaler → `transform()`
* RandomForest → `predict()` + `predict_proba()`
* IsolationForest → `predict()` + `decision_function()`

### **8. Generate results DataFrame**

Includes:

* predicted label
* anomaly score
* true label (if available)
* timestamp (real or synthetic)

### **9. Display analytics & plots**

### **10. Export:

* CSV predictions
* PDF report (multi-page)**

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

Your browser will open automatically.

---

## 📊 Output Files

| File                   | Description                                        |
| ---------------------- | -------------------------------------------------- |
| `unsw_predictions.csv` | Model + anomaly predictions for the loaded dataset |
| `unsw_report.pdf`      | Auto-generated multi-page analysis report          |

---

## 📦 Notes

* The dashboard gracefully handles missing labels (ROC & confusion matrix disabled).
* Missing timestamps are auto-generated for timeline charts.
* The input dataset will be normalized and aligned to training features.
* Compatible with any UNSW-NB15–based dataset trained using the same script.

---


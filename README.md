```markdown
# 🔥 RealTime Alert Analysis (CICIDS2018)

A complete machine learning pipeline for **real-time intrusion detection** using the **CICIDS2018 dataset**.  
This project supports RAM-efficient model training, feature preprocessing, attack classification, anomaly detection, **risk-level scoring**, CSV export, and multi-page **PDF report generation**, along with a full Streamlit dashboard for real-time analysis.

---

## 📂 Project Structure

```

REALTIME_ALERT_ANALYSIS/
│
├── dataset/
│   └── 02-16-2018.csv                    # CICIDS2018 part file(s)
│
├── model/
│   ├── cicids2018_features_A.joblib      # Saved training feature list
│   ├── cicids2018_rf_model_A.joblib      # Trained RandomForest model
│   └── cicids2018_scaler_A.joblib        # StandardScaler used during training
│
├── notebook/
│   └── train.ipynb                       # Exploratory notebook for training/testing
│
├── report/
│   ├── test_report/
│   │   └── unsw_report.pdf               # Reference example report
│   │
│   └── training_report/
│       └── classification_report.png     # Auto-generated classification report
│
├── scripts/
│   ├── dashboard.py                      # Streamlit dashboard with risk scoring & PDF
│   └── train.py                          # RAM-efficient model trainer (CICIDS2018)
│
├── venv/                                 # Virtual environment (local)
│
├── requirements.txt
├── .gitattributes
├── .gitignore
├── LICENSE
└── README.md

```

---

## 🚀 Features

### **✔ RAM-Efficient ML Training**
The training pipeline (`scripts/train.py`) provides:

- Chunked CSV loading (streaming input)
- Automatic label & numeric feature detection
- Balanced sampling of classes
- StandardScaler preprocessing
- RandomForest classification
- IsolationForest anomaly scoring
- Model bundle saved using `joblib`
- Classification report auto-generated as **PNG**

---

## ✔ Auto-Generated Reports

### **Training**
During training, the following report is created:

```

report/training_report/classification_report.png

```

### **Dashboard (NEW)**

Using the Streamlit dashboard, the following files are produced:

| File                                              | Description                                          |
|--------------------------------------------------|------------------------------------------------------|
| `cicids_chunked_predictions_with_risk.csv`       | Predictions + anomaly scores + **risk levels**       |
| `cicids_risk_report.pdf`                         | Multi-page analysis report with risk summary         |
| `plots/*.png`                                    | Exported charts (optional)                           |

---

## 🎛 Streamlit Dashboard Capabilities

The dashboard (`scripts/dashboard.py`) now supports:

### ✔ Real-Time Prediction
- Processes huge CSV files in **chunks**
- Predicts attack labels using RandomForest
- Computes IsolationForest anomaly score

### ✔ **Risk Level Classification (NEW)**
Automatically assigns:

- **LOW risk**
- **MEDIUM risk**
- **HIGH risk**

Based on:
- Attack prediction label  
- IsolationForest anomaly score thresholds  

Included in:
- UI  
- CSV export  
- PDF summary  

### ✔ Enhanced Visual Analytics
Includes:

- Confusion matrix heatmap
- Classification report table
- ROC curves (one-vs-rest)
- Precision/Recall bar charts
- Anomaly score histograms
- Attack timeline plots (if timestamp exists)

### ✔ Multi-Page PDF Report (NEW)
Generated PDF includes:

- Summary statistics  
- Risk level summary + bar chart  
- Confusion matrix  
- Classification report  
- ROC curves  
- PR bar charts  
- Anomaly histogram  
- Timeline graph (if timestamp exists)  

### ✔ CSV Export (NEW)
Exports:

```

prediction, iso_score, risk_level, true_label, timestamp

````

---

## 🧰 Tech Stack

| Purpose        | Technology |
|----------------|------------|
| ML Training    | Python, Scikit-Learn |
| Feature Scaling | StandardScaler |
| Visualization  | Matplotlib |
| Dashboard UI   | Streamlit |
| Data Handling  | Pandas, NumPy |
| File Storage   | Git LFS |
| Reporting      | PDF (Matplotlib + PdfPages) |

---

## ⚙️ Setup & Installation

### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/RealTime_Alert_Analysis.git
cd RealTime_Alert_Analysis
````

### **2. Create Virtual Environment**

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux/Mac
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Install Git LFS (Required for Models)**

```bash
git lfs install
git lfs pull
```

---

## 🧠 Training the Model

Run:

```bash
python scripts/train.py
```

This will:

* Load CICIDS2018 CSVs
* Stream using RAM-efficient chunks
* Train RandomForest + IsolationForest
* Save model + scaler + feature list
* Generate classification_report.png

Saved outputs:

```
model/
report/training_report/
```

---

## ▶️ Run the Streamlit Dashboard

Start dashboard:

```bash
streamlit run scripts/dashboard.py
```

Server opens at:

```
http://localhost:8501
```

Dashboard provides:

* Real-time threat classification
* Probability + anomaly + **risk scoring**
* Visualization panels
* PDF export
* Chunk-safe processing

---

## 📈 Example Output Files

### **Training Report (PNG)**

```
report/training_report/classification_report.png
```

### **Enhanced Predictions CSV**

```
cicids_chunked_predictions_with_risk.csv
```

### **PDF Analytics Report**

```
cicids_risk_report.pdf
```

---

## ⭐ Notes

* Dashboard auto-aligns incoming data with training features.
* If true labels missing → ROC/Confusion disabled.
* Synthetic timestamps created only for visualization.
* Designed for **large datasets** using chunk streaming.
* All risk scoring logic is transparent and adjustable.

---


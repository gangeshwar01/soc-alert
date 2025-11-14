```markdown
# 🔥 RealTime Alert Analysis (CICIDS2018)

A complete machine learning pipeline for **real-time intrusion detection** using the **CICIDS2018 dataset**.  
This project supports RAM-efficient training, preprocessing, model generation, anomaly detection, evaluation reports (PNG), and a Streamlit-based dashboard for real-time security alert analysis.

---

## 📂 Project Structure

```

REALTIME_ALERT_ANALYSIS/
│
├── dataset/
│   └── 02-16-2018.csv                    # CICIDS2018 part file(s)
│
├── model/
│   ├── cicids2018_features_A.joblib      # Saved feature list
│   ├── cicids2018_rf_model_A.joblib      # Trained RandomForest model
│   └── cicids2018_scaler_A.joblib        # Scaler used during training
│
├── notebook/
│   └── train.ipynb                       # Exploratory training/testing notebook
│
├── report/
│   ├── test_report/
│   │   └── unsw_report.pdf               # Example reference report
│   │
│   └── training_report/
│       └── classification_report.png     # Auto-generated classification report
│
├── scripts/
│   ├── dashboard.py                      # Streamlit dashboard for live predictions
│   └── train.py                          # RAM-efficient training script (CICIDS2018)
│
├── venv/                                 # Virtual environment
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
Your training script (`scripts/train.py`) includes:

- Stream-based CSV loading (chunked processing)
- Automatic numeric column detection
- Automatic label column detection
- Balanced dataset sampling per class
- Scaling → RandomForest training
- Isolation Forest anomaly detector
- Model bundle saved using `joblib`
- Classification report saved as **PNG**

### **✔ Auto-Generated Image Reports**
Training automatically produces:

- `classification_report.png`  
  Stored under:  
```

report/training_report/classification_report.png

````

### **✔ Streamlit Dashboard**
The dashboard (`scripts/dashboard.py`) provides:

- Real-time intrusion prediction
- Probability/Risk scores
- Feature visualizations
- JSON / Manual input support

### **✔ Model Bundle Files**
Saved inside `model/`:

- `cicids2018_rf_model_A.joblib`
- `cicids2018_scaler_A.joblib`
- `cicids2018_features_A.joblib`

### **✔ Clean Project & Reproducibility**
- Git LFS ready for `.joblib` files  
- Reproducible experiments  
- Organized folder structure  

---

## 🧰 Tech Stack

| Purpose        | Technology |
|----------------|------------|
| ML Training    | Python, Scikit-Learn |
| Feature Scaling | StandardScaler |
| Dashboard      | Streamlit |
| Plotting       | Matplotlib |
| Data Handling  | Pandas, NumPy |
| File Storage   | Git LFS |
| Environment    | venv |

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

### **4. Install Git LFS (Required for .joblib Models)**

```bash
git lfs install
git lfs pull
```

---

## 🧠 Training the Model

Run training:

```bash
python scripts/train.py
```

This script will:

✔ Load CICIDS2018 CSVs
✔ Stream & balance data
✔ Train RandomForest
✔ Train Isolation Forest
✔ Save model bundle
✔ Generate PNG classification report

All saved output is stored here:

```
model/
report/training_report/
```

---

## ▶️ Run the Streamlit Dashboard

```bash
streamlit run scripts/dashboard.py
```

Open the URL displayed in your terminal, typically:

```
http://localhost:8501
```

The dashboard provides:

* Real-time attack classification
* Interactive UI
* Probability visualizations
* JSON/Row-level input

---

## 📈 Example Saved Output

### **Classification Report (PNG)**

Generated after training:

```
report/training_report/classification_report.png
```

---


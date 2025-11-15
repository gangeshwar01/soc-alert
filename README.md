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
│   ├── dashboard.py                      # Streamlit dashboard (chunked inference, export, PDF)
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

## 🚀 Highlights & Features

### ✔ RAM-Efficient ML Training (`scripts/train.py`)

* Chunked CSV loading (streaming to keep memory low)
* Automatic label and numeric-feature detection
* Balanced per-class sampling to build a small representative training pool
* StandardScaler preprocessing
* RandomForest classifier (persisted with `joblib`)
* IsolationForest anomaly detector (optional in the model bundle)
* Saves: model bundle (`.joblib`), scaler, feature list
* Generates training `classification_report.png` automatically

### ✔ Streamlit Dashboard (`scripts/dashboard.py`)

* Processes **very large CSVs** by reading in chunks (RAM safe)
* Auto-detects label and timestamp columns
* Aligns incoming data to training features (fills missing cols with zeros)
* Generates:

  * Predictions (labels)
  * IsolationForest anomaly scores
  * **Risk level** for each row (`LOW`, `MEDIUM`, `HIGH`)
* Visual analytics:

  * Confusion matrix (readable labels & boxed counts)
  * Classification report
  * ROC curves (One-vs-Rest) — when probabilities & true labels exist
  * Precision/Recall/F1 bar plots
  * Anomaly score histogram
  * Attack timeline (per minute) if timestamps available
* Exports:

  * CSV: `cicids_chunked_predictions_with_risk.csv` (prediction, iso_score, risk_level, true_label, timestamp)
  * Excel: multi-sheet workbook (`Predictions`, `Pivot`, `Risk Chart`, `Confusion Matrix`)
  * Multi-page PDF report (summary, risk chart, confusion matrix, classification report, ROC, PR, anomaly histogram, timeline) — optional and configurable

### ✔ Risk Level Scoring

Risk is assigned using transparent rules combining:

* predicted label keywords (e.g. `attack`, `dos`, `infiltration`, `sql`, etc.)
* IsolationForest anomaly score thresholds (tunable)
  Risk is included across UI, CSV, Excel, and PDF exports.

---

## 🧰 Tech Stack

|         Purpose |                           Technology |
| --------------: | -----------------------------------: |
|     ML Training |                 Python, scikit-learn |
| Feature Scaling |        scikit-learn `StandardScaler` |
|    Dashboard UI |                            Streamlit |
|        Plotting |                           Matplotlib |
|   Data Handling |                        Pandas, NumPy |
|   Excel reports |                             openpyxl |
|     PDF reports |                Matplotlib `PdfPages` |
|     Large files |           Chunked CSV reads (pandas) |
|     Model files | joblib (store via Git LFS if needed) |

---

## ⚙️ Quick Setup

1. Clone the repository

```bash
git clone https://github.com/yourusername/RealTime_Alert_Analysis.git
cd RealTime_Alert_Analysis
```

2. Create & activate virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. (Optional) Install Git LFS if you plan to push model files

```bash
git lfs install
git lfs pull
```

---

## 🧠 Train the Model

Run the RAM-efficient trainer:

```bash
python scripts/train.py
```

Outputs:

* `model/cicids2018_rf_model_A.joblib`
* `model/cicids2018_scaler_A.joblib`
* `model/cicids2018_features_A.joblib`
* `report/training_report/classification_report.png`

Notes:

* Training is performed on sampled balanced pools — for full production models train on the whole dataset or tune sampling strategy.
* Large `.joblib` files should be stored with Git LFS.

---

## ▶ Run the Streamlit Dashboard

Start the dashboard:

```bash
streamlit run scripts/dashboard.py
```

Open the local URL (typically `http://localhost:8501`) and use the sidebar to point to model/scaler files and the main UI to upload or provide a path to a large CSV.

### Typical workflow in the UI:

1. Provide model & scaler paths (defaults point to `model/`).
2. Upload a CSV or enter the local CSV path (preferred for huge files).
3. Configure `chunk size` (smaller if RAM-constrained).
4. Click **Start Processing** — progress bar updates while chunks are processed.
5. View the summary, charts, and tables.
6. Export:

   * Click **Download CSV** to get `cicids_chunked_predictions_with_risk.csv`.
   * Click **Generate Excel file** to download a workbook with `Predictions`, `Pivot`, `Risk Chart`, and `Confusion Matrix`.
   * (If enabled in your copy) click **Generate PDF** to download the full multi-page analytic report.

---

## 📈 Output Files

* `report/training_report/classification_report.png` — training evaluation (PNG)
* `cicids_chunked_predictions_with_risk.csv` — per-row predictions, iso_score, risk_level
* `cicids_report.xlsx` — Excel workbook (Predictions, Pivot, Risk Chart, Confusion Matrix)
* `cicids_risk_report.pdf` — multi-page PDF analysis (if PDF export enabled in dashboard)

---

## 📝 Implementation Notes & Best Practices

* **Chunked processing**: always supply a local path for very large CSVs; browser uploads may be memory-limited.
* **Feature alignment**: dashboard aligns incoming columns to `cicids2018_features_A.joblib`. Missing features are filled with zeros.
* **Header rows inside data**: the dashboard removes accidental header rows found inside chunks (e.g., repeated CSV header row) so exported files remain clean.
* **Session state**: Streamlit session state is used so exports (CSV/Excel/PDF) do not re-run the heavy pipeline and the UI does not clear after clicking a download button.
* **Confusion matrix readability**: the app draws numeric counts with white boxed backgrounds and a soft colormap to avoid overlap/low contrast.
* **Risk rules**: configurable thresholds are defined in the dashboard; you can fine-tune thresholds and keyword lists in `scripts/dashboard.py`.

---

## 🔧 Troubleshooting

* If the Excel file shows columns vertically or weird table layout: ensure all result lists (`prediction`, `iso_score`, `risk_level`, `true_label`, `timestamp`) are the same length — the dashboard pads shorter lists automatically but if you modified code re-check padding logic.
* If confusion matrix contains a `Label` row: that indicates a header row existed inside your CSV data. The dashboard now attempts to drop such rows, but if CSV is very messy pre-clean the CSV or open it in a text editor to remove stray header lines.
* If model/scaler load fails: verify paths and that the `.joblib` files were produced by `scripts/train.py` and that your environment uses compatible scikit-learn versions.

---

## 📦 Example `requirements.txt` (minimal)

```
numpy
pandas
scikit-learn
matplotlib
streamlit
joblib
openpyxl
tqdm
```

(Use `pip freeze > requirements.txt` to pin exact versions in your environment.)


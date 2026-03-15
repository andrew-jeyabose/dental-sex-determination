# 🦷 Dental Sex Determination Tool

A machine learning web application that predicts biological sex from dental tooth measurements (Mesio-Distal and Bucco-Lingual dimensions) using a trained **LightGBM classifier**.

---

## 🔬 About

This tool was developed as part of a research study on forensic dental sex determination. It uses tooth measurements from all four quadrants (56 features total) following the FDI tooth notation system.

| Feature | Detail |
|---|---|
| **Model** | LightGBM Classifier (max_depth=2) |
| **Input Features** | 56 (MD + BL measurements for teeth 11–17, 21–27, 31–37, 41–47) |
| **Output** | Male / Female with confidence score |
| **Accuracy** | 81.2% |
| **F1-Score** | 0.811 |

---

## 🚀 Run the App

### Option 1 — Run Locally

**Requirements:** Python 3.8 or higher

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/dental-sex-determination.git
cd dental-sex-determination

# 2. Create and activate virtual environment
# macOS / Linux:
python3 -m venv venv
source venv/bin/activate

# Windows:
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open in your browser at: **http://localhost:8501**

---

### Option 2 — Streamlit Cloud (Online)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://YOUR_APP_URL.streamlit.app)

---

## 📋 How to Use

### Upload CSV (Recommended)
1. Click the **📂 Upload CSV** tab
2. Upload your CSV file with 56 tooth measurements
3. Values populate automatically
4. Click **🔍 Determine Sex from CSV**
5. View result and optionally download as CSV

### Manual Entry
1. Click the **✏️ Manual Entry** tab
2. Enter MD and BL values for each tooth
3. Click **🔍 Determine Sex (Manual)**

---

## 📐 CSV File Format

Your CSV must contain exactly **56 columns** in this order:

```
11MD, 11BL, 12MD, 12BL, 13MD, 13BL, 14MD, 14BL, 15MD, 15BL, 16MD, 16BL, 17MD, 17BL,
21MD, 21BL, 22MD, 22BL, 23MD, 23BL, 24MD, 24BL, 25MD, 25BL, 26MD, 26BL, 27MD, 27BL,
31MD, 31BL, 32MD, 32BL, 33MD, 33BL, 34MD, 34BL, 35MD, 35BL, 36MD, 36BL, 37MD, 37BL,
41MD, 41BL, 42MD, 42BL, 43MD, 43BL, 44MD, 44BL, 45MD, 45BL, 46MD, 46BL, 47MD, 47BL
```

> **MD** = Mesio-Distal | **BL** = Bucco-Lingual | Numbers = FDI tooth notation

---

## 📁 Repository Structure

```
dental-sex-determination/
├── app.py                  # Main Streamlit application
├── lgbm_tooth_model.pkl    # Trained LightGBM model
├── feature_names.json      # Feature names list
├── label_encoder.pkl       # Label encoder (Male/Female)
├── sample_template.csv     # Sample CSV template
├── requirements.txt        # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

---

## 🧪 Model Performance

| Metric | Score |
|---|---|
| Accuracy | 0.812 |
| Precision | 0.816 |
| Recall | 0.824 |
| F1-Score | 0.811 |
| ROC-AUC | — |

---

## 🛠️ Built With

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)

---

## ⚠️ Disclaimer

This tool is intended for **forensic and clinical assistance only**.
Results should be interpreted by a qualified dental or forensic professional.
This tool does not replace expert clinical or forensic judgment.

---

## 📧 Contact

For questions or collaborations, please contact: **[your.email@institution.edu]**

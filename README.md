# HHS Unaccompanied Children Program — Data Analysis & Visualization Dashboard

**MCA Major Project** | Python · Streamlit · Plotly · Scikit-learn

---

## Project Overview

This dashboard analyses daily records published by the **U.S. Department of Health and Human Services (HHS) Office of Refugee Resettlement (ORR)**, tracking unaccompanied migrant children through the U.S. immigration custody pipeline — from initial apprehension by Customs and Border Protection (CBP) through placement in HHS-funded shelters until discharge to a sponsor or family.

The dataset spans **January 2023 – December 2025** and contains **719 daily observations** across five key metrics.

---

## Project Structure

```
child_care_dashboard/
├── main.py                    # Main Streamlit application (all 6 pages)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── data/
│   └── hhs_children_data.csv  # Source dataset (719 rows × 6 columns)
└── modules/
    ├── data_loader.py         # Data I/O, cleaning, aggregation utilities
    ├── visualizations.py      # All Plotly chart functions
    └── ml_models.py           # Regression & forecasting models
```

---

## Dataset Columns

| Column | Description |
|---|---|
| `Date` | Calendar date of the observation |
| `Children_Apprehended` | New apprehensions by CBP that day |
| `Children_in_CBP` | Children in Border Patrol holding that day |
| `Children_Transferred` | Children moved from CBP → HHS that day |
| `Children_in_HHS` | Running census in HHS/ORR shelters |
| `Children_Discharged` | Children released to a sponsor/family that day |

---

## Dashboard Pages

### 🏠 Home
High-level project overview with five KPI cards (total apprehended, peak HHS census, average daily transfers, total discharged, latest HHS count) and a 30-day moving average trend of the HHS census.

### 📊 Data Overview
Interactive data table with column-level filtering, downloadable CSV export, descriptive statistics (count, mean, std, quartiles), null value report, and dataset metadata (shape, date range, data types).

### 🧹 Data Cleaning
Step-by-step cleaning report documenting seven pipeline stages: column standardisation, date parsing, chronological sorting, numeric coercion, median imputation, duplicate removal, and feature engineering.  Includes a before/after comparison table and a null-value heatmap.

### 📈 Visualizations
Nine interactive chart types accessible via a dropdown selector:
- Trend lines with configurable multi-column selection
- Trend lines with moving average overlay (adjustable window 3–90 days)
- Monthly grouped bar charts
- Year-over-year bar charts
- Pie/donut composition charts
- Pearson correlation heatmap
- Year-wise box plots
- Monthly heatmap calendar
- Scatter plots between any two variables
- Stacked area charts
- Distribution histograms

### 💡 Insights
Six data-driven insight panels covering peak/trough HHS custody dates, correlation analysis, transfer efficiency, seasonal migration patterns, and year-over-year trend narrative.  Accompanied by an annual comparison table and two supporting charts.

### 🤖 Prediction
Three machine learning forecast models:
- **Linear Regression** — trend extrapolation with 95% confidence interval
- **Polynomial Regression** — degree-2 and degree-3 variants for non-linear trends
- **Rolling Average** — 30-day rolling mean projection as an interpretable baseline

Each model displays MAE, RMSE, and R² metrics.  A separate feature importance panel identifies which daily metrics most strongly predict the chosen target variable using standardised regression coefficients.

---

## How to Run Locally

**Step 1 — Clone or download the project folder.**

**Step 2 — Create and activate a virtual environment (recommended):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

**Step 3 — Install dependencies:**
```bash
pip install -r requirements.txt
```

**Step 4 — Launch the dashboard:**
```bash
streamlit run main.py
```

The app will open at `http://localhost:8501` in your browser.

---

## Deploying to Streamlit Cloud (Free)

1. Push the entire `child_care_dashboard/` folder to a public GitHub repository.
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, and set **Main file path** to `main.py`.
4. Click **Deploy** — your app will be live at a public URL within minutes.

No changes to the code are needed for deployment; Streamlit Cloud reads `requirements.txt` automatically.

---

## Technical Stack

| Component | Library | Version |
|---|---|---|
| Web framework | Streamlit | ≥ 1.32 |
| Data manipulation | Pandas, NumPy | ≥ 2.0, ≥ 1.25 |
| Interactive charts | Plotly | ≥ 5.18 |
| Machine learning | Scikit-learn | ≥ 1.3 |
| Excel support | openpyxl | ≥ 3.1 |

---

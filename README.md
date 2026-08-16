# Telecom Churn Intelligence (TCI) — V3
IBM Q2D UG Level 2 — Case Study #16

**Predict churn. Understand why. Prioritize retention.**

## What's New in V3
- **Fixed the sidebar/URL encoding bug** (mojibake like `=fôè Executive Overview`) by switching from
  filename-embedded emoji to Streamlit's `st.navigation`/`st.Page` API, where titles/icons are plain
  Python strings — nothing left for the filesystem or URL router to mis-encode.
- **Professional light theme by default**, with the native Streamlit Settings menu (top-right) still
  available for Light/Dark/System switching. Every custom card now uses Streamlit's own CSS variables
  instead of hardcoded colors, so it stays readable under any theme the user picks.
- **Custom favicon** (`assets/favicon.png`) — signal-bars icon in the brand blue.
- **Grouped sidebar navigation**: Overview / Customer Analytics / Model / Documentation.
- **New pages**: Customer Segments (dedicated persona analysis), Model Performance (full metrics + ROC
  curve), External Validation (honest Cell2Cell analysis), About.
- **External dataset analysis**: a second, independent Decision Tree trained only on the Cell2Cell
  telecom dataset (`train_cell2cell_model.py`) — NOT merged with IBM Telco, and NOT presented as a
  validation of the IBM model, since the two datasets have incompatible schemas. See the External
  Validation page for the full honest explanation.

## Setup
```bash
pip install -r requirements.txt
```
Primary IBM model artifacts are already trained and included in `models/`. Only re-run if you need to
regenerate them:
```bash
python train_model.py
```
The Cell2Cell external model is also pre-trained (`models_cell2cell/`). Only re-run if needed:
```bash
python train_cell2cell_model.py
```

## Run
```bash
streamlit run app.py
```
Opens at `http://localhost:8501`.

## Project Structure
```
telecom-churn-project/
├── .streamlit/config.toml       (light theme default + performance settings)
├── assets/favicon.png            (custom app icon)
├── data/
│   ├── Telco_customer_churn.xlsx        (primary dataset)
│   └── cell2cell/                        (external dataset — train + unlabeled holdout)
├── models/                       (primary IBM Telco model — unchanged since V1)
├── models_cell2cell/              (separate, independent Cell2Cell model)
├── outputs/
├── app.py                        (router — st.navigation/st.Page)
├── views/
│   ├── executive_overview.py     (landing page + portfolio KPI dashboard)
│   ├── customer_risk.py
│   ├── what_if.py
│   ├── customer_segments.py
│   ├── churn_insights.py
│   ├── model_performance.py      (NEW — metrics + ROC curve)
│   ├── external_validation.py    (NEW — Cell2Cell analysis)
│   ├── methodology.py
│   └── about.py                  (NEW)
├── theme.py                      (design system — CSS-variable based)
├── utils.py                      (shared model loading + inference + portfolio scoring)
├── train_model.py                (IBM Telco training — unchanged, no retraining occurred)
├── train_cell2cell_model.py      (NEW — separate Cell2Cell training script)
└── requirements.txt / README.md
```

## Important Notes for Your Report/Viva
- The Cell2Cell "External Validation" page does **not** claim to validate the IBM Telco model —
  it explains why direct transfer isn't valid (incompatible schemas) and presents a separate,
  independently-trained model's honest (weaker) results instead.
- `cell2cellholdout.csv` has an empty `Churn` column for all 20,000 rows and is **not** used for any
  metric calculation anywhere in this project.
- Student ID and personal contact links are intentionally not shown in the live app (see the note in
  `views/about.py`) — keep those in your report/README/viva submission instead.

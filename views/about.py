"""
views/about.py
Public-facing project information. Sensitive/personal identifiers (student ID,
personal LinkedIn/GitHub) are intentionally NOT displayed here — see the note
at the bottom of this file for what belongs in the report/README instead.
"""

import streamlit as st
from utils import load_artifacts
from theme import inject_css, section_header, app_card_start, app_card_end, disclaimer

inject_css()

artifacts = load_artifacts()
metadata = artifacts["metadata"]

st.title("About")

app_card_start()
st.markdown(
    """
    <div class="hero-title" style="font-size:1.6rem;">Telecom Churn Intelligence (TCI)</div>
    <div class="hero-tagline">Predict churn. Understand why. Prioritize retention.</div>
    <br>
    <b>Purpose:</b> An interpretable machine-learning decision-support prototype for telecom
    customer retention analysis, built for IBM Q2D UG Level 2 — Case Study #16:
    Telecom Churn Driver Discovery &amp; Persona Profiler.
    """,
    unsafe_allow_html=True,
)
app_card_end()

section_header("Technology Stack")
app_card_start()
st.markdown(
    """
    - **Language:** Python
    - **Application framework:** Streamlit
    - **Data processing:** Pandas, NumPy
    - **Modeling:** scikit-learn (Decision Tree, K-Means)
    - **Charting:** Streamlit native charts
    """
)
app_card_end()

section_header("Models")
app_card_start()
st.markdown(
    f"""
    - **Decision Tree** (depth {metadata['selected_tree_depth']}) — interpretable churn prediction
    - **K-Means Clustering** (k={metadata['selected_k']}) — behavioral persona segmentation
    """
)
app_card_end()

section_header("Datasets")
app_card_start()
st.markdown(
    """
    - **IBM Telco Customer Churn** — primary dataset, 7,043 customers, used to train and evaluate the primary model
    - **Cell2Cell Telecom Churn** — independent external dataset, used for a separate methodological analysis (see External Validation)
    """
)
app_card_end()

disclaimer(
    "This application provides model-based analytical estimates for educational/prototype "
    "purposes. Predictions are not guarantees of customer behavior."
)

# NOTE (not rendered in-app): student ID, personal name, personal GitHub/LinkedIn links belong
# in the project report/README/viva submission, not in a public-facing demo page — keeping
# personal identifiers out of the live application is a reasonable default for a prototype that
# may be screen-shared or deployed with a public URL during judging.

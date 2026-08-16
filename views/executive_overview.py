"""
views/executive_overview.py
This is the application's landing page (there is no separate Home page in
V3 — Executive Overview absorbs the hero/intro content plus the portfolio
KPI dashboard, per the V3 architecture).
"""

import plotly.express as px
import streamlit as st
from utils import load_artifacts, compute_portfolio_scores, compute_kpis, format_currency, CURRENCY_OPTIONS
from theme import inject_css, kpi_card, section_header, capability_card, disclaimer, RISK_COLORS

inject_css()

artifacts = load_artifacts()
metadata = artifacts["metadata"]

# --- Hero ---
st.markdown('<div class="hero-title">Telecom Churn Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-tagline">Predict churn. Understand why. Prioritize retention.</div>', unsafe_allow_html=True)
st.caption("IBM Q2D UG Level 2 — Case Study #16: Telecom Churn Driver Discovery & Persona Profiler")

st.markdown("")

# --- Portfolio KPIs ---
with st.spinner("Scoring the customer portfolio..."):
    scored_df = compute_portfolio_scores(
        artifacts["tree_model"], artifacts["kmeans_model"], artifacts["tree_encoders"],
        artifacts["cluster_encoders"], artifacts["cluster_scaler"],
        metadata["tree_feature_columns"], metadata["cluster_feature_columns"], metadata["personas"],
    )
kpis = compute_kpis(scored_df)
m = metadata["tree_metrics"]

currency = st.selectbox("Display currency", list(CURRENCY_OPTIONS.keys()), index=0,
                         help="Display-layer conversion only — the underlying model and dataset remain in USD.")

section_header("Portfolio KPIs")
c1, c2, c3 = st.columns(3)
with c1:
    kpi_card("👥", f"{kpis['total_customers']:,}", "Total Customers")
with c2:
    kpi_card("📉", f"{kpis['historical_churn_rate']*100:.1f}%", "Historical Churn Rate")
with c3:
    kpi_card("🔴", f"{kpis['high_risk']:,}", "High-Risk Customers")

c4, c5, c6 = st.columns(3)
with c4:
    kpi_card("🎯", f"{m['accuracy']*100:.1f}%", "Model Accuracy")
with c5:
    kpi_card("📐", f"{m['f1_score']:.2f}", "F1 Score")
with c6:
    kpi_card("💰", format_currency(kpis['estimated_monthly_revenue_at_risk'], currency),
              "Estimated Revenue at Risk")

st.markdown("")

# --- Risk distribution donut ---
section_header("Predicted Risk Distribution", "Model-predicted risk tier across all customers.")
risk_counts = scored_df["Predicted Risk"].value_counts().reindex(["Low", "Medium", "High"]).fillna(0).reset_index()
risk_counts.columns = ["Risk", "Customers"]
fig = px.pie(
    risk_counts, names="Risk", values="Customers", hole=0.55,
    color="Risk", color_discrete_map=RISK_COLORS,
)
fig.update_layout(
    margin=dict(t=10, b=10, l=10, r=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=-0.15),
    height=320,
    font_color="#6B7280",  # medium gray — legible on both light and dark Streamlit themes
)
fig.update_traces(textinfo="percent+label")
st.plotly_chart(fig, use_container_width=True)

st.markdown("")

# --- What can you do ---
section_header("What Can You Do?")
cap1, cap2 = st.columns(2)
with cap1:
    capability_card("Customer Risk",
                     "Identify a specific customer's predicted churn risk and understand the "
                     "factors behind it.")
    capability_card("What-If Simulator",
                     "Change a customer attribute — like contract type — and compare the model's "
                     "prediction before and after.")
    capability_card("Customer Segments",
                     "Understand the behavioral personas the model discovered across the customer base.")
with cap2:
    capability_card("Churn Insights",
                     "Explore the strongest churn drivers and business-level patterns.")
    capability_card("Model Performance",
                     "Inspect accuracy, precision, recall, F1, ROC-AUC, and the confusion matrix.")
    capability_card("External Validation",
                     "See how the same modeling approach performs on an independent telecom dataset.")

st.markdown("")
disclaimer(
    "\"Predicted Churn Risk\" is a model-based estimate from customer profile patterns, not a "
    "guarantee. \"Historical Churn\" reflects customers who had already churned in this dataset. "
    "\"Estimated Revenue at Risk\" is a probability-weighted estimate — see Methodology for the calculation."
)

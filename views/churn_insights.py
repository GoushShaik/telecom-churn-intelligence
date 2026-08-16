"""
views/churn_insights.py
Churn drivers and business-pattern analysis. Persona-specific content moved
to views/customer_segments.py; model metrics moved to views/model_performance.py.
"""

import pandas as pd
import streamlit as st
from utils import load_artifacts, compute_portfolio_scores
from theme import inject_css, section_header, app_card_start, app_card_end

inject_css()

artifacts = load_artifacts()
metadata = artifacts["metadata"]

st.title("Churn Insights")
st.caption("The strongest churn drivers and how churn varies across contract, tenure, and payment method.")

# --- Feature importance (horizontal bar, as requested — easier to read many labels) ---
section_header("Top Churn Drivers (Feature Importance)")
fi = metadata["feature_importance"]
fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
fi_df_nonzero = fi_df[fi_df["Importance"] > 0]
st.bar_chart(fi_df_nonzero.set_index("Feature"), horizontal=True)

top_feature = fi_df_nonzero.sort_values("Importance", ascending=False).iloc[0]["Feature"]
app_card_start()
st.markdown(f"**What this means for the business:** `{top_feature}` is by far the strongest churn "
            f"driver the model found. This is a correlational pattern from historical data, not a "
            f"proven causal effect — but it is a reasonable, defensible signal for prioritizing "
            f"retention effort.")
app_card_end()

st.markdown("")

# --- Business breakdowns (reuses the cached portfolio scoring — fast after first computation) ---
section_header("Churn by Contract, Payment Method, and Tenure",
               "Computed from the actual 7,043 customers in the validated dataset.")
with st.spinner("Loading portfolio breakdowns..."):
    scored_df = compute_portfolio_scores(
        artifacts["tree_model"], artifacts["kmeans_model"], artifacts["tree_encoders"],
        artifacts["cluster_encoders"], artifacts["cluster_scaler"],
        metadata["tree_feature_columns"], metadata["cluster_feature_columns"], metadata["personas"],
    )

bc1, bc2 = st.columns(2)
with bc1:
    st.markdown("**By Contract Type**")
    st.bar_chart(scored_df.groupby("Contract")["Churn Value"].mean().sort_values(ascending=False) * 100)
with bc2:
    st.markdown("**By Payment Method**")
    st.bar_chart(scored_df.groupby("Payment Method")["Churn Value"].mean().sort_values(ascending=False) * 100)

st.markdown("**By Tenure Group** (ordered — earlier tenure on the left)")
tenure_order = ["0-12 months", "12-24 months", "24-48 months", "48-60 months", "60+ months"]
churn_by_tenure = scored_df.groupby("Tenure Bucket")["Churn Value"].mean().reindex(tenure_order) * 100
st.line_chart(churn_by_tenure)

app_card_start()
st.markdown("**What this means for the business:** month-to-month customers and newer customers "
            "(0-12 months) both show substantially higher historical churn than longer-contract or "
            "longer-tenured customers — the two strongest, most consistent patterns in this dataset.")
app_card_end()

st.markdown("")

# --- Decision tree overview ---
section_header("Decision Tree Overview")
st.markdown(f"**Selected tree depth:** {metadata['selected_tree_depth']} "
            f"(chosen for the best balance of readability and predictive performance)")
with st.expander("View top decision rules extracted from the tree"):
    for rule in metadata["readable_rules"]:
        st.markdown(f"- {rule}")

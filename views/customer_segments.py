"""
views/customer_segments.py
Dedicated behavioral persona (K-Means) analysis page.
"""

import pandas as pd
import streamlit as st
from utils import load_artifacts
from theme import inject_css, section_header, app_card_start, app_card_end

inject_css()

artifacts = load_artifacts()
metadata = artifacts["metadata"]
personas = metadata["personas"]

st.title("Customer Segments")
st.caption("Behavioral personas discovered by K-Means clustering, based on tenure, spend, contract, "
           "and service usage patterns.")

section_header("Persona Comparison")
persona_rows = []
for cid, p in personas.items():
    persona_rows.append({
        "Persona": p["name"], "Size": p["size"], "Churn Rate": f"{p['churn_rate']*100:.1f}%",
        "Avg Tenure (months)": p["avg_tenure_months"], "Avg Monthly Charges ($)": p["avg_monthly_charges"],
        "Avg Services Used": p["avg_service_count"], "Most Common Contract": p["most_common_contract"],
    })
persona_df = pd.DataFrame(persona_rows).sort_values("Churn Rate", ascending=False)
st.dataframe(persona_df, use_container_width=True, hide_index=True)

p1, p2 = st.columns(2)
with p1:
    st.markdown("**Persona Sizes**")
    size_chart = pd.DataFrame({p["name"]: [p["size"]] for p in personas.values()}).T.rename(columns={0: "Customers"})
    st.bar_chart(size_chart)
with p2:
    st.markdown("**Churn Rate by Persona**")
    churn_chart = pd.DataFrame({p["name"]: [p["churn_rate"] * 100] for p in personas.values()}).T.rename(columns={0: "Churn Rate (%)"})
    st.bar_chart(churn_chart)

st.markdown("")
section_header("Persona Profiles & Business Interpretation")
for cid, p in personas.items():
    app_card_start()
    st.markdown(
        f"**{p['name']}** ({p['size']:,} customers, {p['churn_rate']*100:.1f}% historical churn rate)<br>"
        f"{p['description']}<br><br>"
        f"Avg tenure: {p['avg_tenure_months']} months &nbsp;|&nbsp; "
        f"Avg monthly charges: ${p['avg_monthly_charges']:.2f} &nbsp;|&nbsp; "
        f"Avg services used: {p['avg_service_count']} &nbsp;|&nbsp; "
        f"Most common contract: {p['most_common_contract']}",
        unsafe_allow_html=True,
    )
    app_card_end()

app_card_start()
st.markdown("**What this means for the business:** the highest-churn persona should be the retention "
            "team's top priority. Persona names and descriptions come from transparent threshold rules "
            "on each cluster's actual average tenure, spend, and churn rate — not a black-box process.")
app_card_end()

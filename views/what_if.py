"""
pages/3_🔄_What‑If_Simulator.py
Model-based what-if comparison: current customer profile vs. one modified
attribute, both scored with the EXISTING trained model. No retraining.
This is decision support, not a causal guarantee — labeled as such throughout.
"""

import streamlit as st
from utils import (
    load_artifacts, encode_customer_input, predict_churn,
    build_cluster_features, predict_persona
)
from theme import inject_css, section_header, app_card_start, app_card_end, risk_badge_html, disclaimer

inject_css()

artifacts = load_artifacts()
tree_model = artifacts["tree_model"]
kmeans_model = artifacts["kmeans_model"]
tree_encoders = artifacts["tree_encoders"]
cluster_encoders = artifacts["cluster_encoders"]
cluster_scaler = artifacts["cluster_scaler"]
metadata = artifacts["metadata"]
tree_feature_columns = metadata["tree_feature_columns"]
cluster_feature_columns = metadata["cluster_feature_columns"]

st.title("🔄 What-If Retention Simulator")
st.caption("Compare a customer's current predicted risk against a modified profile — "
           "using the same trained model, with one attribute changed.")

disclaimer(
    "This is a model-based what-if estimate, not a causal guarantee. Changing a customer's "
    "contract does not guarantee retention — it shows how the model's prediction shifts for "
    "customers with that different profile, based on historical patterns."
)

st.markdown("")

# Pre-fill from the last Customer Risk prediction if available, otherwise use sensible defaults
default_input = st.session_state.get("last_prediction_input", {
    "Gender": "Female", "Senior Citizen": "No", "Partner": "No", "Dependents": "No",
    "Tenure Months": 6, "Phone Service": "Yes", "Multiple Lines": "No",
    "Internet Service": "Fiber optic", "Online Security": "No", "Online Backup": "No",
    "Device Protection": "No", "Tech Support": "No", "Streaming TV": "No", "Streaming Movies": "No",
    "Contract": "Month-to-month", "Paperless Billing": "Yes", "Payment Method": "Electronic check",
    "Monthly Charges": 80.0, "Total Charges": 480.0,
})

if "last_prediction_input" in st.session_state:
    st.success("Loaded the customer profile from your last Customer Risk prediction.")
else:
    st.info("No prior prediction found — using a sample high-risk profile. "
            "Visit Customer Risk first to simulate a specific customer.")

section_header("Baseline Customer Profile")
app_card_start()
b1, b2, b3, b4 = st.columns(4)
b1.markdown(f"**Contract**<br>{default_input['Contract']}", unsafe_allow_html=True)
b2.markdown(f"**Tenure**<br>{default_input['Tenure Months']} months", unsafe_allow_html=True)
b3.markdown(f"**Monthly Charges**<br>${default_input['Monthly Charges']:.2f}", unsafe_allow_html=True)
b4.markdown(f"**Internet Service**<br>{default_input['Internet Service']}", unsafe_allow_html=True)
app_card_end()

section_header("Choose a What-If Change")
change_type = st.selectbox(
    "What should we change?",
    ["Upgrade Contract", "Add Online Security", "Add Tech Support", "Switch to Automatic Payment"],
)

modified_input = dict(default_input)
change_description = ""

if change_type == "Upgrade Contract":
    new_contract = st.selectbox("New contract", ["One year", "Two year"])
    modified_input["Contract"] = new_contract
    change_description = f"Contract changed from **{default_input['Contract']}** to **{new_contract}**"

elif change_type == "Add Online Security":
    modified_input["Online Security"] = "Yes"
    change_description = "Online Security changed to **Yes**"

elif change_type == "Add Tech Support":
    modified_input["Tech Support"] = "Yes"
    change_description = "Tech Support changed to **Yes**"

elif change_type == "Switch to Automatic Payment":
    modified_input["Payment Method"] = "Bank transfer (automatic)"
    change_description = f"Payment Method changed from **{default_input['Payment Method']}** to **Bank transfer (automatic)**"

if st.button("Run What-If Comparison", use_container_width=True):

    def score(raw_input):
        fdf = encode_customer_input(raw_input, tree_encoders, tree_feature_columns)
        prob, risk = predict_churn(tree_model, fdf)
        cdf = build_cluster_features(raw_input, cluster_encoders, cluster_feature_columns)
        _, persona = predict_persona(kmeans_model, cluster_scaler, cdf, metadata)
        return prob, risk, persona

    current_prob, current_risk, current_persona = score(default_input)
    new_prob, new_risk, new_persona = score(modified_input)
    delta = (current_prob - new_prob) * 100

    st.divider()
    st.markdown("## Comparison Result")
    st.markdown(change_description)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Current Profile")
        st.markdown(f"**{current_prob*100:.1f}%** churn probability")
        st.progress(min(current_prob, 1.0))
        st.markdown(risk_badge_html(current_risk), unsafe_allow_html=True)
        st.caption(f"Persona: {current_persona['name']}")

    with col2:
        st.markdown("### What-If Profile")
        st.markdown(f"**{new_prob*100:.1f}%** churn probability")
        st.progress(min(new_prob, 1.0))
        st.markdown(risk_badge_html(new_risk), unsafe_allow_html=True)
        st.caption(f"Persona: {new_persona['name']}")

    with col3:
        st.markdown("### Change")
        if delta > 0:
            st.markdown(f"### 🟢 -{delta:.1f} pts")
            st.caption("Predicted risk decreased under the what-if scenario.")
        elif delta < 0:
            st.markdown(f"### 🔴 +{abs(delta):.1f} pts")
            st.caption("Predicted risk increased under the what-if scenario.")
        else:
            st.markdown("### No change")
            st.caption("This modification did not shift the model's prediction.")

    st.caption("This comparison uses the existing trained Decision Tree with no retraining. "
               "It reflects how customers with the modified profile have historically churned, "
               "not a guaranteed outcome for this specific customer.")

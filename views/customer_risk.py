"""
pages/2_🔮_Customer_Risk.py
Individual customer churn prediction, redesigned with a sectioned form and a
strong results dashboard (badges, driver cards, persona summary).
"""

import streamlit as st
from utils import (
    load_artifacts, encode_customer_input, predict_churn,
    build_cluster_features, predict_persona, explain_decision_path, recommend_action
)
from theme import inject_css, section_header, app_card_start, app_card_end, risk_badge_html

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

st.title("🔮 Customer Risk Predictor")
st.caption("Enter a hypothetical customer's details to see their churn risk, persona, and the reasoning behind the prediction.")

with st.form("customer_form"):
    st.markdown("#### A. Customer Profile")
    d1, d2, d3, d4 = st.columns(4)
    gender = d1.selectbox("Gender", tree_encoders["Gender"].classes_)
    senior = d2.selectbox("Senior Citizen", tree_encoders["Senior Citizen"].classes_)
    partner = d3.selectbox("Partner", tree_encoders["Partner"].classes_)
    dependents = d4.selectbox("Dependents", tree_encoders["Dependents"].classes_)

    st.markdown("#### B. Contract & Tenure")
    a1, a2, a3, a4 = st.columns(4)
    tenure = a1.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    contract = a2.selectbox("Contract", tree_encoders["Contract"].classes_)
    paperless = a3.selectbox("Paperless Billing", tree_encoders["Paperless Billing"].classes_)
    payment_method = a4.selectbox("Payment Method", tree_encoders["Payment Method"].classes_)

    st.markdown("#### C. Services")
    s1, s2, s3 = st.columns(3)
    phone_service = s1.selectbox("Phone Service", tree_encoders["Phone Service"].classes_)

    if phone_service == "No":
        multiple_lines = "No phone service"
        s2.selectbox("Multiple Lines", ["No phone service"], disabled=True)
    else:
        multiple_lines = s2.selectbox("Multiple Lines", ["No", "Yes"])

    internet_service = s3.selectbox("Internet Service", tree_encoders["Internet Service"].classes_)

    internet_dependent_cols = [
        "Online Security", "Online Backup", "Device Protection",
        "Tech Support", "Streaming TV", "Streaming Movies"
    ]
    service_values = {}
    si_cols = st.columns(3)
    for i, col_name in enumerate(internet_dependent_cols):
        target_col = si_cols[i % 3]
        if internet_service == "No":
            service_values[col_name] = "No internet service"
            target_col.selectbox(col_name, ["No internet service"], key=f"{col_name}_disabled", disabled=True)
        else:
            service_values[col_name] = target_col.selectbox(col_name, ["No", "Yes"], key=col_name)

    st.markdown("#### D. Billing")
    b1, b2 = st.columns(2)
    monthly_charges = b1.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.5)
    total_charges = b2.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0,
                                     value=float(monthly_charges * tenure), step=1.0)

    submitted = st.form_submit_button("Predict Churn Risk", use_container_width=True)

if submitted:
    raw_input = {
        "Gender": gender, "Senior Citizen": senior, "Partner": partner, "Dependents": dependents,
        "Tenure Months": tenure, "Phone Service": phone_service, "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": service_values["Online Security"],
        "Online Backup": service_values["Online Backup"],
        "Device Protection": service_values["Device Protection"],
        "Tech Support": service_values["Tech Support"],
        "Streaming TV": service_values["Streaming TV"],
        "Streaming Movies": service_values["Streaming Movies"],
        "Contract": contract, "Paperless Billing": paperless, "Payment Method": payment_method,
        "Monthly Charges": monthly_charges, "Total Charges": total_charges,
    }
    st.session_state["last_prediction_input"] = raw_input  # available to the What-If page

    feature_df = encode_customer_input(raw_input, tree_encoders, tree_feature_columns)
    churn_probability, risk_level = predict_churn(tree_model, feature_df)

    cluster_df = build_cluster_features(raw_input, cluster_encoders, cluster_feature_columns)
    cluster_id, persona = predict_persona(kmeans_model, cluster_scaler, cluster_df, metadata)

    path_steps, leaf_prob = explain_decision_path(tree_model, feature_df, tree_feature_columns, tree_encoders)
    recommendations = recommend_action(raw_input, metadata["feature_importance"])

    st.divider()
    st.markdown("## Results")

    r1, r2, r3 = st.columns([1, 1, 1.4])
    with r1:
        st.markdown(f"### {churn_probability*100:.1f}%")
        st.progress(min(churn_probability, 1.0))
        st.caption("Churn Probability")
    with r2:
        st.markdown(risk_badge_html(risk_level), unsafe_allow_html=True)
        st.caption("Predicted Risk Level")
    with r3:
        st.markdown(f"**Persona:** {persona['name']}")
        st.caption(f"Historical churn rate for this persona: {persona['churn_rate']*100:.1f}% "
                   f"across {persona['size']:,} similar customers")

    app_card_start()
    st.markdown(f"**Persona description:** {persona['description']}")
    app_card_end()

    section_header("Why this prediction? (Decision Tree Path)")
    if path_steps:
        for i, step in enumerate(path_steps, 1):
            st.markdown(f'<div class="driver-row">{i}. {step}</div>', unsafe_allow_html=True)
        st.caption(f"This decision path leads to a training-data churn rate of {leaf_prob*100:.1f}% "
                   f"for customers matching this exact profile.")
    else:
        st.markdown("This customer reached a decision immediately at the root of the tree.")

    section_header("Retention Recommendation",
                   "Derived from feature importance and this customer's own values — not a model output.")
    for rec in recommendations:
        st.markdown(f"- {rec}")

    st.info("💡 Want to see how a specific change (e.g. switching to a 1-year contract) would affect "
            "this customer's risk? Try the **What-If Simulator** page — it reuses this exact prediction.")

    st.caption("Risk level bucketing (Low <30%, Medium 30-60%, High >60%) is a presentation choice "
               "for this app and is not a value stored in the trained model itself.")

"""
views/model_performance.py
Dedicated model evaluation page for the primary IBM Telco Decision Tree.
ROC-AUC/curve are computed via inference on the already-fitted model (see
utils.compute_ibm_roc_curve) — no retraining occurs anywhere on this page.
"""

import pandas as pd
import streamlit as st
from utils import load_artifacts, compute_ibm_roc_curve
from theme import inject_css, section_header, app_card_start, app_card_end

inject_css()

artifacts = load_artifacts()
metadata = artifacts["metadata"]
m = metadata["tree_metrics"]

st.title("Model Performance")
st.caption("Evaluation of the primary Decision Tree, trained on the IBM Telco Customer Churn dataset.")

section_header("Classification Metrics")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{m['accuracy']*100:.1f}%")
c2.metric("Precision", f"{m['precision']*100:.1f}%")
c3.metric("Recall", f"{m['recall']*100:.1f}%")
c4.metric("F1-Score", f"{m['f1_score']:.3f}")

st.markdown("")
section_header("Confusion Matrix")
cm = m["confusion_matrix"]
cm_df = pd.DataFrame(cm, index=["Actual: No Churn", "Actual: Churn"],
                      columns=["Predicted: No Churn", "Predicted: Churn"])
st.dataframe(cm_df, use_container_width=True)

st.markdown("")
section_header("ROC Curve", "Computed by inference on the already-trained model against its original held-out test split.")
with st.spinner("Computing ROC curve..."):
    fpr, tpr, auc = compute_ibm_roc_curve(
        artifacts["tree_model"], artifacts["tree_encoders"], metadata["tree_feature_columns"]
    )
roc_df = pd.DataFrame({"False Positive Rate": fpr, "True Positive Rate": tpr}).set_index("False Positive Rate")
st.line_chart(roc_df)
st.metric("ROC-AUC", f"{auc:.3f}")

app_card_start()
st.markdown(
    f"**Why recall matters here:** for a churn-prevention system, failing to flag an actual "
    f"at-risk customer (a false negative) is typically more costly than flagging a customer who "
    f"was not going to churn (a false positive) — a missed at-risk customer is a lost retention "
    f"opportunity, while a false alarm just costs a small amount of unnecessary outreach. This "
    f"model was trained with `class_weight='balanced'` specifically to prioritize recall "
    f"({m['recall']*100:.1f}%) over precision ({m['precision']*100:.1f}%). "
    f"An ROC-AUC of {auc:.3f} indicates the model separates churners from non-churners "
    f"meaningfully better than random guessing (0.5), though it is not a perfect classifier — "
    f"no claim of near-perfect accuracy is being made here."
)
app_card_end()

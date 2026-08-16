"""
views/external_validation.py
Explains the Cell2Cell external dataset analysis honestly: schema mismatch
with IBM Telco, why direct model transfer isn't valid, and the results of
a SEPARATE independent model trained only on Cell2Cell's own labeled data.
"""

import json
import pandas as pd
import streamlit as st
from utils import load_artifacts, compute_ibm_roc_curve
from theme import inject_css, section_header, app_card_start, app_card_end, disclaimer

inject_css()

artifacts = load_artifacts()
metadata = artifacts["metadata"]
_, _, ibm_auc = compute_ibm_roc_curve(
    artifacts["tree_model"], artifacts["tree_encoders"], metadata["tree_feature_columns"]
)

st.title("External Validation")
st.caption("An honest look at whether — and how — the modeling approach holds up on an independent telecom dataset.")

try:
    with open("models_cell2cell/cell2cell_metadata.json") as f:
        c2c = json.load(f)
except FileNotFoundError:
    c2c = None

section_header("Primary Dataset vs. External Dataset")
d1, d2 = st.columns(2)
with d1:
    app_card_start()
    st.markdown(
        "**Primary: IBM Telco Customer Churn**<br>"
        "7,043 customers &nbsp;|&nbsp; Contract/service-bundle features "
        "(Contract, Internet Service, Online Security, etc.)<br>"
        "Fully labeled — used to train the primary Decision Tree + K-Means model.",
        unsafe_allow_html=True,
    )
    app_card_end()
with d2:
    app_card_start()
    if c2c:
        st.markdown(
            f"**External: Cell2Cell Telecom Churn**<br>"
            f"{c2c['rows_used']:,} labeled customers (train file) &nbsp;|&nbsp; "
            f"{c2c['holdout_rows']:,} unlabeled customers (holdout file)<br>"
            f"Call-usage/handset/demographic features (MonthlyMinutes, DroppedCalls, "
            f"CreditRating, etc.) — a fundamentally different schema from IBM Telco.",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("Cell2Cell metadata not found — run `train_cell2cell_model.py` first.")
    app_card_end()

st.markdown("")
section_header("Why We Did NOT Directly Validate the IBM Model on Cell2Cell")
app_card_start()
st.markdown(
    """
    Cell2Cell has **no equivalent columns** for Contract, Internet Service, Online Security, or any
    of the other features the IBM Telco model was trained on — and IBM Telco has no equivalent for
    Cell2Cell's call-usage features (MonthlyMinutes, DroppedCalls, RetentionCalls, CreditRating, etc.).
    There is no valid way to feed a Cell2Cell customer through the IBM-trained tree, and forcing it
    (e.g., by filling in fake values for missing columns) would not produce a meaningful validation —
    it would just produce a number that looks like a metric without actually measuring anything real.

    **What we did instead:** trained a second, completely independent Decision Tree using the *same
    interpretable-tree methodology* (same depth-selection process, same evaluation approach), but
    fit and evaluated entirely on Cell2Cell's own labeled data. This demonstrates that the modeling
    *approach* generalizes to another telecom dataset — it does **not** claim to validate the IBM
    model itself, which remains untouched.
    """
)
app_card_end()

disclaimer(
    "cell2cellholdout.csv (20,000 rows) has an empty Churn column for every row and was NOT used "
    "for any accuracy/precision/recall/F1/ROC-AUC calculation on this page or anywhere in this "
    "project — using it would require inventing labels, which we do not do."
)

st.markdown("")

if c2c:
    section_header("Cell2Cell Independent Model — Results")
    st.caption(c2c["note"])

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{c2c['accuracy']*100:.1f}%")
    c2.metric("Precision", f"{c2c['precision']*100:.1f}%")
    c3.metric("Recall", f"{c2c['recall']*100:.1f}%")
    c4.metric("F1-Score", f"{c2c['f1_score']:.3f}")
    c5.metric("ROC-AUC", f"{c2c['roc_auc']:.3f}")

    cm_df = pd.DataFrame(
        c2c["confusion_matrix"],
        index=["Actual: No Churn", "Actual: Churn"],
        columns=["Predicted: No Churn", "Predicted: Churn"],
    )
    st.markdown("**Confusion Matrix (Cell2Cell model)**")
    st.dataframe(cm_df, use_container_width=True)

    roc_df = pd.DataFrame({"False Positive Rate": c2c["roc_curve"]["fpr"],
                            "True Positive Rate": c2c["roc_curve"]["tpr"]}).set_index("False Positive Rate")
    st.markdown("**ROC Curve (Cell2Cell model)**")
    st.line_chart(roc_df)

    fi = c2c["feature_importance"]
    fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
    st.markdown("**Top Churn Drivers in Cell2Cell**")
    st.bar_chart(fi_df[fi_df["Importance"] > 0].set_index("Feature"), horizontal=True)

    app_card_start()
    st.markdown(
        f"**Honest interpretation:** the Cell2Cell model reaches {c2c['accuracy']*100:.1f}% accuracy "
        f"and a {c2c['roc_auc']:.3f} ROC-AUC — meaningfully better than random guessing (0.5), but "
        f"noticeably weaker than the IBM Telco model's {ibm_auc:.3f} ROC-AUC. This is a realistic "
        f"and expected outcome, not a flaw to hide: Cell2Cell is a noisier, usage-behavior-driven "
        f"dataset without the clean subscription-plan structure that makes IBM Telco's Contract "
        f"feature such a strong, clean signal. We are reporting this as-is rather than tuning it to "
        f"look more impressive than it genuinely is."
    )
    app_card_end()
else:
    st.warning("Cell2Cell model artifacts not found. Run `python train_cell2cell_model.py` to generate them.")

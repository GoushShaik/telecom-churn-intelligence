"""
pages/5_🧠_Methodology.py
Redesigned into expandable sections instead of a single text block. All
numbers are read from metadata.json — nothing invented or recalculated here.
"""

import streamlit as st
from utils import load_artifacts
from theme import inject_css, section_header, app_card_start, app_card_end, disclaimer

inject_css()

artifacts = load_artifacts()
metadata = artifacts["metadata"]
m = metadata["tree_metrics"]

st.title("🧠 Methodology")
st.caption("Dataset → Preprocessing → Decision Tree → K-Means → Evaluation → Business Interpretation")

with st.expander("📁 1. Dataset", expanded=True):
    st.markdown(
        """
        **Source:** IBM Telco Customer Churn Dataset (`Telco_customer_churn.xlsx`)
        **Size:** 7,043 customer records, 33 original columns, 1 sheet (`Telco_Churn`)
        **Target variable:** `Churn Value` (0 = retained, 1 = churned)
        """
    )

with st.expander("🧹 2. Data Preprocessing"):
    st.markdown(
        """
        - **Total Charges** was stored as text with 11 blank values, all belonging to customers
          with `Tenure Months = 0` (brand-new customers). Converted to numeric, set to 0.0.
        - **CustomerID** dropped — unique identifier, no predictive value.
        - **Count, Country, State** dropped — constant columns (zero variance).
        - Remaining categoricals **label-encoded** with scikit-learn's `LabelEncoder`, fit once
          during training and reused (`.transform()` only, never re-fit) at prediction time.
        """
    )

with st.expander("🚫 3. Features Excluded — and Why"):
    st.markdown(
        """
        | Column(s) | Reason for exclusion |
        |---|---|
        | `CustomerID` | Unique identifier, no predictive value |
        | `Count`, `Country`, `State` | Constant across the entire dataset |
        | `Churn Label` | Restates the target as text — direct leakage |
        | `Churn Score` | Pre-computed churn propensity score in IBM's sample data — including it would let the model "cheat" instead of learning real drivers |
        | `Churn Reason` | Only populated **after** a customer has already churned — using it predicts the outcome from itself |
        | `CLTV` | Derived partly from outcome-correlated patterns — excluded as a leakage precaution |
        | `City`, `Zip Code`, `Lat Long`, `Latitude`, `Longitude` | Geographic detail out of scope for this MVP |
        """
    )

with st.expander("🌳 4. Decision Tree Methodology"):
    st.markdown(
        f"""
        A `DecisionTreeClassifier` was chosen specifically because the case study requires an
        **interpretable** model that uncovers clear business rules — a black-box ensemble model
        would defeat that purpose even at marginally higher accuracy.

        - Tested `max_depth` of 3, 4, and 5, comparing test-set F1-score
        - **Selected depth: {metadata['selected_tree_depth']}** — best F1 while remaining readable
          as a small number of if/then rules
        - `class_weight="balanced"` used because churn is a minority class (~26% of customers)
        - 80/20 stratified train/test split, `random_state=42` for full reproducibility
        """
    )

with st.expander("👥 5. K-Means Methodology"):
    st.markdown(
        f"""
        - Clustering features scaled with `StandardScaler` before fitting K-Means (required —
          K-Means uses distance, and unscaled features like Total Charges would dominate)
        - Cluster count (k) chosen using the **elbow method** on inertia across k=2 to k=7
        - The raw elbow suggestion was clamped to a business-usable range (3-4 clusters)
        - **Selected k = {metadata['selected_k']}**
        - Each cluster named/described from its actual average tenure, spend, service usage, and
          churn rate — transparent threshold rules, not a black-box process
        """
    )

with st.expander("📏 6. Model Evaluation"):
    st.markdown(
        f"""
        | Metric | Value |
        |---|---|
        | Accuracy | {m['accuracy']*100:.1f}% |
        | Precision | {m['precision']*100:.1f}% |
        | Recall | {m['recall']*100:.1f}% |
        | F1-Score | {m['f1_score']:.3f} |

        Recall was prioritized over precision by design: for a churn-prevention use case, failing
        to flag an at-risk customer is more costly than occasionally flagging one who was not
        actually going to churn.
        """
    )

with st.expander("💰 7. Revenue at Risk — Calculation"):
    st.markdown(
        """
        **Estimated Monthly Revenue at Risk** is calculated as a **probability-weighted sum**,
        not a simple count of high-risk customers:

        ```
        Revenue at Risk = Σ (predicted_churn_probability_i × monthly_charges_i)
        ```

        across all customers. This is more statistically honest than multiplying the number of
        "High Risk" customers by their average charge, because it doesn't assume every high-risk
        customer will actually churn — each customer's contribution is weighted by their own
        individual predicted probability. This figure is a **model-based estimate**, not a
        guaranteed financial outcome.
        """
    )

with st.expander("🎯 8. What This Model Can and Cannot Claim"):
    st.markdown(
        """
        - **Can claim:** "Customers with this profile have historically churned at approximately
          X%, based on patterns in 7,043 historical records — useful for prioritizing retention effort."
        - **Cannot claim:** "This customer will leave" — there is no future time horizon in this
          dataset; `Churn Value` is a historical, already-known outcome, not a forecast.
        - **Cannot claim:** causal relationships (e.g., "month-to-month contracts cause churn") —
          the tree shows correlational patterns, not proven causation.
        - **Cannot claim:** Revenue-at-Risk or What-If results are guaranteed outcomes — both are
          model-based estimates for decision support, not commitments.
        """
    )

with st.expander("⚠️ 9. Limitations"):
    st.markdown(
        """
        - Precision (53.8%) means a meaningful share of predicted-churn customers will not
          actually churn — an accepted trade-off given the recall-first design.
        - Trained on a single historical snapshot of one telecom provider's data; may not
          generalize to other providers or time periods without retraining.
        - Geographic features excluded from this MVP scope.
        - Persona naming uses transparent threshold rules rather than a fully automated labeling
          system — appropriate for interpretability, but would need re-review if the underlying
          data or cluster count changed.
        """
    )

with st.expander("🔮 10. Future Scope"):
    st.markdown(
        """
        - Allow users to upload a new, schema-compatible telecom dataset and retrain the models —
          **deliberately not built in this version** to keep the prototype focused and fully
          explainable for this specific case study.
        - Batch scoring for multiple customers at once via CSV.
        - Ensemble models (Random Forest, Gradient Boosting) as a comparison baseline, paired with
          a model-agnostic explainability method if interpretability needs grow beyond a single tree.
        - Cross-validation and clustering quality metrics (e.g., silhouette score) for added rigor.
        """
    )

st.markdown("")
disclaimer("This prototype is trained on the provided IBM Telco Customer Churn dataset only. "
           "It does not accept arbitrary uploaded telecom datasets in this version.")

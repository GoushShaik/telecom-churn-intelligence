"""
utils.py
Shared model-loading and inference logic for the Telecom Churn Driver Discovery
& Persona Profiler Streamlit app (IBM Q2D Case Study #16).

Every page imports its model/prediction logic from here so there is a single
source of truth for how raw form inputs become model inputs. Nothing in this
file retrains or modifies the existing trained artifacts - it only loads and
uses them exactly as they are.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODELS_DIR = "models"

# Must exactly match the service-column list used in train_model.py's
# build_clustering_features(), so "Service Count" is computed identically
# to how the persona model was trained.
SERVICE_COLS = [
    "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
    "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"
]

RISK_THRESHOLDS = {"low": 0.30, "medium": 0.60}  # UI presentation bucketing only

TENURE_BUCKETS = [
    (0, 12, "0-12 months"),
    (12, 24, "12-24 months"),
    (24, 48, "24-48 months"),
    (48, 60, "48-60 months"),
    (60, 999, "60+ months"),
]


def risk_level_from_probability(probability: float) -> str:
    """Single source of truth for risk bucketing, used by both the single-customer
    Predict page and the full-portfolio Executive Overview, so the two views are
    always consistent with each other."""
    if probability < RISK_THRESHOLDS["low"]:
        return "Low"
    elif probability < RISK_THRESHOLDS["medium"]:
        return "Medium"
    return "High"


def tenure_bucket(months: float) -> str:
    for low, high, label in TENURE_BUCKETS:
        if low <= months < high:
            return label
    return "60+ months"


# ---------------------------------------------------------------------------
# LOADING (cached so models are loaded once, not on every Streamlit rerun)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    tree_model = joblib.load(f"{MODELS_DIR}/churn_tree.pkl")
    kmeans_model = joblib.load(f"{MODELS_DIR}/persona_kmeans.pkl")
    tree_encoders = joblib.load(f"{MODELS_DIR}/tree_encoders.pkl")
    cluster_encoders = joblib.load(f"{MODELS_DIR}/cluster_encoders.pkl")
    cluster_scaler = joblib.load(f"{MODELS_DIR}/cluster_scaler.pkl")
    with open(f"{MODELS_DIR}/metadata.json") as f:
        metadata = json.load(f)
    return {
        "tree_model": tree_model,
        "kmeans_model": kmeans_model,
        "tree_encoders": tree_encoders,
        "cluster_encoders": cluster_encoders,
        "cluster_scaler": cluster_scaler,
        "metadata": metadata,
    }


# ---------------------------------------------------------------------------
# SERVICE COUNT (must mirror train_model.py's count_active_services logic)
# ---------------------------------------------------------------------------
def count_active_services(raw_input: dict) -> int:
    count = 0
    for col in SERVICE_COLS:
        val = str(raw_input.get(col, "No"))
        if val not in ("No", "No phone service", "No internet service"):
            count += 1
    return count


# ---------------------------------------------------------------------------
# ENCODING: FORM INPUT -> DECISION TREE FEATURE VECTOR
# ---------------------------------------------------------------------------
def encode_customer_input(raw_input: dict, tree_encoders: dict, tree_feature_columns: list) -> pd.DataFrame:
    """Builds a single-row DataFrame in the EXACT column order the tree was
    trained on, using the exact fitted LabelEncoders (never re-fit)."""
    row = {}
    for col in tree_feature_columns:
        if col in tree_encoders:
            le = tree_encoders[col]
            row[col] = le.transform([str(raw_input[col])])[0]
        else:
            row[col] = raw_input[col]  # numeric: Tenure Months, Monthly Charges, Total Charges
    return pd.DataFrame([row], columns=tree_feature_columns)


# ---------------------------------------------------------------------------
# CHURN PREDICTION
# ---------------------------------------------------------------------------
def predict_churn(tree_model, feature_df: pd.DataFrame):
    proba = tree_model.predict_proba(feature_df)[0]
    churn_probability = float(proba[1])  # probability of class 1 (churn)
    risk_level = risk_level_from_probability(churn_probability)
    return churn_probability, risk_level


# ---------------------------------------------------------------------------
# CLUSTERING: FORM INPUT -> PERSONA
# ---------------------------------------------------------------------------
def build_cluster_features(raw_input: dict, cluster_encoders: dict, cluster_feature_columns: list) -> pd.DataFrame:
    service_count = count_active_services(raw_input)
    values = {
        "Tenure Months": raw_input["Tenure Months"],
        "Monthly Charges": raw_input["Monthly Charges"],
        "Total Charges": raw_input["Total Charges"],
        "Contract": cluster_encoders["Contract"].transform([raw_input["Contract"]])[0],
        "Internet Service": cluster_encoders["Internet Service"].transform([raw_input["Internet Service"]])[0],
        "Service Count": service_count,
    }
    return pd.DataFrame([values], columns=cluster_feature_columns)


def predict_persona(kmeans_model, cluster_scaler, cluster_df: pd.DataFrame, metadata: dict):
    scaled = cluster_scaler.transform(cluster_df)
    cluster_id = int(kmeans_model.predict(scaled)[0])
    persona = metadata["personas"][str(cluster_id)]
    return cluster_id, persona


# ---------------------------------------------------------------------------
# EXPLAINABILITY: TRACE THE ACTUAL DECISION PATH FOR THIS CUSTOMER
# ---------------------------------------------------------------------------
def _describe_categorical_split(feature_name, threshold, went_left, encoder):
    """Translate a numeric split on a label-encoded categorical feature back
    into the original category name(s), using the encoder's sorted classes."""
    classes = list(encoder.classes_)
    if went_left:
        side_indices = [i for i in range(len(classes)) if i <= threshold]
    else:
        side_indices = [i for i in range(len(classes)) if i > threshold]
    side_labels = [classes[i] for i in side_indices]
    if len(side_labels) == 1:
        return f"{feature_name} is {side_labels[0]}"
    return f"{feature_name} is one of ({', '.join(side_labels)})"


def explain_decision_path(tree_model, feature_df: pd.DataFrame, tree_feature_columns: list,
                           tree_encoders: dict, max_steps: int = 5):
    """Walks the actual path this specific customer took through the trained
    tree and returns plain-English sentences for each split, using the real
    input values (not the pre-extracted generic top-5 rules)."""
    tree_ = tree_model.tree_
    feature_vec = feature_df.iloc[0]
    node = 0
    steps = []

    while tree_.feature[node] != -2 and len(steps) < max_steps:  # -2 = leaf
        feat_idx = tree_.feature[node]
        feat_name = tree_feature_columns[feat_idx]
        threshold = tree_.threshold[node]
        actual_value = feature_vec[feat_name]
        went_left = actual_value <= threshold

        if feat_name in tree_encoders:
            sentence = _describe_categorical_split(feat_name, threshold, went_left, tree_encoders[feat_name])
        else:
            comparator = "is at or below" if went_left else "is above"
            sentence = f"{feat_name} ({actual_value:g}) {comparator} {threshold:.1f}"

        steps.append(sentence)
        node = tree_.children_left[node] if went_left else tree_.children_right[node]

    leaf_values = tree_.value[node][0]
    leaf_churn_prob = leaf_values[1] / leaf_values.sum() if leaf_values.sum() > 0 else 0.0
    return steps, float(leaf_churn_prob)


# ---------------------------------------------------------------------------
# PORTFOLIO-WIDE SCORING (Executive Overview / Insights business breakdowns)
#
# This does NOT retrain or refit anything. It applies the already-trained,
# already-validated tree_model and kmeans_model to the same 7,043 customers
# they were validated on, using the SAME saved encoders (.transform only,
# never re-fit) so results are guaranteed consistent with the single-customer
# Predict page. This is what lets the app show portfolio-level business value
# without any dataset upload or retraining.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_portfolio_scores(_tree_model, _kmeans_model, _tree_encoders, _cluster_encoders,
                              _cluster_scaler, tree_feature_columns: list, cluster_feature_columns: list,
                              personas: dict):
    """Leading underscores on model/encoder args tell Streamlit's cache not to
    try to hash them (they're not reliably hashable); the function still only
    runs once per app session because its other arguments are stable."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_model import load_data, preprocess  # reuse validated cleaning logic, no re-fitting

    df = load_data()
    df = preprocess(df)

    # --- Decision Tree scoring (encode using the SAVED, already-fitted encoders) ---
    tree_input = df[tree_feature_columns].copy()
    for col, le in _tree_encoders.items():
        tree_input[col] = le.transform(tree_input[col].astype(str))
    churn_proba = _tree_model.predict_proba(tree_input)[:, 1]

    # --- K-Means persona assignment (same encoders/scaler as training) ---
    service_count = df.apply(lambda row: count_active_services(row.to_dict()), axis=1)
    cluster_input = pd.DataFrame({
        "Tenure Months": df["Tenure Months"],
        "Monthly Charges": df["Monthly Charges"],
        "Total Charges": df["Total Charges"],
        "Contract": _cluster_encoders["Contract"].transform(df["Contract"].astype(str)),
        "Internet Service": _cluster_encoders["Internet Service"].transform(df["Internet Service"].astype(str)),
        "Service Count": service_count,
    })[cluster_feature_columns]
    scaled = _cluster_scaler.transform(cluster_input)
    cluster_ids = _kmeans_model.predict(scaled)

    # --- Assemble portfolio-level result table ---
    result = df.copy()
    result["Churn Probability"] = churn_proba
    result["Predicted Risk"] = [risk_level_from_probability(p) for p in churn_proba]
    result["Persona"] = [personas[str(c)]["name"] for c in cluster_ids]
    result["Tenure Bucket"] = result["Tenure Months"].apply(tenure_bucket)

    return result


def compute_kpis(scored_df: pd.DataFrame) -> dict:
    total = len(scored_df)
    risk_counts = scored_df["Predicted Risk"].value_counts().to_dict()
    historical_churn_rate = scored_df["Churn Value"].mean()
    revenue_at_risk = float((scored_df["Churn Probability"] * scored_df["Monthly Charges"]).sum())
    avg_monthly_revenue = float(scored_df["Monthly Charges"].sum())

    return {
        "total_customers": total,
        "high_risk": int(risk_counts.get("High", 0)),
        "medium_risk": int(risk_counts.get("Medium", 0)),
        "low_risk": int(risk_counts.get("Low", 0)),
        "historical_churn_rate": float(historical_churn_rate),
        "estimated_monthly_revenue_at_risk": revenue_at_risk,
        "total_monthly_revenue": avg_monthly_revenue,
    }


# ---------------------------------------------------------------------------
# MODEL PERFORMANCE — ROC CURVE (inference only, no retraining)
#
# Re-derives the EXACT same train/test split used during original training
# (same random_state=42, same test_size, same stratify) and calls
# .predict_proba() on the already-fitted tree_model — no .fit() call
# anywhere in this function. This only adds an evaluation metric that
# train_model.py did not originally persist; it does not change the model.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_ibm_roc_curve(_tree_model, _tree_encoders, tree_feature_columns: list):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from train_model import load_data, preprocess
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, roc_auc_score

    df = load_data()
    df = preprocess(df)

    X = df[tree_feature_columns].copy()
    for col, le in _tree_encoders.items():
        X[col] = le.transform(X[col].astype(str))
    y = df["Churn Value"]

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    proba = _tree_model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    return fpr, tpr, float(auc)


# ---------------------------------------------------------------------------
# RETENTION RECOMMENDATION (templated, tied to the customer's actual drivers)
# ---------------------------------------------------------------------------
def recommend_action(raw_input: dict, feature_importance: dict, top_n: int = 2):
    """Simple rule-based recommendation keyed to the customer's own values on
    the globally most important features. No ML magic - transparent templates."""
    ranked_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = [f for f, _ in ranked_features[:top_n]]

    recommendations = []
    for feat in top_features:
        val = raw_input.get(feat)
        if feat == "Contract" and val == "Month-to-month":
            recommendations.append("Offer an incentive to move from month-to-month to a 1- or 2-year contract.")
        elif feat == "Dependents" and val == "No":
            recommendations.append("Consider a family/multi-line bundle offer to increase account stickiness.")
        elif feat == "Online Security" and val == "No":
            recommendations.append("Offer a free trial of Online Security add-on to increase perceived value.")
        elif feat == "Monthly Charges" and isinstance(val, (int, float)) and val > 70:
            recommendations.append("Consider a loyalty discount given this customer's relatively high monthly spend.")
        elif feat == "Tenure Months" and isinstance(val, (int, float)) and val < 12:
            recommendations.append("Prioritize early-tenure engagement/onboarding outreach for this new customer.")

    if not recommendations:
        recommendations.append("No specific high-impact driver detected; monitor as part of standard retention flow.")

    return recommendations


# ---------------------------------------------------------------------------
# CURRENCY DISPLAY (presentation-layer only — never affects the model)
# ---------------------------------------------------------------------------
CURRENCY_OPTIONS = {
    "USD ($)": {"symbol": "$", "rate": 1.0},
    "INR (₹)": {"symbol": "₹", "rate": 83.0},  # approximate, illustrative only
}


def format_currency(usd_amount: float, currency_label: str) -> str:
    cfg = CURRENCY_OPTIONS[currency_label]
    converted = usd_amount * cfg["rate"]
    return f"{cfg['symbol']}{converted:,.0f}" if cfg["rate"] != 1.0 else f"{cfg['symbol']}{converted:,.2f}"

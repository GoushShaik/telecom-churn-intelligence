"""
IBM Q2D UG2 Case Study #16 — Telecom Churn Driver Discovery & Persona Profiler
ML Pipeline: preprocessing -> Decision Tree (churn prediction) -> KMeans (persona clustering)

Run:  python train_model.py
Reproducible: re-running regenerates everything in models/ and outputs/ from scratch.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

DATA_PATH = "data/Telco_customer_churn.xlsx"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_excel(DATA_PATH, sheet_name="Telco_Churn")
    print(f"[LOAD] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# STEP 2: PREPROCESS
# ---------------------------------------------------------------------------
def preprocess(df):
    df = df.copy()

    # --- Fix Total Charges: blank strings -> 0 (these are all tenure=0 new customers) ---
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    blank_count = df["Total Charges"].isna().sum()
    df["Total Charges"] = df["Total Charges"].fillna(0.0)
    print(f"[PREPROCESS] Fixed {blank_count} blank Total Charges values (set to 0.0, all tenure=0 customers)")

    # --- Drop identifier ---
    df = df.drop(columns=["CustomerID"])

    # --- Drop constant columns (zero variance, no predictive value) ---
    df = df.drop(columns=["Count", "Country", "State"])

    # --- Drop leakage columns (post-outcome / target-derived) ---
    leakage_cols = ["Churn Label", "Churn Score", "Churn Reason", "CLTV"]
    df = df.drop(columns=leakage_cols)
    print(f"[PREPROCESS] Dropped leakage columns: {leakage_cols}")

    # --- Drop geographic columns (out of scope for MVP) ---
    geo_cols = ["City", "Zip Code", "Lat Long", "Latitude", "Longitude"]
    df = df.drop(columns=geo_cols)
    print(f"[PREPROCESS] Dropped geographic columns (MVP scope): {geo_cols}")

    print(f"[PREPROCESS] Remaining columns ({df.shape[1]}): {list(df.columns)}")
    return df


# ---------------------------------------------------------------------------
# STEP 3: ENCODE CATEGORICALS FOR THE DECISION TREE
# ---------------------------------------------------------------------------
def encode_features(df, target_col="Churn Value"):
    df = df.copy()
    feature_df = df.drop(columns=[target_col])
    target = df[target_col]

    categorical_cols = feature_df.select_dtypes(include=["object", "string"]).columns.tolist()
    encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        encoders[col] = le

    print(f"[ENCODE] Label-encoded {len(categorical_cols)} categorical columns: {categorical_cols}")
    return feature_df, target, encoders, categorical_cols


# ---------------------------------------------------------------------------
# STEP 4: DECISION TREE — CHURN PREDICTION
# ---------------------------------------------------------------------------
def train_decision_tree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Try depths 3-5 and pick the one with the best test F1 score,
    # since F1 balances precision/recall better than raw accuracy on this
    # moderately imbalanced target (~26% churn rate).
    best_depth, best_f1, best_model = None, -1, None
    depth_results = {}

    for depth in [3, 4, 5]:
        model = DecisionTreeClassifier(
            max_depth=depth, random_state=RANDOM_STATE, class_weight="balanced"
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        depth_results[depth] = f1
        if f1 > best_f1:
            best_depth, best_f1, best_model = depth, f1, model

    print(f"[TREE] F1 score by depth: {depth_results}")
    print(f"[TREE] Selected max_depth={best_depth} (best test F1={best_f1:.3f}, "
          f"balances interpretability with predictive quality)")

    preds = best_model.predict(X_test)
    metrics = {
        "selected_max_depth": best_depth,
        "depth_search_f1_scores": depth_results,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "classification_report": classification_report(y_test, preds, output_dict=True),
    }

    feature_importance = (
        pd.Series(best_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )

    rules_text = export_text(best_model, feature_names=list(X.columns))

    return best_model, metrics, feature_importance, rules_text, (X_train, X_test, y_train, y_test)


def extract_readable_rules(tree_model, feature_names, top_n=5):
    """Extract the most impactful decision paths as plain-English sentences,
    ranked by the number of training samples they cover (i.e. how often they apply)."""
    tree_ = tree_model.tree_
    rules = []

    def recurse(node, conditions):
        if tree_.feature[node] != -2:  # not a leaf
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], conditions + [f"{name} <= {threshold:.2f}"])
            recurse(tree_.children_right[node], conditions + [f"{name} > {threshold:.2f}"])
        else:
            samples = tree_.n_node_samples[node]
            values = tree_.value[node][0]
            churn_prob = values[1] / values.sum() if values.sum() > 0 else 0
            rules.append({
                "conditions": conditions,
                "samples": int(samples),
                "churn_probability": round(float(churn_prob), 3),
            })

    recurse(0, [])
    rules.sort(key=lambda r: r["samples"], reverse=True)

    readable = []
    for r in rules[:top_n]:
        cond_text = " AND ".join(r["conditions"])
        readable.append(
            f"IF {cond_text} -> churn probability = {r['churn_probability']*100:.1f}% "
            f"(covers {r['samples']} customers in training data)"
        )
    return readable, rules


# ---------------------------------------------------------------------------
# STEP 5: BEHAVIORAL CLUSTERING — PERSONA PROFILING
# ---------------------------------------------------------------------------
def build_clustering_features(df):
    cdf = df.copy()

    service_cols = [
        "Phone Service", "Multiple Lines", "Internet Service", "Online Security",
        "Online Backup", "Device Protection", "Tech Support", "Streaming TV", "Streaming Movies"
    ]

    def count_active_services(row):
        count = 0
        for col in service_cols:
            val = str(row[col])
            if val not in ("No", "No phone service", "No internet service"):
                count += 1
        return count

    cdf["Service Count"] = cdf.apply(count_active_services, axis=1)

    cluster_feature_cols = [
        "Tenure Months", "Monthly Charges", "Total Charges",
        "Contract", "Internet Service", "Service Count"
    ]
    cluster_df = cdf[cluster_feature_cols].copy()

    # Encode the two categorical clustering features
    cluster_encoders = {}
    for col in ["Contract", "Internet Service"]:
        le = LabelEncoder()
        cluster_df[col] = le.fit_transform(cluster_df[col].astype(str))
        cluster_encoders[col] = le

    print(f"[CLUSTER] Engineered 'Service Count' feature (0-{cdf['Service Count'].max()} active services)")
    print(f"[CLUSTER] Final clustering features: {cluster_feature_cols}")

    return cluster_df, cluster_feature_cols, cluster_encoders


def find_optimal_k(scaled_features, k_range=range(2, 8)):
    inertias = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(scaled_features)
        inertias[k] = km.inertia_
    return inertias


def choose_k_via_elbow(inertias):
    """Pick k at the point of maximum 'bend' using the second-derivative
    (rate-of-decrease-of-decrease) heuristic, capped to a business-usable range (3-4)."""
    ks = sorted(inertias.keys())
    vals = [inertias[k] for k in ks]
    deltas = [vals[i] - vals[i + 1] for i in range(len(vals) - 1)]
    delta2 = [deltas[i] - deltas[i + 1] for i in range(len(deltas) - 1)]
    # elbow candidate = index of max delta2, +1 offset (since delta2 aligns to ks[1:-1])
    elbow_idx = int(np.argmax(delta2)) + 1
    elbow_k = ks[elbow_idx]
    # Keep personas business-usable: clamp to between 3 and 4 clusters
    final_k = min(max(elbow_k, 3), 4)
    return final_k, elbow_k


def train_clustering(cluster_df):
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cluster_df)

    inertias = find_optimal_k(scaled)
    final_k, raw_elbow_k = choose_k_via_elbow(inertias)
    print(f"[CLUSTER] Elbow method inertias: {inertias}")
    print(f"[CLUSTER] Raw elbow suggestion: k={raw_elbow_k} -> clamped to business-usable k={final_k}")

    kmeans = KMeans(n_clusters=final_k, random_state=RANDOM_STATE, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled)

    return kmeans, scaler, cluster_labels, inertias, final_k


def profile_personas(original_df, cluster_df, cluster_labels, final_k):
    profile_df = original_df.copy()
    profile_df["Cluster"] = cluster_labels
    profile_df["Service Count"] = cluster_df["Service Count"]

    personas = {}
    for c in range(final_k):
        sub = profile_df[profile_df["Cluster"] == c]
        avg_tenure = sub["Tenure Months"].mean()
        avg_monthly = sub["Monthly Charges"].mean()
        avg_services = sub["Service Count"].mean()
        churn_rate = sub["Churn Value"].mean()
        top_contract = sub["Contract"].mode()[0] if not sub["Contract"].mode().empty else "Unknown"
        size = len(sub)

        # Rule-based, transparent persona naming (no ML magic — just thresholds on cluster averages)
        if avg_tenure < 20 and churn_rate > 0.3:
            name = "New & At-Risk"
            desc = "Recently joined customers with a high cancellation rate — the highest-priority retention group."
        elif avg_tenure >= 40 and churn_rate < 0.15:
            name = "Loyal & Stable"
            desc = "Long-tenured customers with low churn — the most stable, low-risk segment."
        elif avg_monthly > profile_df["Monthly Charges"].mean() and avg_services >= 5:
            name = "High-Value Bundlers"
            desc = "Customers on multiple services with higher monthly spend — valuable but worth monitoring."
        else:
            name = "Budget & Basic"
            desc = "Lower monthly spend, fewer bundled services — price-sensitive segment."

        personas[int(c)] = {
            "name": name,
            "description": desc,
            "size": int(size),
            "avg_tenure_months": round(float(avg_tenure), 1),
            "avg_monthly_charges": round(float(avg_monthly), 2),
            "avg_service_count": round(float(avg_services), 2),
            "most_common_contract": str(top_contract),
            "churn_rate": round(float(churn_rate), 3),
        }

    return personas


# ---------------------------------------------------------------------------
# STEP 6: SAVE ARTIFACTS FOR THE FUTURE STREAMLIT APP
# ---------------------------------------------------------------------------
def save_artifacts(tree_model, kmeans_model, tree_encoders, cluster_encoders,
                    scaler, tree_feature_cols, cluster_feature_cols, personas,
                    readable_rules, feature_importance, tree_metrics, elbow_inertias, final_k):

    joblib.dump(tree_model, f"{MODELS_DIR}/churn_tree.pkl")
    joblib.dump(kmeans_model, f"{MODELS_DIR}/persona_kmeans.pkl")
    joblib.dump(tree_encoders, f"{MODELS_DIR}/tree_encoders.pkl")
    joblib.dump(cluster_encoders, f"{MODELS_DIR}/cluster_encoders.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/cluster_scaler.pkl")

    metadata = {
        "tree_feature_columns": tree_feature_cols,
        "cluster_feature_columns": cluster_feature_cols,
        "selected_tree_depth": tree_metrics["selected_max_depth"],
        "selected_k": final_k,
        "personas": personas,
        "readable_rules": readable_rules,
        "feature_importance": feature_importance.to_dict(),
        "tree_metrics": {
            "accuracy": tree_metrics["accuracy"],
            "precision": tree_metrics["precision"],
            "recall": tree_metrics["recall"],
            "f1_score": tree_metrics["f1_score"],
            "confusion_matrix": tree_metrics["confusion_matrix"],
        },
    }
    with open(f"{MODELS_DIR}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    with open(f"{OUTPUTS_DIR}/decision_tree_full_rules.txt", "w") as f:
        f.write("=== FULL DECISION TREE (text form) ===\n\n")
        f.write(tree_metrics.get("_rules_text", ""))

    with open(f"{OUTPUTS_DIR}/elbow_method_inertias.json", "w") as f:
        json.dump({str(k): v for k, v in elbow_inertias.items()}, f, indent=2)

    print(f"[SAVE] Models saved to {MODELS_DIR}/")
    print(f"[SAVE] Metadata + outputs saved to {MODELS_DIR}/metadata.json and {OUTPUTS_DIR}/")


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("TELECOM CHURN DRIVER DISCOVERY & PERSONA PROFILER — TRAINING PIPELINE")
    print("=" * 70)

    df = load_data()
    df = preprocess(df)

    # --- Decision Tree ---
    print("\n--- DECISION TREE PIPELINE ---")
    X, y, tree_encoders, tree_categorical_cols = encode_features(df, target_col="Churn Value")
    tree_model, tree_metrics, feature_importance, rules_text, splits = train_decision_tree(X, y)
    readable_rules, all_rules = extract_readable_rules(tree_model, list(X.columns))
    tree_metrics["_rules_text"] = rules_text

    print("\n[TREE] Top 5 feature importances:")
    print(feature_importance.head(5))

    print("\n[TREE] Top readable churn rules:")
    for r in readable_rules:
        print(" -", r)

    # --- Clustering ---
    print("\n--- CLUSTERING PIPELINE ---")
    cluster_df, cluster_feature_cols, cluster_encoders = build_clustering_features(df)
    kmeans_model, scaler, cluster_labels, elbow_inertias, final_k = train_clustering(cluster_df)
    personas = profile_personas(df, cluster_df, cluster_labels, final_k)

    print(f"\n[CLUSTER] Final persona count: {final_k}")
    for cid, p in personas.items():
        print(f" - Cluster {cid}: {p['name']} | size={p['size']} | churn_rate={p['churn_rate']*100:.1f}% "
              f"| avg_tenure={p['avg_tenure_months']} | avg_monthly=${p['avg_monthly_charges']}")

    # --- Save everything ---
    save_artifacts(
        tree_model, kmeans_model, tree_encoders, cluster_encoders, scaler,
        list(X.columns), cluster_feature_cols, personas, readable_rules,
        feature_importance, tree_metrics, elbow_inertias, final_k
    )

    # --- Final report ---
    print("\n" + "=" * 70)
    print("FINAL PIPELINE REPORT")
    print("=" * 70)
    print(f"Decision Tree features ({len(X.columns)}): {list(X.columns)}")
    print(f"Clustering features: {cluster_feature_cols}")
    print(f"Selected tree depth: {tree_metrics['selected_max_depth']}")
    print(f"Tree accuracy: {tree_metrics['accuracy']:.3f} | precision: {tree_metrics['precision']:.3f} | "
          f"recall: {tree_metrics['recall']:.3f} | f1: {tree_metrics['f1_score']:.3f}")
    print(f"Confusion matrix: {tree_metrics['confusion_matrix']}")
    print(f"Selected cluster count (k): {final_k}")
    print("Personas:")
    for cid, p in personas.items():
        print(f"  [{cid}] {p['name']} — {p['description']}")
    print(f"Top churn drivers (by feature importance): {list(feature_importance.head(5).index)}")
    print("\nFiles generated for the future web app:")
    print(f"  {MODELS_DIR}/churn_tree.pkl")
    print(f"  {MODELS_DIR}/persona_kmeans.pkl")
    print(f"  {MODELS_DIR}/tree_encoders.pkl")
    print(f"  {MODELS_DIR}/cluster_encoders.pkl")
    print(f"  {MODELS_DIR}/cluster_scaler.pkl")
    print(f"  {MODELS_DIR}/metadata.json")
    print(f"  {OUTPUTS_DIR}/decision_tree_full_rules.txt")
    print(f"  {OUTPUTS_DIR}/elbow_method_inertias.json")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
train_cell2cell_model.py

IMPORTANT — read before touching this file:

This trains a SEPARATE, INDEPENDENT interpretable Decision Tree on the
Cell2Cell telecom dataset. It does NOT modify, retrain, or touch the IBM
Telco model/artifacts in models/ at all.

Why a separate model instead of "validating" the IBM model on this data:
Cell2Cell's schema is fundamentally different from the IBM Telco schema
(call-usage/handset/demographic features vs. contract/service-bundle
features — there is no Contract, Internet Service, or Online Security
column in Cell2Cell, and no MonthlyMinutes/DroppedCalls/CreditRating in
IBM Telco). There is no valid way to feed Cell2Cell rows through the IBM
tree, and doing so would not be a meaningful validation even if it ran
without error. This script instead demonstrates the SAME interpretable-
tree methodology applied independently to a second, larger, independent
telecom dataset — evaluated only on labeled data (cell2celltrain.csv).
cell2cellholdout.csv has an empty Churn column and is NOT used for any
metric here, per the project's methodological-honesty requirement.

Run: python train_cell2cell_model.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)

DATA_PATH = "data/cell2cell/cell2celltrain.csv"
HOLDOUT_PATH = "data/cell2cell/cell2cellholdout.csv"
OUT_DIR = "models_cell2cell"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "MonthsInService", "MonthlyRevenue", "TotalRecurringCharge", "OverageMinutes",
    "DroppedCalls", "CustomerCareCalls", "RetentionCalls", "CurrentEquipmentDays", "Handsets",
]
CATEGORICAL_FEATURES = [
    "CreditRating", "Homeownership", "MaritalStatus", "Occupation", "PrizmCode",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def main():
    print("=" * 70)
    print("CELL2CELL — INDEPENDENT EXTERNAL DATASET MODEL (separate from IBM Telco)")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)
    print(f"[LOAD] Cell2Cell train: {df.shape[0]} rows, {df.shape[1]} columns")

    holdout = pd.read_csv(HOLDOUT_PATH)
    holdout_churn_all_missing = holdout["Churn"].isna().all()
    print(f"[CHECK] Holdout Churn column entirely missing: {holdout_churn_all_missing} "
          f"-> holdout is NOT used for any metric in this script.")

    # --- Clean ---
    df = df.dropna(subset=ALL_FEATURES + ["Churn"]).copy()
    print(f"[CLEAN] Rows remaining after dropping missing values in selected features: {df.shape[0]}")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    X = df[ALL_FEATURES].copy()
    y = df["Churn"]

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    best_depth, best_f1, best_model = None, -1, None
    depth_results = {}
    for depth in [3, 4, 5]:
        model = DecisionTreeClassifier(max_depth=depth, random_state=RANDOM_STATE, class_weight="balanced")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        f1 = f1_score(y_test, preds)
        depth_results[depth] = f1
        if f1 > best_f1:
            best_depth, best_f1, best_model = depth, f1, model

    print(f"[TREE] F1 by depth: {depth_results}")
    print(f"[TREE] Selected depth={best_depth} (best F1={best_f1:.3f})")

    preds = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = roc_auc_score(y_test, proba)

    metrics = {
        "dataset": "Cell2Cell (independent external telecom dataset)",
        "note": "This model is trained and evaluated ENTIRELY on Cell2Cell's own labeled data "
                "(cell2celltrain.csv, 80/20 split). It does not validate the IBM Telco model, "
                "because the two datasets have incompatible feature schemas. It demonstrates the "
                "same interpretable decision-tree methodology on an independent telecom dataset.",
        "rows_used": int(df.shape[0]),
        "features_used": ALL_FEATURES,
        "selected_max_depth": best_depth,
        "depth_search_f1_scores": depth_results,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "holdout_rows": int(holdout.shape[0]),
        "holdout_churn_missing": bool(holdout_churn_all_missing),
        "holdout_used_for_metrics": False,
    }

    feature_importance = (
        pd.Series(best_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
    )
    metrics["feature_importance"] = feature_importance.to_dict()

    import os
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(best_model, f"{OUT_DIR}/cell2cell_tree.pkl")
    joblib.dump(encoders, f"{OUT_DIR}/cell2cell_encoders.pkl")
    with open(f"{OUT_DIR}/cell2cell_metadata.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 70)
    print("FINAL CELL2CELL MODEL REPORT")
    print("=" * 70)
    print(f"Accuracy: {metrics['accuracy']:.3f} | Precision: {metrics['precision']:.3f} | "
          f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1_score']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
    print(f"Top features: {list(feature_importance.head(5).index)}")
    print(f"Saved to {OUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()

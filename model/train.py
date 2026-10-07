"""Train the XGBoost fraud classifier.

Splits first, fits category_stats on the training split only, tunes the
threshold on validation, and reports final metrics on the held-out test set.
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from pipeline.features import build_features, raw_to_frame

DATA_PATH = os.environ.get("VALIDATED_PATH", "data/validated.jsonl")
MODEL_DIR = os.environ.get("MODEL_DIR", "model")


def load_docs(path: str) -> pd.DataFrame:
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line))
    return raw_to_frame(docs)


def main():
    print("Loading validated records...")
    df = load_docs(DATA_PATH)
    print(f"  {len(df)} records, fraud rate = {df['is_fraud'].mean():.4%}")

    # Split before computing any statistic.
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["is_fraud"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["is_fraud"], random_state=42
    )
    print(f"  train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    # Fit stats on train only, reuse for val/test.
    X_train, category_stats = build_features(train_df)
    X_val, _ = build_features(val_df, category_stats=category_stats)
    X_test, _ = build_features(test_df, category_stats=category_stats)

    y_train, y_val, y_test = train_df["is_fraud"], val_df["is_fraud"], test_df["is_fraud"]

    feature_columns = list(X_train.columns)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    # Config selected by model/experiments.py.
    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "eta": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "scale_pos_weight": 1,
        "seed": 42,
    }

    print("Training XGBoost...")
    evals_result = {}
    booster = xgb.train(
        params, dtrain,
        num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=25,
        evals_result=evals_result,
        verbose_eval=False,
    )
    print(f"  best iteration: {booster.best_iteration}")

    # Pick the decision threshold on val, score once on test.
    val_probs = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    best_thresh, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, (val_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    print(f"  threshold chosen on val: {best_thresh:.2f} (val F1={best_f1:.4f})")

    test_probs = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    test_preds = (test_probs >= best_thresh).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "threshold": float(best_thresh),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "fraud_rate": float(df["is_fraud"].mean()),
        "confusion_matrix": confusion_matrix(y_test, test_preds).tolist(),
        "best_iteration": int(booster.best_iteration),
    }
    print(json.dumps(metrics, indent=2))
    print(classification_report(y_test, test_preds, target_names=["legit", "fraud"]))

    importance = booster.get_score(importance_type="gain")
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))

    booster.save_model(f"{MODEL_DIR}/fraud_model.json")
    with open(f"{MODEL_DIR}/feature_columns.json", "w") as f:
        json.dump(feature_columns, f)
    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{MODEL_DIR}/feature_importance.json", "w") as f:
        json.dump(importance, f, indent=2)
    category_stats.to_json(f"{MODEL_DIR}/category_stats.json", orient="index")

    print(f"\nSaved model + artifacts to {MODEL_DIR}/")


if __name__ == "__main__":
    main()

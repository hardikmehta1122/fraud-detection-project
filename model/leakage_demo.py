"""Quantify the effect of fitting category_stats before the split.

Trains with category_stats computed on the full dataset (leaky) and
compares test metrics against the train-only pipeline in metrics.json.
Run after model/train.py.
"""

import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from model.train import DATA_PATH, MODEL_DIR, load_docs
from pipeline.features import build_features


def main():
    df = load_docs(DATA_PATH)

    # Leaky path: fit category_stats on the full dataset, then split.
    _, leaky_category_stats = build_features(df)
    y_all = df["is_fraud"]

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=y_all, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["is_fraud"], random_state=42)

    X_train, _ = build_features(train_df, category_stats=leaky_category_stats)
    X_val, _ = build_features(val_df, category_stats=leaky_category_stats)
    X_test, _ = build_features(test_df, category_stats=leaky_category_stats)
    y_train, y_val, y_test = train_df["is_fraud"], val_df["is_fraud"], test_df["is_fraud"]

    feature_columns = list(X_train.columns)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    params = {
        "objective": "binary:logistic", "eval_metric": "aucpr", "max_depth": 6,
        "eta": 0.08, "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 3, "scale_pos_weight": scale_pos_weight, "seed": 42,
    }
    booster = xgb.train(
        params, dtrain, num_boost_round=400,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=25, verbose_eval=False,
    )

    val_probs = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    best_thresh, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, (val_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t

    test_probs = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    test_preds = (test_probs >= best_thresh).astype(int)

    leaky_metrics = {
        "accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(y_test, test_preds),
        "recall": recall_score(y_test, test_preds),
        "f1": f1_score(y_test, test_preds),
        "roc_auc": roc_auc_score(y_test, test_probs),
        "threshold": float(best_thresh),
    }

    with open(f"{MODEL_DIR}/metrics.json") as f:
        fixed_metrics = json.load(f)

    print(f"{'metric':<12}{'leaky (buggy)':<16}{'fixed':<10}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"{k:<12}{leaky_metrics[k]:<16.4f}{fixed_metrics[k]:<10.4f}")

    with open(f"{MODEL_DIR}/leakage_comparison.json", "w") as f:
        json.dump({"leaky": leaky_metrics, "fixed": fixed_metrics}, f, indent=2)


if __name__ == "__main__":
    main()

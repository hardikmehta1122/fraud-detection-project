"""Quantify the effect of training on unvalidated records.

Compares a model trained on clean (validated) data against one trained on
the same data plus the malformed records ingestion would reject. Both are
scored on the same clean held-out test set.
"""

import json
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from pydantic import ValidationError
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from model.train import DATA_PATH, MODEL_DIR, load_docs
from pipeline.features import build_features, raw_to_frame
from pipeline.schemas import FinancialDocument

RAW_PATH = os.environ.get("RAW_PATH", "data/raw_extracted.jsonl")

TRAIN_PARAMS = {
    "objective": "binary:logistic", "eval_metric": "aucpr",
    "max_depth": 6, "eta": 0.08, "subsample": 0.85,
    "colsample_bytree": 0.85, "min_child_weight": 3,
    "scale_pos_weight": 1, "seed": 42,
}


def load_raw_and_split_by_validity():
    clean_ids = set()
    corrupted_raw = []
    with open(RAW_PATH) as f:
        for line in f:
            raw = json.loads(line)
            try:
                FinancialDocument.model_validate(raw)
                clean_ids.add(raw["document_id"])
            except ValidationError:
                corrupted_raw.append(raw)
    return clean_ids, corrupted_raw


def fit_eval(train_df, val_df, test_df, feature_columns_ref=None):
    X_train, category_stats = build_features(train_df)
    X_val, _ = build_features(val_df, category_stats=category_stats)
    X_test, _ = build_features(test_df, category_stats=category_stats)
    # Malformed input can produce infinities; XGBoost rejects inf but
    # treats NaN as missing, so coerce inf -> NaN.
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_val = X_val.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y_train, y_val, y_test = train_df["is_fraud"], val_df["is_fraud"], test_df["is_fraud"]
    feature_columns = list(X_train.columns)

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    booster = xgb.train(
        TRAIN_PARAMS, dtrain, num_boost_round=400,
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

    return {
        "n_train": len(train_df),
        "threshold": float(best_thresh),
        "accuracy": float(accuracy_score(y_test, test_preds)),
        "precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, test_probs)),
    }


def main():
    print("Loading validated records for the reference split...")
    df = load_docs(DATA_PATH)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["is_fraud"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["is_fraud"], random_state=42)
    print(f"  clean split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    print("Identifying raw records that fail validation...")
    _, corrupted_raw = load_raw_and_split_by_validity()
    print(f"  {len(corrupted_raw)} malformed records in the raw feed")

    corrupted_df = raw_to_frame(corrupted_raw)

    print("\nClean training data:")
    fixed_metrics = fit_eval(train_df, val_df, test_df)
    print(json.dumps(fixed_metrics, indent=2))

    print("\nWith unvalidated records added to training:")
    with np.errstate(invalid="ignore", divide="ignore"):
        buggy_train_df = pd.concat([train_df, corrupted_df], ignore_index=True)
        buggy_metrics = fit_eval(buggy_train_df, val_df, test_df)
    print(json.dumps(buggy_metrics, indent=2))

    print(f"\n{'metric':<12}{'buggy':<12}{'fixed':<12}{'delta':<10}")
    for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        delta = buggy_metrics[k] - fixed_metrics[k]
        print(f"{k:<12}{buggy_metrics[k]:<12.4f}{fixed_metrics[k]:<12.4f}{delta:+.4f}")

    with open(f"{MODEL_DIR}/validation_timing_leakage_comparison.json", "w") as f:
        json.dump({"buggy": buggy_metrics, "fixed": fixed_metrics,
                    "n_corrupted_records": len(corrupted_raw)}, f, indent=2)


if __name__ == "__main__":
    main()

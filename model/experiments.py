"""Compare model configurations on a fixed split.

Same data, split and features throughout; only the config and
imbalance-handling strategy vary. The winner is picked by validation F1
and written to EXPERIMENTS.md.
"""

import json
import os

import numpy as np
import pandas as pd
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


def pick_threshold(y_val, val_probs):
    best_thresh, best_f1 = 0.5, -1
    for t in np.arange(0.05, 0.95, 0.01):
        f1 = f1_score(y_val, (val_probs >= t).astype(int))
        if f1 > best_f1:
            best_f1, best_thresh = f1, t
    return best_thresh, best_f1


def run_config(name, params, num_boost_round, X_train, y_train, X_val, y_val, X_test, y_test,
               feature_columns, oversample=False):
    Xt, yt = X_train, y_train
    if oversample:
        # Duplicate minority rows in the training split toward balance.
        minority = X_train[y_train == 1]
        minority_y = y_train[y_train == 1]
        n_majority = (y_train == 0).sum()
        reps = max(1, n_majority // max(len(minority), 1))
        Xt = pd.concat([X_train] + [minority] * reps, ignore_index=True)
        yt = pd.concat([y_train] + [minority_y] * reps, ignore_index=True)

    dtrain = xgb.DMatrix(Xt, label=yt, feature_names=feature_columns)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_columns)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_columns)

    booster = xgb.train(
        params, dtrain, num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=25, verbose_eval=False,
    )

    val_probs = booster.predict(dval, iteration_range=(0, booster.best_iteration + 1))
    threshold, val_f1 = pick_threshold(y_val, val_probs)

    test_probs = booster.predict(dtest, iteration_range=(0, booster.best_iteration + 1))
    test_preds = (test_probs >= threshold).astype(int)

    return {
        "name": name,
        "params": {k: v for k, v in params.items() if k not in ("seed", "eval_metric", "objective")},
        "oversampled": oversample,
        "best_iteration": int(booster.best_iteration),
        "threshold": float(threshold),
        "val_f1": float(val_f1),
        "test_accuracy": float(accuracy_score(y_test, test_preds)),
        "test_precision": float(precision_score(y_test, test_preds, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_preds, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_preds, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_probs)),
    }


def main():
    print("Loading validated records...")
    df = load_docs(DATA_PATH)

    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df["is_fraud"], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df["is_fraud"], random_state=42)

    X_train, category_stats = build_features(train_df)
    X_val, _ = build_features(val_df, category_stats=category_stats)
    X_test, _ = build_features(test_df, category_stats=category_stats)
    y_train, y_val, y_test = train_df["is_fraud"], val_df["is_fraud"], test_df["is_fraud"]
    feature_columns = list(X_train.columns)

    balanced_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    base = {"objective": "binary:logistic", "eval_metric": "aucpr", "seed": 42}

    configs = [
        dict(name="baseline_balanced",
             params={**base, "max_depth": 6, "eta": 0.08, "subsample": 0.85,
                     "colsample_bytree": 0.85, "min_child_weight": 3,
                     "scale_pos_weight": balanced_weight},
             num_boost_round=400),
        dict(name="shallow_trees",
             params={**base, "max_depth": 3, "eta": 0.08, "subsample": 0.85,
                     "colsample_bytree": 0.85, "min_child_weight": 3,
                     "scale_pos_weight": balanced_weight},
             num_boost_round=400),
        dict(name="deep_slow_learn",
             params={**base, "max_depth": 9, "eta": 0.03, "subsample": 0.8,
                     "colsample_bytree": 0.8, "min_child_weight": 5,
                     "scale_pos_weight": balanced_weight},
             num_boost_round=800),
        dict(name="no_imbalance_handling",
             params={**base, "max_depth": 6, "eta": 0.08, "subsample": 0.85,
                     "colsample_bytree": 0.85, "min_child_weight": 3,
                     "scale_pos_weight": 1},
             num_boost_round=400),
        dict(name="oversample_minority",
             params={**base, "max_depth": 6, "eta": 0.08, "subsample": 0.85,
                     "colsample_bytree": 0.85, "min_child_weight": 3,
                     "scale_pos_weight": 1},
             num_boost_round=400, oversample=True),
    ]

    results = []
    for cfg in configs:
        print(f"Running: {cfg['name']}...")
        res = run_config(
            cfg["name"], cfg["params"], cfg["num_boost_round"],
            X_train, y_train, X_val, y_val, X_test, y_test, feature_columns,
            oversample=cfg.get("oversample", False),
        )
        results.append(res)
        print(f"  val_f1={res['val_f1']:.4f}  test_f1={res['test_f1']:.4f}  "
              f"test_precision={res['test_precision']:.4f}  test_recall={res['test_recall']:.4f}")

    winner = max(results, key=lambda r: r["val_f1"])
    print(f"\nWinner (by val F1): {winner['name']}")

    with open(f"{MODEL_DIR}/experiments.json", "w") as f:
        json.dump({"results": results, "winner": winner["name"]}, f, indent=2)

    lines = [
        "# Experiment log\n",
        "Same split and feature set throughout; only the model config and",
        "imbalance-handling strategy vary. Winner picked by validation F1.\n",
        "| config | max_depth | eta | imbalance handling | val F1 | test acc | test prec | test recall | test F1 | test AUC |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        p = r["params"]
        imb = "oversampling" if r["oversampled"] else (
            "scale_pos_weight" if p.get("scale_pos_weight", 1) != 1 else "none"
        )
        marker = " (winner)" if r["name"] == winner["name"] else ""
        lines.append(
            f"| {r['name']}{marker} | {p['max_depth']} | {p['eta']} | {imb} | "
            f"{r['val_f1']:.4f} | {r['test_accuracy']:.4f} | {r['test_precision']:.4f} | "
            f"{r['test_recall']:.4f} | {r['test_f1']:.4f} | {r['test_roc_auc']:.4f} |"
        )
    lines.append(f"\nSelected config: `{winner['name']}` (used by `model/train.py`).")

    with open(os.environ.get("EXPERIMENTS_MD", "EXPERIMENTS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Wrote model/experiments.json and EXPERIMENTS.md")


if __name__ == "__main__":
    main()

"""Feature engineering for the fraud model.

Split-dependent statistics are passed in by the caller rather than
computed internally, so they can be fit on the training split only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

CATEGORY_LIST = [
    "electronics", "grocery", "travel", "software", "gambling",
    "jewelry", "utilities", "restaurant", "apparel", "crypto_exchange",
]
PAYMENT_METHOD_LIST = ["credit_card", "debit_card", "ach", "wire", "digital_wallet"]

FIELD_NAMES = [
    "vendor_name", "account_number", "transaction_date", "amount", "currency",
    "merchant_category", "payment_method", "billing_country", "shipping_country",
]


def load_validated_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.json_normalize(rows)


def raw_to_frame(docs: list[dict]) -> pd.DataFrame:
    """Flatten the nested FinancialDocument dicts into a working frame."""
    records = []
    for d in docs:
        rec = {
            "document_id": d["document_id"],
            "template": d["template"],
            "account_age_days": d["account_age_days"],
            "documents_last_24h": d["documents_last_24h"],
            "hour_of_day": d["hour_of_day"],
            "is_fraud": d.get("is_fraud"),
        }
        for field in FIELD_NAMES:
            rec[f"{field}_value"] = d[field]["value"]
            rec[f"{field}_confidence"] = d[field]["confidence"]
        rec["bounding_boxes"] = d["bounding_boxes"]
        records.append(rec)
    return pd.DataFrame.from_records(records)


def build_features(df: pd.DataFrame, category_stats: pd.DataFrame | None = None):
    """Build the feature matrix.

    category_stats (mean/std amount per category) is computed from df when
    None and returned, so the caller can reuse the train-fit stats on
    val/test rather than recomputing per split.
    """
    out = pd.DataFrame(index=df.index)

    if category_stats is None:
        category_stats = (
            df.groupby("merchant_category_value")["amount_value"]
            .agg(["mean", "std"]).rename(columns={"mean": "cat_mean", "std": "cat_std"})
        )
        category_stats["cat_std"] = category_stats["cat_std"].fillna(1.0).clip(lower=1.0)

    joined = df.join(category_stats, on="merchant_category_value")
    out["amount"] = df["amount_value"]
    out["amount_log"] = np.log1p(df["amount_value"])
    out["amount_category_zscore"] = (
        (joined["amount_value"] - joined["cat_mean"]) / joined["cat_std"]
    )

    out["account_age_days"] = df["account_age_days"]
    out["is_new_account"] = (df["account_age_days"] < 14).astype(int)
    out["is_recent_account"] = (df["account_age_days"] < 45).astype(int)

    out["documents_last_24h"] = df["documents_last_24h"]
    out["high_velocity"] = (df["documents_last_24h"] >= 6).astype(int)

    out["hour_of_day"] = df["hour_of_day"]
    out["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    out["is_late_night"] = (df["hour_of_day"] <= 4).astype(int)

    out["country_mismatch"] = (
        df["billing_country_value"] != df["shipping_country_value"]
    ).astype(int)

    for cat in CATEGORY_LIST:
        out[f"category_{cat}"] = (df["merchant_category_value"] == cat).astype(int)
    for pm in PAYMENT_METHOD_LIST:
        out[f"payment_{pm}"] = (df["payment_method_value"] == pm).astype(int)

    conf_cols = [f"{f}_confidence" for f in FIELD_NAMES]
    out["min_field_confidence"] = df[conf_cols].min(axis=1)
    out["mean_field_confidence"] = df[conf_cols].mean(axis=1)
    out["low_confidence_field_count"] = (df[conf_cols] < 0.6).sum(axis=1)
    out["account_number_confidence"] = df["account_number_confidence"]
    out["amount_confidence"] = df["amount_confidence"]
    out["shipping_country_confidence"] = df["shipping_country_confidence"]

    return out, category_stats


def get_feature_columns(sample_df: pd.DataFrame) -> list[str]:
    feats, _ = build_features(sample_df.head(50))
    return list(feats.columns)

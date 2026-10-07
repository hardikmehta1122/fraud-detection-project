"""Broker-independent scoring used by the Faust worker and the API.

Kept separate from the Faust app so it can be tested without a broker. Uses
the same feature engineering and model artifacts as the batch pipeline.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import pandas as pd
import xgboost as xgb
from pydantic import ValidationError

from pipeline.features import build_features, raw_to_frame
from pipeline.schemas import FinancialDocument

MODEL_DIR = os.environ.get("MODEL_DIR", "model")


class ScoringArtifacts:
    """Loads model + feature config once and holds them for reuse."""

    def __init__(self, model_dir: str = MODEL_DIR):
        with open(os.path.join(model_dir, "feature_columns.json")) as f:
            self.feature_columns = json.load(f)
        with open(os.path.join(model_dir, "metrics.json")) as f:
            self.threshold = json.load(f)["threshold"]
        self.category_stats = pd.read_json(
            os.path.join(model_dir, "category_stats.json"), orient="index"
        )
        self.booster = xgb.Booster()
        self.booster.load_model(os.path.join(model_dir, "fraud_model.json"))


@lru_cache(maxsize=1)
def get_artifacts() -> ScoringArtifacts:
    return ScoringArtifacts()


class EventValidationError(Exception):
    """Raised when an incoming event fails schema validation at ingestion."""


def score_event(event: dict, artifacts: ScoringArtifacts | None = None) -> dict:
    """Validate and score a single transaction event.

    Raises EventValidationError on malformed input so the caller can route
    it to a dead-letter topic rather than scoring it.
    """
    artifacts = artifacts or get_artifacts()

    try:
        FinancialDocument.model_validate(event)
    except ValidationError as e:
        raise EventValidationError(str(e.errors()[0]["msg"])) from e

    frame = raw_to_frame([event])
    X, _ = build_features(frame, category_stats=artifacts.category_stats)
    X = X[artifacts.feature_columns].replace([float("inf"), float("-inf")], float("nan"))

    dmat = xgb.DMatrix(X, feature_names=artifacts.feature_columns)
    risk = float(artifacts.booster.predict(dmat)[0])

    return {
        "document_id": event["document_id"],
        "vendor_name": event["vendor_name"]["value"],
        "amount": event["amount"]["value"],
        "currency": event["currency"]["value"],
        "merchant_category": event["merchant_category"]["value"],
        "risk_score": round(risk, 6),
        "is_fraud_pred": int(risk >= artifacts.threshold),
        "threshold": artifacts.threshold,
    }


# Defaults for fields the playground does not expose.
_DEFAULT_CONF = 0.95
_FIELD_DEFAULTS = {
    "vendor_name": "Playground Vendor",
    "account_number": "000111222333",
    "transaction_date": "2025-01-01",
}
_BBOX_STUB = {
    f: {"x": 0.05, "y": 0.05, "w": 0.3, "h": 0.05}
    for f in [
        "vendor_name", "account_number", "transaction_date", "amount", "currency",
        "merchant_category", "payment_method", "billing_country", "shipping_country",
    ]
}


def fields_to_event(fields: dict) -> dict:
    """Build a schema-valid event from the subset of fields the playground
    exposes, filling the rest with defaults."""
    conf = fields.get("confidence", {})

    def f(name, value):
        return {"value": value, "confidence": conf.get(name, _DEFAULT_CONF)}

    return {
        "document_id": fields.get("document_id", "PLAYGROUND"),
        "template": "template_a",
        "vendor_name": f("vendor_name", fields.get("vendor_name", _FIELD_DEFAULTS["vendor_name"])),
        "account_number": f("account_number", _FIELD_DEFAULTS["account_number"]),
        "transaction_date": f("transaction_date", _FIELD_DEFAULTS["transaction_date"]),
        "amount": f("amount", float(fields["amount"])),
        "currency": f("currency", fields["currency"]),
        "merchant_category": f("merchant_category", fields["merchant_category"]),
        "payment_method": f("payment_method", fields["payment_method"]),
        "billing_country": f("billing_country", fields["billing_country"]),
        "shipping_country": f("shipping_country", fields["shipping_country"]),
        "account_age_days": int(fields["account_age_days"]),
        "documents_last_24h": int(fields["documents_last_24h"]),
        "hour_of_day": int(fields["hour_of_day"]),
        "bounding_boxes": _BBOX_STUB,
    }


def score_fields(fields: dict, artifacts: ScoringArtifacts | None = None, top_k: int = 6) -> dict:
    """Score a transaction from field inputs, returning the risk score,
    prediction, and the top signed feature contributions."""
    artifacts = artifacts or get_artifacts()
    event = fields_to_event(fields)

    try:
        FinancialDocument.model_validate(event)
    except ValidationError as e:
        raise EventValidationError(str(e.errors()[0]["msg"])) from e

    frame = raw_to_frame([event])
    X, _ = build_features(frame, category_stats=artifacts.category_stats)
    X = X[artifacts.feature_columns].replace([float("inf"), float("-inf")], float("nan"))

    dmat = xgb.DMatrix(X, feature_names=artifacts.feature_columns)
    risk = float(artifacts.booster.predict(dmat)[0])
    contribs = artifacts.booster.predict(dmat, pred_contribs=True)[0][:-1]  # drop bias

    order = sorted(range(len(contribs)), key=lambda j: abs(contribs[j]), reverse=True)[:top_k]
    top = [
        {"feature": artifacts.feature_columns[j], "contribution": round(float(contribs[j]), 5)}
        for j in order
    ]

    return {
        "risk_score": round(risk, 6),
        "is_fraud_pred": int(risk >= artifacts.threshold),
        "threshold": artifacts.threshold,
        "top_contributions": top,
    }

"""Unit tests for the scorer (no broker required)."""

import json
import os

import pytest

from streaming.scorer import EventValidationError, get_artifacts, score_event


def _load_events(n=200):
    path = os.environ.get("VALIDATED_PATH", "data/validated.jsonl")
    out = []
    with open(path) as f:
        for i, line in enumerate(f):
            out.append(json.loads(line))
            if i + 1 >= n:
                break
    return out


@pytest.fixture(scope="module")
def events():
    return _load_events()


def test_score_event_shape(events):
    r = score_event(events[0])
    assert set(["document_id", "risk_score", "is_fraud_pred", "threshold"]).issubset(r)
    assert 0.0 <= r["risk_score"] <= 1.0
    assert r["is_fraud_pred"] in (0, 1)


def test_prediction_follows_threshold(events):
    a = get_artifacts()
    for e in events[:50]:
        r = score_event(e)
        assert r["is_fraud_pred"] == int(r["risk_score"] >= a.threshold)


def test_stream_matches_batch(events):
    """Stream scoring must equal the batch model path (no drift)."""
    import numpy as np
    import xgboost as xgb
    from pipeline.features import build_features, raw_to_frame

    a = get_artifacts()
    frame = raw_to_frame(events)
    X, _ = build_features(frame, category_stats=a.category_stats)
    X = X[a.feature_columns].replace([np.inf, -np.inf], np.nan)
    batch = a.booster.predict(xgb.DMatrix(X, feature_names=a.feature_columns))
    stream = np.array([score_event(e)["risk_score"] for e in events])
    assert float(np.max(np.abs(batch - stream))) < 1e-5


def test_malformed_event_rejected(events):
    bad = json.loads(json.dumps(events[0]))
    bad["currency"]["value"] = "ZZZ"
    with pytest.raises(EventValidationError):
        score_event(bad)


def test_negative_amount_rejected(events):
    bad = json.loads(json.dumps(events[0]))
    bad["amount"]["value"] = -5.0
    with pytest.raises(EventValidationError):
        score_event(bad)

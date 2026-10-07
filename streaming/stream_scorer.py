"""Faust worker: score transactions off Redpanda in real time.

Consumes transactions.raw and emits scored / flagged / deadletter topics,
exposing Prometheus metrics on METRICS_PORT.

    faust -A streaming.stream_scorer worker -l info
"""

from __future__ import annotations

import os
import time

import faust

from streaming import metrics
from streaming.scorer import EventValidationError, get_artifacts, score_event

BROKER = os.environ.get("REDPANDA_BROKER", "kafka://redpanda:9092")
METRICS_PORT = int(os.environ.get("METRICS_PORT", "9100"))

app = faust.App(
    "fraud-stream-scorer",
    broker=BROKER,
    value_serializer="json",
    consumer_auto_offset_reset="latest",
)

raw_topic = app.topic("transactions.raw")
scored_topic = app.topic("transactions.scored")
flagged_topic = app.topic("transactions.flagged")
deadletter_topic = app.topic("transactions.deadletter")


@app.agent(raw_topic)
async def score(stream):
    async for event in stream:
        start = time.perf_counter()
        try:
            result = score_event(event)
        except EventValidationError as exc:
            metrics.EVENTS_REJECTED.inc()
            await deadletter_topic.send(value={"event": event, "reason": str(exc)})
            continue

        metrics.SCORING_LATENCY.observe(time.perf_counter() - start)
        metrics.TRANSACTIONS_PROCESSED.inc()
        metrics.RISK_SCORE.observe(result["risk_score"])
        metrics.FRAUD_PREDICTIONS.labels(
            prediction="fraud" if result["is_fraud_pred"] else "legit"
        ).inc()

        await scored_topic.send(value=result)
        if result["is_fraud_pred"]:
            await flagged_topic.send(value=result)


@app.task
async def on_started():
    # Warm the model and expose metrics before the first message.
    artifacts = get_artifacts()
    metrics.MODEL_THRESHOLD.set(artifacts.threshold)
    metrics.start_metrics_server(METRICS_PORT)

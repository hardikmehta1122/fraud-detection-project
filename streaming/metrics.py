"""Prometheus metrics for the stream scorer."""

from prometheus_client import Counter, Gauge, Histogram, start_http_server

TRANSACTIONS_PROCESSED = Counter(
    "fraud_transactions_processed_total",
    "Total transaction events consumed and scored",
)

FRAUD_PREDICTIONS = Counter(
    "fraud_predictions_total",
    "Scored events by predicted class",
    ["prediction"],
)

EVENTS_REJECTED = Counter(
    "fraud_events_rejected_total",
    "Events rejected at ingestion for failing schema validation",
)

RISK_SCORE = Histogram(
    "fraud_risk_score",
    "Distribution of model risk scores on the stream",
    buckets=[0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)

SCORING_LATENCY = Histogram(
    "fraud_scoring_latency_seconds",
    "Per-event validate + feature-build + predict latency",
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)

MODEL_THRESHOLD = Gauge(
    "fraud_model_threshold",
    "Decision threshold the worker is applying",
)


def start_metrics_server(port: int = 9100) -> None:
    start_http_server(port)

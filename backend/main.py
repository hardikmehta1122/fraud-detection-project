"""FastAPI backend for the fraud-review dashboard.

Serves the review queue, document detail, analyst decisions, live scoring,
Prometheus metrics, and the static frontend.
"""

import json
import os
import random
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator

DB_PATH = Path(os.environ.get("DB_PATH", Path(__file__).parent / "fraud.db"))

app = FastAPI(title="Fraud Review Dashboard API")

API_REQUESTS = Counter(
    "api_requests_total", "API requests", ["method", "endpoint", "status"]
)
API_LATENCY = Histogram(
    "api_request_latency_seconds", "API request latency", ["endpoint"]
)
REVIEW_DECISIONS = Counter(
    "api_review_decisions_total", "Analyst review decisions", ["decision"]
)


@app.middleware("http")
async def record_metrics(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    endpoint = request.scope.get("route").path if request.scope.get("route") else request.url.path
    API_LATENCY.labels(endpoint=endpoint).observe(time.perf_counter() - start)
    API_REQUESTS.labels(
        method=request.method, endpoint=endpoint, status=response.status_code
    ).inc()
    return response


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


class ReviewRequest(BaseModel):
    decision: str
    analyst_note: str | None = None

    @field_validator("decision")
    @classmethod
    def valid_decision(cls, v: str) -> str:
        if v not in ("approved", "rejected"):
            raise ValueError("decision must be 'approved' or 'rejected'")
        return v


CATEGORIES = [
    "electronics", "grocery", "travel", "software", "gambling",
    "jewelry", "utilities", "restaurant", "apparel", "crypto_exchange",
]
PAYMENT_METHODS = ["credit_card", "debit_card", "ach", "wire", "digital_wallet"]
CURRENCIES = ["USD", "CAD", "EUR", "GBP", "AUD"]
COUNTRIES = ["US", "CA", "GB", "DE", "FR", "AU", "NG", "RU", "CN", "BR"]


class ScoreRequest(BaseModel):
    amount: float = Field(gt=0, le=1_000_000)
    currency: str
    merchant_category: str
    payment_method: str
    billing_country: str
    shipping_country: str
    account_age_days: int = Field(ge=0, le=4000)
    documents_last_24h: int = Field(ge=0, le=40)
    hour_of_day: int = Field(ge=0, le=23)

    @field_validator("currency")
    @classmethod
    def _cur(cls, v):
        if v not in CURRENCIES:
            raise ValueError(f"currency must be one of {CURRENCIES}")
        return v

    @field_validator("merchant_category")
    @classmethod
    def _cat(cls, v):
        if v not in CATEGORIES:
            raise ValueError(f"merchant_category must be one of {CATEGORIES}")
        return v

    @field_validator("payment_method")
    @classmethod
    def _pm(cls, v):
        if v not in PAYMENT_METHODS:
            raise ValueError(f"payment_method must be one of {PAYMENT_METHODS}")
        return v

    @field_validator("billing_country", "shipping_country")
    @classmethod
    def _ctry(cls, v):
        if v not in COUNTRIES:
            raise ValueError(f"country must be one of {COUNTRIES}")
        return v


@app.get("/api/score/options")
def score_options():
    """Enumerations the playground UI uses to build its inputs."""
    return {
        "categories": CATEGORIES,
        "payment_methods": PAYMENT_METHODS,
        "currencies": CURRENCIES,
        "countries": COUNTRIES,
    }


@app.post("/api/score")
def score(req: ScoreRequest):
    """Score user-supplied fields with the model."""
    from streaming.scorer import score_fields

    return score_fields(req.model_dump())


# Background consumer of the scored topic, populated only when the
# streaming stack is running (REDPANDA_BOOTSTRAP set).
_REDPANDA_BOOTSTRAP = os.environ.get("REDPANDA_BOOTSTRAP")
_SCORED_TOPIC = os.environ.get("SCORED_TOPIC", "transactions.scored")
_live_events: deque = deque(maxlen=50)
_consumer_started = False


def _consume_scored():
    try:
        from confluent_kafka import Consumer
    except ImportError:
        return
    consumer = Consumer({
        "bootstrap.servers": _REDPANDA_BOOTSTRAP,
        "group.id": "dashboard-live-feed",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([_SCORED_TOPIC])
    while True:
        msg = consumer.poll(1.0)
        if msg is None or msg.error():
            continue
        try:
            _live_events.appendleft(json.loads(msg.value()))
        except (ValueError, TypeError):
            pass


@app.on_event("startup")
def _start_consumer():
    global _consumer_started
    if _REDPANDA_BOOTSTRAP and not _consumer_started:
        threading.Thread(target=_consume_scored, daemon=True).start()
        _consumer_started = True


# When the streaming stack is absent, the feed shows a rolling sample of
# transactions scored by the model, so the deployed site can demonstrate it.
_DEMO_VENDORS = [
    "Northgate Supply Co", "Summit Electronics", "Prairie Grocer",
    "Vantage Travel", "Golden Aspen Jewelers", "Quantum Crypto Exchange",
    "Fairview Pharmacy", "Lakeside Bistro", "Alpine Outfitters",
    "Sterling Gambling Co", "Nimbus Cloud Services", "Copperfield Books",
]
_demo_events: deque = deque(maxlen=50)


def _make_demo_event() -> dict:
    from streaming.scorer import score_fields

    if random.random() < 0.25:
        fields = {
            "amount": round(random.uniform(800, 6000), 2),
            "currency": random.choice(CURRENCIES),
            "merchant_category": random.choice(["crypto_exchange", "gambling", "jewelry"]),
            "payment_method": random.choice(["wire", "digital_wallet"]),
            "billing_country": "US",
            "shipping_country": random.choice(["RU", "NG", "CN"]),
            "account_age_days": random.randint(1, 30),
            "documents_last_24h": random.randint(5, 15),
            "hour_of_day": random.randint(0, 4),
        }
    else:
        country = random.choice(["US", "CA", "GB"])
        fields = {
            "amount": round(random.uniform(10, 500), 2),
            "currency": random.choice(CURRENCIES),
            "merchant_category": random.choice(CATEGORIES),
            "payment_method": random.choice(["credit_card", "debit_card", "ach"]),
            "billing_country": country,
            "shipping_country": country,
            "account_age_days": random.randint(60, 2000),
            "documents_last_24h": random.randint(0, 3),
            "hour_of_day": random.randint(8, 22),
        }

    scored = score_fields(fields)
    return {
        "document_id": f"DOC{random.randint(0, 999_999_999):09d}",
        "vendor_name": random.choice(_DEMO_VENDORS),
        "amount": fields["amount"],
        "currency": fields["currency"],
        "merchant_category": fields["merchant_category"],
        "risk_score": scored["risk_score"],
        "is_fraud_pred": scored["is_fraud_pred"],
    }


@app.get("/api/stream/recent")
def stream_recent():
    if _REDPANDA_BOOTSTRAP:
        return {"enabled": True, "demo": False, "events": list(_live_events)}

    for _ in range(random.randint(1, 2)):
        _demo_events.appendleft(_make_demo_event())
    return {"enabled": True, "demo": True, "events": list(_demo_events)}


@app.get("/api/metrics")
def get_metrics():
    conn = get_conn()
    row = conn.execute("SELECT metrics_json FROM model_metrics WHERE id = 1").fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "metrics not found — run backend/build_serving_db.py")
    return json.loads(row["metrics_json"])


@app.get("/api/stats")
def get_stats():
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) c FROM documents").fetchone()["c"]
    pending_high_risk = conn.execute(
        "SELECT COUNT(*) c FROM documents WHERE status='pending' AND risk_score >= 0.5"
    ).fetchone()["c"]
    approved = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='approved'").fetchone()["c"]
    rejected = conn.execute("SELECT COUNT(*) c FROM documents WHERE status='rejected'").fetchone()["c"]
    conn.close()
    return {
        "total_documents": total,
        "pending_high_risk": pending_high_risk,
        "approved": approved,
        "rejected": rejected,
    }


@app.get("/api/transactions")
def list_transactions(
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
    status: str | None = Query(None, pattern="^(pending|approved|rejected)$"),
    min_risk: float = Query(0.0, ge=0.0, le=1.0),
    sort: str = Query("risk_desc", pattern="^(risk_desc|risk_asc|date_desc)$"),
):
    conn = get_conn()
    where = ["risk_score >= ?"]
    params: list = [min_risk]
    if status:
        where.append("status = ?")
        params.append(status)
    where_clause = " AND ".join(where)

    order = {
        "risk_desc": "risk_score DESC",
        "risk_asc": "risk_score ASC",
        "date_desc": "document_id DESC",
    }[sort]

    total = conn.execute(
        f"SELECT COUNT(*) c FROM documents WHERE {where_clause}", params
    ).fetchone()["c"]

    offset = (page - 1) * page_size
    rows = conn.execute(
        f"""SELECT document_id, template, fields_json, risk_score, status
            FROM documents WHERE {where_clause}
            ORDER BY {order} LIMIT ? OFFSET ?""",
        [*params, page_size, offset],
    ).fetchall()
    conn.close()

    items = []
    for r in rows:
        fields = json.loads(r["fields_json"])
        items.append({
            "document_id": r["document_id"],
            "vendor_name": fields["vendor_name"]["value"],
            "amount": fields["amount"]["value"],
            "currency": fields["currency"]["value"],
            "merchant_category": fields["merchant_category"]["value"],
            "risk_score": round(r["risk_score"], 4),
            "status": r["status"],
        })

    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
    }


@app.get("/api/transactions/{document_id}")
def get_transaction(document_id: str):
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "document not found")

    return {
        "document_id": row["document_id"],
        "template": row["template"],
        "fields": json.loads(row["fields_json"]),
        "bounding_boxes": json.loads(row["bounding_boxes_json"]),
        "account_age_days": row["account_age_days"],
        "documents_last_24h": row["documents_last_24h"],
        "hour_of_day": row["hour_of_day"],
        "risk_score": round(row["risk_score"], 4),
        "top_contributions": json.loads(row["contributions_json"]),
        "status": row["status"],
        "analyst_note": row["analyst_note"],
        "reviewed_at": row["reviewed_at"],
    }


@app.post("/api/transactions/{document_id}/review")
def review_transaction(document_id: str, review: ReviewRequest):
    conn = get_conn()
    row = conn.execute(
        "SELECT document_id FROM documents WHERE document_id = ?", (document_id,)
    ).fetchone()
    if not row:
        conn.close()
        raise HTTPException(404, "document not found")

    conn.execute(
        """UPDATE documents SET status = ?, analyst_note = ?, reviewed_at = ?
           WHERE document_id = ?""",
        (review.decision, review.analyst_note, datetime.now(timezone.utc).isoformat(), document_id),
    )
    conn.commit()
    conn.close()
    REVIEW_DECISIONS.labels(decision=review.decision).inc()
    return {"document_id": document_id, "status": review.decision}


# Explicit route so /metrics isn't shadowed by the static mount below.
@app.get("/metrics")
def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


frontend_dir = Path(__file__).parent.parent / "frontend"
app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

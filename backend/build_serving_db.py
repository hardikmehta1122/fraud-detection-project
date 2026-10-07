"""Score every validated document and write the SQLite serving table.

Pre-scoring keeps model inference off the dashboard's request path.
"""

import json
import os
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline.features import FIELD_NAMES, build_features, raw_to_frame

MODEL_DIR = os.environ.get("MODEL_DIR", "model")
DATA_PATH = os.environ.get("VALIDATED_PATH", "data/validated.jsonl")
DB_PATH = os.environ.get("DB_PATH", "backend/fraud.db")


def main():
    print("Loading validated documents...")
    docs = []
    with open(DATA_PATH) as f:
        for line in f:
            docs.append(json.loads(line))
    df = raw_to_frame(docs)

    with open(f"{MODEL_DIR}/feature_columns.json") as f:
        feature_columns = json.load(f)
    with open(f"{MODEL_DIR}/metrics.json") as f:
        metrics = json.load(f)
    category_stats = pd.read_json(f"{MODEL_DIR}/category_stats.json", orient="index")

    print("Building features + scoring with trained model...")
    X, _ = build_features(df, category_stats=category_stats)
    X = X[feature_columns]

    booster = xgb.Booster()
    booster.load_model(f"{MODEL_DIR}/fraud_model.json")
    dmat = xgb.DMatrix(X, feature_names=feature_columns)

    risk_scores = booster.predict(dmat)
    contribs = booster.predict(dmat, pred_contribs=True)  # last column is bias

    print("Writing SQLite serving DB...")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute("""
        CREATE TABLE documents (
            document_id TEXT PRIMARY KEY,
            template TEXT,
            fields_json TEXT,
            bounding_boxes_json TEXT,
            account_age_days INTEGER,
            documents_last_24h INTEGER,
            hour_of_day INTEGER,
            is_fraud_actual INTEGER,
            risk_score REAL,
            contributions_json TEXT,
            status TEXT DEFAULT 'pending',
            analyst_note TEXT,
            reviewed_at TEXT
        )
    """)
    cur.execute("CREATE INDEX idx_risk ON documents(risk_score)")
    cur.execute("CREATE INDEX idx_status ON documents(status)")

    rows_to_insert = []
    for i, doc in enumerate(docs):
        fields = {fn: doc[fn] for fn in FIELD_NAMES}
        row_contribs = contribs[i][:-1]  # drop bias term
        top_idx = np.argsort(-np.abs(row_contribs))[:5]
        top_contribs = [
            {"feature": feature_columns[j], "contribution": round(float(row_contribs[j]), 5)}
            for j in top_idx
        ]
        rows_to_insert.append((
            doc["document_id"], doc["template"], json.dumps(fields),
            json.dumps(doc["bounding_boxes"]), doc["account_age_days"],
            doc["documents_last_24h"], doc["hour_of_day"], doc["is_fraud"],
            float(risk_scores[i]), json.dumps(top_contribs),
        ))

    cur.executemany(
        """INSERT INTO documents
           (document_id, template, fields_json, bounding_boxes_json,
            account_age_days, documents_last_24h, hour_of_day, is_fraud_actual,
            risk_score, contributions_json)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        rows_to_insert,
    )

    cur.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            metrics_json TEXT
        )
    """)
    cur.execute("DELETE FROM model_metrics")
    cur.execute("INSERT INTO model_metrics (id, metrics_json) VALUES (1, ?)", (json.dumps(metrics),))

    conn.commit()
    conn.close()
    print(f"Wrote {len(rows_to_insert)} scored documents to {DB_PATH}")


if __name__ == "__main__":
    main()

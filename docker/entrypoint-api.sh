#!/usr/bin/env bash
# API startup. In DEMO_MODE, build a small serving DB with the committed
# model before serving, so a hosted deploy needs no large DB or retraining.
set -e

DB="${DB_PATH:-backend/fraud.db}"
mkdir -p "$(dirname "$DB")"

if [ ! -f "$DB" ] && [ "${DEMO_MODE:-0}" = "1" ]; then
  echo "Building ${DEMO_N:-5000}-row demo serving DB..."
  GEN_TARGET="${DEMO_N:-5000}" python data/generate_data.py
  python -m pipeline.ingest
  python -m backend.build_serving_db
fi

exec uvicorn backend.main:app --host 0.0.0.0 --port "${PORT:-8000}"

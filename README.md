# Docket — Fraud Detection Pipeline

A full-stack fraud detection system: structured transaction extraction →
Pydantic-validated ingestion → feature engineering → XGBoost classifier →
FastAPI backend → an analyst review dashboard, with an optional real-time
scoring layer on Redpanda and Faust.

The dataset is synthetic (`data/generate_data.py`); see [RESULTS.md](RESULTS.md).

## Architecture

**Batch pipeline** — trains the model and builds the review dashboard:

```
data/generate_data.py        synthetic extractor output (~1.2% malformed)
pipeline/schemas.py          Pydantic schema; validation boundary
pipeline/ingest.py           validate raw -> validated.jsonl (before any split)
pipeline/features.py         feature engineering (split-fit statistics)
model/train.py               split -> XGBoost -> threshold on val -> score on test
backend/build_serving_db.py  score every record -> SQLite serving table
backend/main.py              FastAPI: queue, detail, review, scoring, /metrics
frontend/                    dashboard: queue, playground, live feed, model
```

**Streaming layer** — scores a live feed with the same model:

```
producer -> Redpanda (transactions.raw) -> Faust worker
                                             -> transactions.scored / flagged / deadletter
                                             -> Prometheus metrics -> Grafana
```

The streaming worker and the batch pipeline share `streaming/scorer.py`, so a
transaction gets the same score either way (verified to ~1e-6 in the tests).

## Run the pipeline

```bash
pip install -r requirements.txt

python data/generate_data.py          # generate synthetic data
python -m pipeline.ingest             # validate
python -m model.train                 # train + evaluate
python -m backend.build_serving_db    # build the serving DB
uvicorn backend.main:app --reload     # http://127.0.0.1:8000
```

`model/` ships pre-trained artifacts, so steps 4–5 alone will serve the
dashboard. Generated data and the serving DB are not checked in.

Optional analysis:

```bash
python -m model.experiments                      # config comparison -> EXPERIMENTS.md
python -m model.leakage_demo                      # pre-split statistics check
python -m model.validation_timing_leakage_demo   # unvalidated-training check
```

## Run the full stack (Docker)

```bash
docker compose up --build
```

| service | URL | description |
|---|---|---|
| Dashboard / API | http://localhost:8000 | review UI + REST |
| Redpanda Console | http://localhost:8080 | topics, messages, consumer groups |
| Prometheus | http://localhost:9090 | metrics + targets |
| Grafana | http://localhost:3000 | Fraud Stream Scorer dashboard (admin/admin) |

`pipeline-init` builds the data, model and serving DB into shared volumes
before the other services start.

## Deploy

`render.yaml` deploys the dashboard and scoring playground as a single web
service (push to GitHub, then render.com → New → Blueprint → select the repo).
On startup it builds a small serving DB with the committed model, so no large
database or retraining is needed.

The streaming stack is not part of the hosted deploy; it runs locally with
`docker compose up`. On the hosted deploy the Live feed shows a rolling sample
of transactions scored by the model, labeled as a preview.

Run the API image directly:

```bash
docker build -f docker/Dockerfile.api -t fraud-api .
docker run -p 8000:8000 -e DEMO_MODE=1 fraud-api
```

## CI/CD

`Jenkinsfile`: install → ruff lint → pytest → build pipeline artifacts →
leakage/experiment checks → build images → compose smoke test → push to GHCR
on `main`.

## Dataset

The data is synthetic. Fraud and legitimate records are drawn from distinct
feature distributions (account age, country match, velocity, hour,
category-relative amount, extraction confidence), which makes the classes more
separable than real transaction data; metrics should be read in that light.
The document layouts in the dashboard are generated mock-ups, not OCR output.

## Layout

```
data/            synthetic data generation
pipeline/        schema, ingestion, feature engineering
model/           training, experiment comparison, leakage checks, artifacts
backend/         FastAPI app, serving DB builder
frontend/        dashboard (HTML/CSS/JS)
streaming/       Faust worker, producer, scorer, metrics
docker/          Dockerfiles, Prometheus + Grafana config
tests/           pytest suite
docker-compose.yml
Jenkinsfile
render.yaml
```

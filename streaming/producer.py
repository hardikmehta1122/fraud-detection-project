"""Publish validated transactions to the transactions.raw topic.

Reads data/validated.jsonl and produces each record at a configurable rate,
keyed by document_id.

    python -m streaming.producer --rate 50 --loop
"""

from __future__ import annotations

import argparse
import json
import os
import time

from confluent_kafka import Producer

BROKER = os.environ.get("REDPANDA_BOOTSTRAP", "redpanda:9092")
TOPIC = os.environ.get("RAW_TOPIC", "transactions.raw")
DATA_PATH = os.environ.get("VALIDATED_PATH", "data/validated.jsonl")


def build_producer() -> Producer:
    return Producer({
        "bootstrap.servers": BROKER,
        "client.id": "fraud-tx-producer",
        "linger.ms": 50,
        "compression.type": "snappy",
    })


def iter_events(path: str):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                # Drop the label; a live feed would not carry it.
                event = json.loads(line)
                event.pop("is_fraud", None)
                yield event


def run(rate: float, loop: bool, limit: int | None):
    producer = build_producer()
    interval = 1.0 / rate if rate > 0 else 0.0
    sent = 0
    while True:
        for event in iter_events(DATA_PATH):
            producer.produce(
                TOPIC,
                key=event["document_id"],
                value=json.dumps(event),
            )
            producer.poll(0)
            sent += 1
            if sent % 1000 == 0:
                print(f"produced {sent} events", flush=True)
            if limit and sent >= limit:
                producer.flush()
                print(f"done, produced {sent} events")
                return
            if interval:
                time.sleep(interval)
        if not loop:
            break
    producer.flush()
    print(f"done, produced {sent} events")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=50, help="events per second (0 = as fast as possible)")
    ap.add_argument("--loop", action="store_true", help="replay the file forever")
    ap.add_argument("--limit", type=int, default=None, help="stop after N events")
    args = ap.parse_args()
    run(args.rate, args.loop, args.limit)


if __name__ == "__main__":
    main()

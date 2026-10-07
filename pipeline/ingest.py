"""Validate raw extracted records against the schema, before any split.

Rejected records are counted by reason; accepted ones are written out as
the validated dataset the rest of the pipeline consumes.
"""

import json
import logging
import os
from dataclasses import dataclass, field

from pydantic import ValidationError

from pipeline.schemas import FinancialDocument

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("ingest")


@dataclass
class IngestReport:
    total: int = 0
    accepted: int = 0
    rejected: int = 0
    rejection_reasons: dict = field(default_factory=dict)

    def log_rejection(self, reason: str):
        self.rejected += 1
        key = reason.split("\n")[0][:80]
        self.rejection_reasons[key] = self.rejection_reasons.get(key, 0) + 1


def ingest_jsonl(path: str) -> tuple[list[FinancialDocument], IngestReport]:
    report = IngestReport()
    accepted: list[FinancialDocument] = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            report.total += 1
            raw = json.loads(line)
            try:
                doc = FinancialDocument.model_validate(raw)
                accepted.append(doc)
                report.accepted += 1
            except ValidationError as e:
                report.log_rejection(str(e.errors()[0]["msg"]))

    return accepted, report


def main():
    docs, report = ingest_jsonl(os.environ.get("RAW_PATH", "data/raw_extracted.jsonl"))
    logger.info(f"Ingested {report.total} raw records")
    logger.info(f"Accepted: {report.accepted}  Rejected: {report.rejected}")
    for reason, count in sorted(report.rejection_reasons.items(), key=lambda x: -x[1]):
        logger.info(f"  rejected ({count}): {reason}")

    out_path = os.environ.get("VALIDATED_PATH", "data/validated.jsonl")
    with open(out_path, "w") as f:
        f.writelines(doc.model_dump_json() + "\n" for doc in docs)
    logger.info(f"Wrote {len(docs)} validated records to {out_path}")


if __name__ == "__main__":
    main()

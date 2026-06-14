"""Consumes the `aml_transactions` stream and scores each transaction via the
PRAGMA-G API (PLAN.md Week 7).

For every message: maps the IBM-AML transaction row to a `TransactionRequest`,
calls `/score`, and writes `(account_id, score, decision, scored_at)` to the
`scored_transactions` Postgres table. Every `monitoring.drift_check_interval`
messages (`configs/pragma_s.yaml`), runs an Evidently drift check over the
buffered transaction features and updates the `pragma_g_drift_detected` gauge.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import yaml
from kafka import KafkaConsumer
from loguru import logger
from sqlalchemy import create_engine, text

from src.monitoring.drift import DRIFT_COLUMNS, build_reference_dataset, run_drift_check

TOPIC = "aml_transactions"
CONFIG_PATH = Path("configs/pragma_s.yaml")


def row_to_request(row: dict[str, Any]) -> dict[str, Any]:
    """Maps one IBM-AML transaction row (`stream_producer` JSON) to a
    `TransactionRequest` payload for `POST /score`."""
    return {
        "account_id": str(row["Account"]),
        "events": [
            {
                "type": "wire",
                "amount": float(row["Amount Paid"]),
                "currency": row["Payment Currency"],
                "amount_received": float(row["Amount Received"]),
                "receiving_currency": row["Receiving Currency"],
                "payment_format": row["Payment Format"],
                "counterparty_account": str(row["Account.1"]),
                "counterparty_bank": str(row["To Bank"]),
                "timestamp": str(row["Timestamp"]),
            }
        ],
    }


def main() -> None:
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    api_url = os.environ.get("PRAGMA_G_API_URL", "http://localhost:8000")
    postgres_url = os.environ.get(
        "POSTGRES_URL", "postgresql://pragma:pragma@localhost:5432/pragma_g"
    )

    with open(CONFIG_PATH) as f:
        drift_check_interval = yaml.safe_load(f)["monitoring"]["drift_check_interval"]

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=bootstrap_servers,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="latest",
    )
    engine = create_engine(postgres_url)
    client = httpx.Client(base_url=api_url, timeout=10.0)
    reference = build_reference_dataset()
    buffer: list[dict[str, Any]] = []

    logger.info(f"Consuming '{TOPIC}' from {bootstrap_servers}, scoring via {api_url}")
    for n, message in enumerate(consumer, start=1):
        row = message.value
        try:
            response = client.post("/score", json=row_to_request(row))
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            logger.error(f"Scoring error for account {row.get('Account')}: {e}")
            continue

        with engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO scored_transactions "
                    "(account_id, score, decision, scored_at) "
                    "VALUES (:account_id, :score, :decision, now())"
                ),
                {
                    "account_id": result["account_id"],
                    "score": result["score"],
                    "decision": result["decision"],
                },
            )

        buffer.append({col: row[col] for col in DRIFT_COLUMNS})
        if n % drift_check_interval == 0:
            drift = run_drift_check(reference, pd.DataFrame(buffer))
            logger.info(f"Drift check at n={n}: {drift}")
            buffer.clear()


if __name__ == "__main__":
    main()

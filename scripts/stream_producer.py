"""Replays IBM AML transactions as a live Kafka/Redpanda stream (PLAN.md Week 7).

Reads `data/HI-Small_Trans.csv` (falls back to synthetic transactions if the
dataset hasn't been downloaded), sorts chronologically, and publishes one JSON
message per transaction to the `aml_transactions` topic at ~100 tx/s.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from loguru import logger

from src.training.dataset import make_synthetic_transactions_df

DATA_PATH = Path("data/HI-Small_Trans.csv")
TOPIC = "aml_transactions"
SEND_INTERVAL_SECONDS = 0.01  # ~100 tx/s


def load_transactions() -> pd.DataFrame:
    if DATA_PATH.exists():
        logger.info(f"Loading transactions from {DATA_PATH}")
        df = pd.read_csv(DATA_PATH)
    else:
        logger.warning(f"{DATA_PATH} not found, streaming synthetic transactions")
        df = make_synthetic_transactions_df(n=10000)
    return df.sort_values("Timestamp").reset_index(drop=True)


def main() -> None:
    bootstrap_servers = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
    )

    df = load_transactions()
    logger.info(f"Streaming {len(df)} transactions to '{TOPIC}' ({bootstrap_servers})")

    for _, row in df.iterrows():
        producer.send(TOPIC, value=row.to_dict())
        time.sleep(SEND_INTERVAL_SECONDS)

    producer.flush()
    logger.info("Done streaming.")


if __name__ == "__main__":
    main()

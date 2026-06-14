"""A/B comparison of PRAGMA-G model versions on synthetic traffic (PLAN.md Week 9).

Builds one `/score`-style request per account from synthetic IBM-AML-shaped
data, scores each account with both the `v1` (Production) and `v2` (Staging)
models loaded by `ModelLoader`, and reports PR-AUC/ROC-AUC for each — the
offline analogue of comparing live `/score?model=v1` vs `/score?model=v2`
traffic.

Usage:
    python -m scripts.compare_models --config configs/pragma_s.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.api.model_loader import ModelLoader
from src.api.schemas import EventRecord, TransactionRequest
from src.training.dataset import make_synthetic_transactions_df
from src.training.metrics import compute_metrics


def build_requests(df: pd.DataFrame) -> tuple[list[TransactionRequest], np.ndarray]:
    """One `TransactionRequest` per sender account; label = any of its
    transactions flagged `Is Laundering`.
    """
    requests = []
    labels = []
    for account_id, group in df.groupby("Account"):
        group = group.sort_values("Timestamp")
        events = [
            EventRecord(
                type="wire",
                amount=row["Amount Paid"],
                currency=row["Payment Currency"],
                amount_received=row["Amount Received"],
                receiving_currency=row["Receiving Currency"],
                payment_format=row["Payment Format"],
                counterparty_account=row["Account.1"],
                timestamp=row["Timestamp"].to_pydatetime(),
            )
            for _, row in group.iterrows()
        ]
        requests.append(TransactionRequest(account_id=str(account_id), events=events))
        labels.append(int(group["Is Laundering"].max()))
    return requests, np.array(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B-compare PRAGMA-G model versions")
    parser.add_argument("--config", type=Path, default=Path("configs/pragma_s.yaml"))
    parser.add_argument("--n", type=int, default=1000, help="Number of synthetic transactions")
    parser.add_argument("--n-accounts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    loader = ModelLoader(config_path=args.config)
    loader.load()

    df = make_synthetic_transactions_df(n=args.n, n_accounts=args.n_accounts, seed=args.seed)
    requests, labels = build_requests(df)

    for version in ("v1", "v2"):
        scores = np.array([loader.score(r, model=version).score for r in requests])
        metrics = compute_metrics(labels, scores)
        model_version = loader.models[version].version
        print(f"{version} ({model_version}): {metrics}")


if __name__ == "__main__":
    main()

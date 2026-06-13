"""XGBoost baseline on hand-crafted transaction features (PLAN.md Week 5).

This is "the production baseline" PRAGMA underperforms on AML in the paper:
per-transaction features (the transaction's own edge features plus
aggregate stats of its source/destination accounts), with no temporal
sequence modelling and no learned graph representation.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from torch_geometric.data import Data

from src.training.metrics import compute_metrics


def build_account_features(
    transactions_df: pd.DataFrame, account_to_idx: dict[str, int]
) -> np.ndarray:
    """Per-account hand-crafted features: `(N, 6)`.

    `[sent_count, sent_mean_amount_paid, sent_mean_amount_received,
    received_count, received_mean_amount_paid, received_mean_amount_received]`.
    """
    n_nodes = len(account_to_idx)
    features = np.zeros((n_nodes, 6), dtype=np.float64)

    sent = transactions_df.groupby("Account").agg(
        count=("Amount Paid", "size"),
        mean_paid=("Amount Paid", "mean"),
        mean_received=("Amount Received", "mean"),
    )
    received = transactions_df.groupby("Account.1").agg(
        count=("Amount Paid", "size"),
        mean_paid=("Amount Paid", "mean"),
        mean_received=("Amount Received", "mean"),
    )

    for account, idx in account_to_idx.items():
        if account in sent.index:
            row = sent.loc[account]
            features[idx, 0:3] = [row["count"], row["mean_paid"], row["mean_received"]]
        if account in received.index:
            row = received.loc[account]
            features[idx, 3:6] = [row["count"], row["mean_paid"], row["mean_received"]]

    return features


def build_edge_features_with_accounts(transactions_df: pd.DataFrame, data: Data) -> np.ndarray:
    """Per-transaction features: `(E, edge_attr_dim + 2 * 6)`.

    Concatenates `data.edge_attr` (amount/time/format/currency) with the
    hand-crafted source- and destination-account features.
    """
    account_features = build_account_features(transactions_df, data.account_to_idx)
    edge_attr = data.edge_attr.numpy()
    src, dst = data.edge_index[0].numpy(), data.edge_index[1].numpy()
    return np.concatenate([edge_attr, account_features[src], account_features[dst]], axis=1)


def run_xgboost_baseline(
    transactions_df: pd.DataFrame, data: Data, pos_weight: float = 19.0, seed: int = 42
) -> dict[str, Any]:
    """Trains an XGBoost classifier on hand-crafted per-transaction features.

    Uses the same edge-level labels/splits as `PRAGMAGClassifier` so the
    PR-AUC numbers are directly comparable. Returns `{"test": metrics_dict,
    "model": fitted XGBClassifier}`.
    """
    features = build_edge_features_with_accounts(transactions_df, data)
    labels = data.y.numpy()
    train_mask, test_mask = data.train_mask.numpy(), data.test_mask.numpy()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=pos_weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(features[train_mask], labels[train_mask])

    scores = model.predict_proba(features[test_mask])[:, 1]
    metrics = compute_metrics(labels[test_mask], scores)
    return {"test": metrics, "model": model}

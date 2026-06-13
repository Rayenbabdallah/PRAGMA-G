"""Transaction graph construction for the PRAGMA-G graph extension (§4.1).

Builds a directed account-to-account graph from IBM AML transactions:
nodes are accounts, edges are transactions (From Account -> To Account),
edge features are `[amount_norm, time_delta_log, payment_format_id,
currency_id]`, and the 60/20/20 train/val/test split is by transaction
timestamp (never random — prevents leakage).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch_geometric.data import Data

from src.tokenizer.tokenizer import log_time_transform
from src.tokenizer.vocab import Vocab

EDGE_FEATURE_DIM = 4


def _categorical_feature(series: pd.Series, vocab: Vocab, field: str) -> np.ndarray:
    fv = vocab.field_vocabs[field]
    assert fv.value2id is not None
    unk_id = fv.size - 1
    ids = series.astype(str).map(lambda v: fv.value2id.get(v, unk_id))
    return (ids.to_numpy(dtype=np.float32)) / fv.size


def build_edge_features(transactions_df: pd.DataFrame, vocab: Vocab) -> Tensor:
    """Builds `(E, 4)` edge features: `[amount_norm, time_delta_log, payment_format_id,
    currency_id]`, each normalised to roughly `[0, 1]`.
    """
    amounts = transactions_df["Amount Paid"].astype(float).to_numpy()
    amount_log = np.log1p(amounts)
    amount_norm = amount_log / max(amount_log.max(), 1.0)

    timestamps = pd.to_datetime(transactions_df["Timestamp"])
    seconds_since_start = (timestamps - timestamps.min()).dt.total_seconds().to_numpy()
    time_delta_log = np.array([log_time_transform(s) for s in seconds_since_start])
    time_delta_log = time_delta_log / max(time_delta_log.max(), 1.0)

    payment_format_id = _categorical_feature(
        transactions_df["Payment Format"], vocab, "payment_format"
    )
    currency_id = _categorical_feature(
        transactions_df["Payment Currency"], vocab, "payment_currency"
    )

    edge_attr = np.stack(
        [amount_norm, time_delta_log, payment_format_id, currency_id], axis=1
    ).astype(np.float32)
    return torch.from_numpy(edge_attr)


def temporal_split_masks(
    n: int, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[Tensor, Tensor, Tensor]:
    """Boolean train/val/test masks over `n` chronologically-ordered edges (60/20/20)."""
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    train_mask = torch.zeros(n, dtype=torch.bool)
    val_mask = torch.zeros(n, dtype=torch.bool)
    test_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[:train_end] = True
    val_mask[train_end:val_end] = True
    test_mask[val_end:] = True
    return train_mask, val_mask, test_mask


def build_transaction_graph(
    transactions_df: pd.DataFrame,
    vocab: Vocab | None = None,
    node_features: Tensor | None = None,
    d_model: int = 192,
) -> Data:
    """Builds a directed PyG `Data` transaction graph.

    Nodes: unique accounts (indexed 0..N), Edges: transactions sorted by
    `Timestamp` (From Account -> To Account). `data.y` holds the per-edge
    `Is Laundering` label and `data.{train,val,test}_mask` the temporal
    60/20/20 split. `data.account_to_idx` maps account ids to node indices.
    """
    vocab = vocab or Vocab().build(transactions_df)
    df = transactions_df.sort_values("Timestamp").reset_index(drop=True)

    accounts = pd.unique(df[["Account", "Account.1"]].values.ravel())
    account_to_idx = {acc: i for i, acc in enumerate(accounts)}

    src = df["Account"].map(account_to_idx).to_numpy()
    dst = df["Account.1"].map(account_to_idx).to_numpy()
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long)

    edge_attr = build_edge_features(df, vocab)
    y = torch.tensor(df["Is Laundering"].to_numpy(), dtype=torch.float32)

    n_nodes = len(accounts)
    x = node_features if node_features is not None else torch.zeros(n_nodes, d_model)

    train_mask, val_mask, test_mask = temporal_split_masks(len(df))

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.account_to_idx = account_to_idx
    return data

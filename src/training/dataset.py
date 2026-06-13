"""Per-account event-history dataset for PRAGMA-Mini pre-training.

Groups IBM AML transactions by sender account, encodes each account's
chronological history with the KVT tokeniser, and collates variable-length
histories into fixed-size batches for `PRAGMAMini`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.tokenizer.tokenizer import EncodedEvent, KVTTokenizer, event_from_row
from src.tokenizer.vocab import Vocab


class AccountHistoryDataset(Dataset[list[EncodedEvent]]):
    """One item per sender account: its chronologically-encoded event history."""

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        vocab: Vocab,
        max_events: int = 64,
    ):
        self.vocab = vocab
        self.tokenizer = KVTTokenizer(vocab)
        self.max_events = max_events
        self.histories: list[list[EncodedEvent]] = []

        for _, group in transactions_df.groupby("Account"):
            group = group.sort_values("Timestamp")
            events = [event_from_row(row) for _, row in group.iterrows()][:max_events]
            if not events:
                continue
            self.histories.append(self.tokenizer.encode_history(events))

    def __len__(self) -> int:
        return len(self.histories)

    def __getitem__(self, idx: int) -> list[EncodedEvent]:
        return self.histories[idx]


def collate_histories(
    batch: list[list[EncodedEvent]], vocab: Vocab, max_events: int
) -> dict[str, Tensor]:
    """Pad a batch of account histories to `(B, T, F)` tensors for `PRAGMAMini`.

    `T = min(max(len(h) for h in batch), max_events)`; `F = len(vocab.field_vocabs)`
    (every transaction field is present for every IBM-AML event).
    """
    batch_size = len(batch)
    n_events = min(max((len(h) for h in batch), default=1), max_events)
    n_fields = len(vocab.field_vocabs)

    event_key_ids = torch.zeros(batch_size, n_events, n_fields, dtype=torch.long)
    event_value_ids = torch.zeros(batch_size, n_events, n_fields, dtype=torch.long)
    event_within_field_pos = torch.zeros(batch_size, n_events, n_fields, dtype=torch.long)
    event_calendar = torch.zeros(batch_size, n_events, 6, dtype=torch.float32)
    event_time_deltas = torch.zeros(batch_size, n_events, dtype=torch.float32)
    event_padding_mask = torch.ones(batch_size, n_events, dtype=torch.bool)

    for i, history in enumerate(batch):
        for t, event in enumerate(history[:n_events]):
            event_padding_mask[i, t] = False
            for f, (key_id, value_ids) in enumerate(zip(event.key_ids, event.value_ids)):
                event_key_ids[i, t, f] = key_id
                event_value_ids[i, t, f] = value_ids[0]
            event_calendar[i, t] = torch.from_numpy(np.asarray(event.calendar))
            event_time_deltas[i, t] = event.time_delta

    # No profile-state fields available in raw transaction data: use a single
    # [PAD] placeholder token (id 0) so the Profile State Encoder still runs.
    profile_key_ids = torch.zeros(batch_size, 1, dtype=torch.long)
    profile_value_ids = torch.zeros(batch_size, 1, dtype=torch.long)
    profile_within_field_pos = torch.zeros(batch_size, 1, dtype=torch.long)
    profile_time_deltas = torch.zeros(batch_size, 1, dtype=torch.float32)

    return {
        "profile_key_ids": profile_key_ids,
        "profile_value_ids": profile_value_ids,
        "profile_within_field_pos": profile_within_field_pos,
        "profile_time_deltas": profile_time_deltas,
        "event_key_ids": event_key_ids,
        "event_value_ids": event_value_ids,
        "event_within_field_pos": event_within_field_pos,
        "event_calendar": event_calendar,
        "event_time_deltas": event_time_deltas,
        "event_padding_mask": event_padding_mask,
    }


def make_synthetic_transactions_df(
    n: int = 2000, n_accounts: int = 50, seed: int = 42
) -> pd.DataFrame:
    """Generate an IBM-AML-shaped synthetic transactions dataframe for dev/smoke runs."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=n, freq="3min"),
            "From Bank": rng.choice(["10", "20", "30", "40"], n),
            "Account": [f"ACC_{i % n_accounts:04d}" for i in range(n)],
            "To Bank": rng.choice(["10", "20", "30", "40"], n),
            "Account.1": [f"ACC_{(i + 7) % n_accounts:04d}" for i in range(n)],
            "Amount Received": np.round(rng.lognormal(4, 1.5, n), 2),
            "Receiving Currency": rng.choice(["US Dollar", "Euro", "Bitcoin"], n),
            "Amount Paid": np.round(rng.lognormal(4, 1.5, n), 2),
            "Payment Currency": rng.choice(["US Dollar", "Euro", "Bitcoin"], n),
            "Payment Format": rng.choice(
                ["Reinvestment", "Wire", "Cheque", "Credit Card", "Cash", "Bitcoin"], n
            ),
            "Is Laundering": rng.choice([0, 1], n, p=[0.95, 0.05]),
        }
    )

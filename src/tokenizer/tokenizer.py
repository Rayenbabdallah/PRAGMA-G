"""Key-Value-Time (KVT) tokeniser for PRAGMA-G (paper §2.2).

Each event field is encoded as a (key_id, value_ids, time_delta) triple.
Temporal gaps use the soft log transform from the paper exactly:
`8 * ln(1 + t/8)`. Calendar features (hour/day-of-week/day-of-month) are
encoded separately as a 6-dim periodic vector.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from src.tokenizer.vocab import COLUMN_TO_KEY, KEY_FIELDS, Vocab

LOG_SCALE_FACTOR = 8.0

# Fields encoded as key-value-time tokens for an event (everything except the
# temporal coordinate itself).
EVENT_FIELDS = [k for k in KEY_FIELDS if k != "created"]


def log_time_transform(seconds_since_last_event: float) -> float:
    """Soft log transform from PRAGMA paper §2.2: 8 * ln(1 + t/8).

    Compresses dynamic range for life-long events while preserving linear
    granularity for recent events.
    """
    return LOG_SCALE_FACTOR * math.log1p(seconds_since_last_event / LOG_SCALE_FACTOR)


def calendar_features(timestamp: datetime) -> np.ndarray:
    """6-dim periodic calendar embedding: sin/cos of hour, day-of-week, day-of-month."""
    hour, dow, dom = timestamp.hour, timestamp.weekday(), timestamp.day
    return np.array(
        [
            math.sin(2 * math.pi * hour / 24),
            math.cos(2 * math.pi * hour / 24),
            math.sin(2 * math.pi * dow / 7),
            math.cos(2 * math.pi * dow / 7),
            math.sin(2 * math.pi * dom / 31),
            math.cos(2 * math.pi * dom / 31),
        ],
        dtype=np.float32,
    )


def event_from_row(row: pd.Series) -> dict:
    """Map a raw IBM AML transaction row to a PRAGMA key-named event dict."""
    return {COLUMN_TO_KEY[col]: row[col] for col in COLUMN_TO_KEY if col in row.index}


def _to_datetime(value: datetime | str) -> datetime:
    return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))


@dataclass
class EncodedEvent:
    """Tokenised representation of a single event."""

    key_ids: list[int]
    value_ids: list[list[int]]
    time_delta: float
    calendar: np.ndarray


class KVTTokenizer:
    """Encodes banking events into key-value-time token sequences."""

    def __init__(self, vocab: Vocab):
        self.vocab = vocab

    def encode_event(self, event: dict, seconds_since_last_event: float) -> EncodedEvent:
        """Encode a single event dict (PRAGMA key names) into KVT tokens.

        `seconds_since_last_event` is the time delta to the previous event in
        the account's history (0 for the first event).
        """
        timestamp = _to_datetime(event["created"])

        key_ids: list[int] = []
        value_ids: list[list[int]] = []
        for key in EVENT_FIELDS:
            if key not in event or event[key] is None:
                continue
            key_ids.append(self.vocab.key2id[key])
            value_ids.append(self.vocab.encode_value(key, event[key]))

        return EncodedEvent(
            key_ids=key_ids,
            value_ids=value_ids,
            time_delta=log_time_transform(max(seconds_since_last_event, 0.0)),
            calendar=calendar_features(timestamp),
        )

    def encode_history(self, events: list[dict]) -> list[EncodedEvent]:
        """Encode a chronologically-sorted event history.

        The first event gets `time_delta = log_time_transform(0) = 0`.
        """
        encoded: list[EncodedEvent] = []
        prev_ts: datetime | None = None
        for event in events:
            ts = _to_datetime(event["created"])
            seconds = (ts - prev_ts).total_seconds() if prev_ts is not None else 0.0
            encoded.append(self.encode_event(event, seconds))
            prev_ts = ts
        return encoded

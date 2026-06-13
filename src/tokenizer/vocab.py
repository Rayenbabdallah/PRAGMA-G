"""Vocabulary builder for PRAGMA-G key-value-time tokenisation (paper §2.2).

Two vocabularies are built:
  - Key vocabulary (~60 tokens, configurable to `key_vocab_size`): one token per
    event/profile field name. Shared between event and profile-state tokens.
  - Value vocabulary (~28K tokens, configurable to `value_vocab_size`): a
    concatenation of per-field segments. Each field is assigned one of three
    encodings based on its data type and cardinality:
      - numerical: percentile bucketing (N_NUMERICAL_BUCKETS buckets) plus a
        dedicated zero bucket (financial amounts are zero-inflated).
      - categorical: one token per unique value, if the field's cardinality is
        below `categorical_threshold`.
      - textual (high-cardinality): deterministic hash bucketing into a fixed
        number of slots. IBM AML has no free-text fields, so this replaces the
        BPE path described in the paper for fields like raw account/bank IDs.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Special tokens — shared id space across key and value vocabularies.
PAD = "[PAD]"
UNK = "[UNK]"
MASK = "[MASK]"
ZERO = "[ZERO]"
SPECIAL_TOKENS = [PAD, UNK, MASK, ZERO]

# Key vocabulary: IBM AML transaction fields + PRAGMA profile-state fields (§2.2).
KEY_FIELDS = [
    "sender_account",
    "receiver_account",
    "from_bank",
    "to_bank",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "created",  # Timestamp -> temporal coordinate, not a value token
    # Profile state fields
    "plan",
    "region",
    "balance_quantile",
    "account_age_days",
    "currency",
]

# IBM AML CSV column -> PRAGMA key name (DATASETS.md schema mapping).
COLUMN_TO_KEY = {
    "Account": "sender_account",
    "Account.1": "receiver_account",
    "From Bank": "from_bank",
    "To Bank": "to_bank",
    "Amount Received": "amount_received",
    "Receiving Currency": "receiving_currency",
    "Amount Paid": "amount_paid",
    "Payment Currency": "payment_currency",
    "Payment Format": "payment_format",
    "Timestamp": "created",
}

NUMERICAL_FIELDS = {"amount_received", "amount_paid", "balance_quantile", "account_age_days"}

N_NUMERICAL_BUCKETS = 100
CATEGORICAL_THRESHOLD = 1000
MAX_VALUE_TOKENS_PER_FIELD = 8


@dataclass
class FieldVocab:
    """Value-vocabulary segment for a single field."""

    field: str
    kind: str  # "numerical" | "categorical" | "textual"
    offset: int
    size: int
    boundaries: list[float] | None = None  # numerical: percentile split points
    value2id: dict[str, int] | None = None  # categorical: value -> local id


class Vocab:
    """Key and value vocabularies for the KVT tokeniser."""

    def __init__(self) -> None:
        self.id2key: list[str] = []
        self.key2id: dict[str, int] = {}
        self.special2id: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        self.field_vocabs: dict[str, FieldVocab] = {}
        self.value_vocab_size: int = 0

    def build(
        self,
        transactions_df: pd.DataFrame,
        key_vocab_size: int = 64,
        value_vocab_size: int = 28672,
        n_numerical_buckets: int = N_NUMERICAL_BUCKETS,
        categorical_threshold: int = CATEGORICAL_THRESHOLD,
    ) -> Vocab:
        """Build key and value vocabularies from a transactions dataframe.

        `transactions_df` is expected to use the raw IBM AML column names
        (see DATASETS.md schema); columns are mapped to PRAGMA key names via
        `COLUMN_TO_KEY`.
        """
        self._build_key_vocab(key_vocab_size)

        renamed = transactions_df.rename(columns=COLUMN_TO_KEY)

        offset = len(SPECIAL_TOKENS)
        self.field_vocabs = {}
        for key in KEY_FIELDS:
            if key == "created" or key not in renamed.columns:
                continue

            series = renamed[key].dropna()
            if key in NUMERICAL_FIELDS:
                fv = self._build_numerical_field(key, series, offset, n_numerical_buckets)
            else:
                fv = self._build_categorical_or_textual_field(
                    key, series, offset, categorical_threshold
                )
            self.field_vocabs[key] = fv
            offset += fv.size

        self.value_vocab_size = max(value_vocab_size, offset)
        return self

    def _build_key_vocab(self, key_vocab_size: int) -> None:
        id2key = list(SPECIAL_TOKENS) + list(KEY_FIELDS)
        if len(id2key) > key_vocab_size:
            raise ValueError(
                f"key_vocab_size={key_vocab_size} too small for {len(id2key)} required keys"
            )
        while len(id2key) < key_vocab_size:
            id2key.append(f"[RESERVED_{len(id2key)}]")
        self.id2key = id2key
        self.key2id = {k: i for i, k in enumerate(id2key)}

    @staticmethod
    def _build_numerical_field(
        key: str, series: pd.Series, offset: int, n_buckets: int
    ) -> FieldVocab:
        nonzero = series[series != 0].astype(float)
        if len(nonzero) >= 2:
            percentiles = np.linspace(0, 100, n_buckets + 1)
            boundaries = np.percentile(nonzero, percentiles)[1:-1].tolist()
        else:
            boundaries = []
        # n_buckets percentile bins (ids 0..n_buckets-1) + 1 dedicated zero bucket.
        return FieldVocab(
            field=key, kind="numerical", offset=offset, size=n_buckets + 1, boundaries=boundaries
        )

    @staticmethod
    def _build_categorical_or_textual_field(
        key: str, series: pd.Series, offset: int, categorical_threshold: int
    ) -> FieldVocab:
        uniques = series.astype(str).unique()
        if len(uniques) < categorical_threshold:
            value2id = {v: i for i, v in enumerate(sorted(uniques))}
            # +1 reserved slot for unseen values at inference time.
            return FieldVocab(
                field=key, kind="categorical", offset=offset,
                size=len(value2id) + 1, value2id=value2id,
            )
        # High-cardinality field: deterministic hash bucketing (textual path).
        return FieldVocab(field=key, kind="textual", offset=offset, size=categorical_threshold)

    def encode_value(self, key: str, value: Any) -> list[int]:
        """Encode a single field value to a (single-token) global value id."""
        fv = self.field_vocabs.get(key)
        if fv is None:
            return [self.special2id[UNK]]

        if value is None or (isinstance(value, float) and math.isnan(value)):
            return [self.special2id[UNK]]

        if fv.kind == "numerical":
            val = float(value)
            if val == 0:
                local = fv.size - 1  # dedicated zero bucket
            else:
                boundaries = fv.boundaries or []
                local = int(np.searchsorted(boundaries, val))
                local = min(local, fv.size - 2)
            return [fv.offset + local]

        if fv.kind == "categorical":
            assert fv.value2id is not None
            cat_local = fv.value2id.get(str(value))
            if cat_local is None:
                cat_local = fv.size - 1  # unseen-value slot
            return [fv.offset + cat_local]

        # textual: deterministic hash bucketing
        digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
        local = int(digest, 16) % fv.size
        return [fv.offset + local]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id2key": self.id2key,
            "value_vocab_size": self.value_vocab_size,
            "field_vocabs": {k: asdict(v) for k, v in self.field_vocabs.items()},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> Vocab:
        with open(path) as f:
            data = json.load(f)

        vocab = cls()
        vocab.id2key = data["id2key"]
        vocab.key2id = {k: i for i, k in enumerate(vocab.id2key)}
        vocab.value_vocab_size = data["value_vocab_size"]
        vocab.field_vocabs = {k: FieldVocab(**v) for k, v in data["field_vocabs"].items()}
        return vocab

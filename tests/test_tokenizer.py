"""Unit tests for the PRAGMA-G key-value-time tokeniser (PLAN.md Week 1)."""
from __future__ import annotations

import math
import time
from datetime import datetime

import numpy as np

from src.tokenizer.tokenizer import (
    KVTTokenizer,
    calendar_features,
    event_from_row,
    log_time_transform,
)
from src.tokenizer.vocab import KEY_FIELDS, SPECIAL_TOKENS, Vocab

# ---------------------------------------------------------------------------
# Vocab
# ---------------------------------------------------------------------------


def test_key_vocab_size_close_to_paper_estimate(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    # ~60 keys per paper (§2.2); we pad to a configurable 64.
    assert len(vocab.id2key) == 64
    assert set(KEY_FIELDS) <= set(vocab.id2key)
    assert set(SPECIAL_TOKENS) <= set(vocab.id2key)


def test_value_vocab_size_matches_config_target(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df, value_vocab_size=28672)
    # ~28K values per paper — our small field set is padded up to the target.
    assert vocab.value_vocab_size == 28672


def test_vocab_save_load_roundtrip(tmp_path, synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    path = tmp_path / "vocab.json"
    vocab.save(path)

    loaded = Vocab.load(path)
    assert loaded.id2key == vocab.id2key
    assert loaded.key2id == vocab.key2id
    assert loaded.value_vocab_size == vocab.value_vocab_size
    assert loaded.field_vocabs.keys() == vocab.field_vocabs.keys()
    for key, fv in vocab.field_vocabs.items():
        assert loaded.field_vocabs[key] == fv


# ---------------------------------------------------------------------------
# Numerical values — percentile bucketing with dedicated zero bucket
# ---------------------------------------------------------------------------


def test_numerical_zero_gets_dedicated_bucket(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    zero_id = vocab.encode_value("amount_paid", 0.0)[0]
    nonzero_id = vocab.encode_value("amount_paid", 1.0)[0]
    assert zero_id != nonzero_id


def test_numerical_bucketing_preserves_ordinal_structure(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    small = vocab.encode_value("amount_paid", 1.0)[0]
    medium = vocab.encode_value("amount_paid", 100.0)[0]
    large = vocab.encode_value("amount_paid", 1_000_000.0)[0]
    assert small < medium <= large


def test_numerical_value_id_within_field_range(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    fv = vocab.field_vocabs["amount_received"]
    value_id = vocab.encode_value("amount_received", 250.0)[0]
    assert fv.offset <= value_id < fv.offset + fv.size


# ---------------------------------------------------------------------------
# Categorical values — single token per known value, UNK for unseen
# ---------------------------------------------------------------------------


def test_categorical_known_value_is_stable(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    known = synthetic_transactions_df["Payment Format"].iloc[0]
    id_a = vocab.encode_value("payment_format", known)[0]
    id_b = vocab.encode_value("payment_format", known)[0]
    assert id_a == id_b


def test_categorical_unseen_value_maps_to_unk_slot(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    known = synthetic_transactions_df["Payment Format"].iloc[0]
    known_id = vocab.encode_value("payment_format", known)[0]
    unseen_id = vocab.encode_value("payment_format", "NeverSeenFormat")[0]
    assert known_id != unseen_id


# ---------------------------------------------------------------------------
# Textual / high-cardinality values — deterministic hash bucketing
# ---------------------------------------------------------------------------


def test_textual_field_encoding_is_deterministic(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df, categorical_threshold=10)
    fv = vocab.field_vocabs["sender_account"]
    assert fv.kind == "textual"

    id_a = vocab.encode_value("sender_account", "ACC_0001")
    id_b = vocab.encode_value("sender_account", "ACC_0001")
    assert id_a == id_b
    assert fv.offset <= id_a[0] < fv.offset + fv.size


def test_textual_field_distributes_across_buckets(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df, categorical_threshold=10)
    ids = {vocab.encode_value("sender_account", f"ACC_{i:04d}")[0] for i in range(50)}
    assert len(ids) > 1  # not all values collapse onto one bucket


# ---------------------------------------------------------------------------
# Temporal log transform (paper §2.2, exactly)
# ---------------------------------------------------------------------------


def test_log_time_transform_zero_is_zero():
    assert log_time_transform(0.0) == 0.0


def test_log_time_transform_exact_formula():
    seconds = 60.0
    expected = 8.0 * math.log(1.0 + seconds / 8.0)
    assert math.isclose(log_time_transform(seconds), expected)


def test_log_time_transform_monotonic():
    assert log_time_transform(10) < log_time_transform(100) < log_time_transform(10_000)


# ---------------------------------------------------------------------------
# Calendar features
# ---------------------------------------------------------------------------


def test_calendar_features_shape_and_range():
    cal = calendar_features(datetime(2026, 6, 1, 14, 32, 0))
    assert cal.shape == (6,)
    assert np.all(cal >= -1.0) and np.all(cal <= 1.0)


def test_calendar_features_differ_across_days():
    cal_mon = calendar_features(datetime(2026, 6, 1, 12, 0, 0))  # Monday
    cal_tue = calendar_features(datetime(2026, 6, 2, 12, 0, 0))  # Tuesday
    assert not np.allclose(cal_mon, cal_tue)


# ---------------------------------------------------------------------------
# Event / history encoding
# ---------------------------------------------------------------------------


def test_event_from_row_maps_columns_to_keys(synthetic_transactions_df):
    event = event_from_row(synthetic_transactions_df.iloc[0])
    assert "sender_account" in event
    assert "amount_paid" in event
    assert "created" in event


def test_encode_event_structure(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    tokenizer = KVTTokenizer(vocab)
    event = event_from_row(synthetic_transactions_df.iloc[0])

    encoded = tokenizer.encode_event(event, seconds_since_last_event=120.0)

    assert len(encoded.key_ids) == len(encoded.value_ids)
    # One token per transaction field present in the synthetic data (profile-only
    # fields like "plan" or "region" are absent from raw transaction events).
    assert len(encoded.key_ids) == len(vocab.field_vocabs)
    assert encoded.time_delta == log_time_transform(120.0)
    assert encoded.calendar.shape == (6,)
    for value_ids in encoded.value_ids:
        assert isinstance(value_ids, list) and len(value_ids) >= 1


def test_encode_event_first_event_has_zero_time_delta(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    tokenizer = KVTTokenizer(vocab)
    event = event_from_row(synthetic_transactions_df.iloc[0])

    encoded = tokenizer.encode_event(event, seconds_since_last_event=0.0)
    assert encoded.time_delta == 0.0


def test_encode_history_speed(synthetic_transactions_df):
    """A single account's event history encodes fast (PLAN.md benchmark: <5ms/event)."""
    vocab = Vocab().build(synthetic_transactions_df)
    tokenizer = KVTTokenizer(vocab)
    events = [event_from_row(row) for _, row in synthetic_transactions_df.head(500).iterrows()]

    start = time.perf_counter()
    encoded = tokenizer.encode_history(events)
    elapsed = time.perf_counter() - start

    assert len(encoded) == 500
    assert elapsed / len(events) < 0.005  # <5ms per event

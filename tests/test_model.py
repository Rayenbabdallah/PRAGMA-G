"""Unit tests for PRAGMA-Mini embeddings + profile encoder (PLAN.md Week 2)."""
from __future__ import annotations

import torch

from src.model.embeddings import (
    CalendarEmbedding,
    KeyEmbedding,
    RotaryPositionEmbedding,
    TokenEmbedding,
    ValueEmbedding,
)
from src.model.event_encoder import EventEncoder
from src.model.history_encoder import HistoryEncoder
from src.model.pragma_mini import PRAGMAMini, PRAGMAMiniConfig
from src.model.profile_encoder import ProfileStateEncoder

D_MODEL = 192
KEY_VOCAB_SIZE = 64
VALUE_VOCAB_SIZE = 28672


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_key_embedding_shape():
    embed = KeyEmbedding(KEY_VOCAB_SIZE, D_MODEL)
    key_ids = torch.randint(0, KEY_VOCAB_SIZE, (4, 10))
    out = embed(key_ids)
    assert out.shape == (4, 10, D_MODEL)


def test_value_embedding_shape_and_within_field_position():
    embed = ValueEmbedding(VALUE_VOCAB_SIZE, D_MODEL)
    value_ids = torch.randint(0, VALUE_VOCAB_SIZE, (4, 10))
    positions = torch.zeros(4, 10, dtype=torch.long)
    out = embed(value_ids, positions)
    assert out.shape == (4, 10, D_MODEL)

    # Different within-field positions for identical value ids -> different embeddings.
    positions_alt = torch.ones(4, 10, dtype=torch.long)
    out_alt = embed(value_ids, positions_alt)
    assert not torch.allclose(out, out_alt)


def test_token_embedding_combines_key_and_value():
    embed = TokenEmbedding(KEY_VOCAB_SIZE, VALUE_VOCAB_SIZE, D_MODEL)
    key_ids = torch.randint(0, KEY_VOCAB_SIZE, (2, 5))
    value_ids = torch.randint(0, VALUE_VOCAB_SIZE, (2, 5))
    positions = torch.zeros(2, 5, dtype=torch.long)
    out = embed(key_ids, value_ids, positions)
    assert out.shape == (2, 5, D_MODEL)


def test_calendar_embedding_shape():
    embed = CalendarEmbedding(D_MODEL)
    calendar = torch.randn(3, 7, 6)
    out = embed(calendar)
    assert out.shape == (3, 7, D_MODEL)


# ---------------------------------------------------------------------------
# RoPE
# ---------------------------------------------------------------------------


def test_rope_preserves_shape():
    rope = RotaryPositionEmbedding(D_MODEL)
    x = torch.randn(2, 5, D_MODEL)
    positions = torch.rand(2, 5) * 100
    out = rope(x, positions)
    assert out.shape == x.shape


def test_rope_differs_for_different_time_deltas():
    rope = RotaryPositionEmbedding(D_MODEL)
    x = torch.randn(1, 1, D_MODEL)
    out_a = rope(x, torch.tensor([[0.0]]))
    out_b = rope(x, torch.tensor([[50.0]]))
    assert not torch.allclose(out_a, out_b)


def test_rope_zero_position_is_identity():
    rope = RotaryPositionEmbedding(D_MODEL)
    x = torch.randn(1, 3, D_MODEL)
    out = rope(x, torch.zeros(1, 3))
    assert torch.allclose(out, x)


def test_rope_rejects_odd_d_model():
    import pytest

    with pytest.raises(ValueError):
        RotaryPositionEmbedding(193)


# ---------------------------------------------------------------------------
# Profile State Encoder
# ---------------------------------------------------------------------------


def test_profile_state_encoder_output_shape():
    encoder = ProfileStateEncoder(d_model=D_MODEL, n_layers=1, n_heads=3, d_ffn=768)
    batch_size, n_tokens = 4, 8
    profile_tokens = torch.randn(batch_size, n_tokens, D_MODEL)
    time_deltas = torch.rand(batch_size, n_tokens) * 100

    usr_embedding = encoder(profile_tokens, time_deltas)
    assert usr_embedding.shape == (batch_size, D_MODEL)


def test_profile_state_encoder_sensitive_to_time_deltas():
    encoder = ProfileStateEncoder(d_model=D_MODEL, n_layers=1, n_heads=3, d_ffn=768)
    encoder.eval()

    profile_tokens = torch.randn(2, 6, D_MODEL)
    time_deltas_a = torch.zeros(2, 6)
    time_deltas_b = torch.rand(2, 6) * 1000

    out_a = encoder(profile_tokens, time_deltas_a)
    out_b = encoder(profile_tokens, time_deltas_b)
    assert not torch.allclose(out_a, out_b)


def test_profile_state_encoder_pragma_s_config():
    encoder = ProfileStateEncoder(d_model=192, n_layers=1, n_heads=3, d_ffn=768, dropout=0.1)
    profile_tokens = torch.randn(1, 32, 192)
    time_deltas = torch.rand(1, 32) * 1000
    out = encoder(profile_tokens, time_deltas)
    assert out.shape == (1, 192)


# ---------------------------------------------------------------------------
# Event Encoder
# ---------------------------------------------------------------------------


def test_event_encoder_output_shapes():
    encoder = EventEncoder(d_model=D_MODEL, n_layers=5, n_heads=3, d_ffn=768)
    batch_size, n_events, n_fields = 2, 4, 9
    event_tokens = torch.randn(batch_size, n_events, n_fields, D_MODEL)
    calendar = torch.randn(batch_size, n_events, 6)

    evt_embs, field_hidden = encoder(event_tokens, calendar)
    assert evt_embs.shape == (batch_size, n_events, D_MODEL)
    assert field_hidden.shape == (batch_size, n_events, n_fields, D_MODEL)


# ---------------------------------------------------------------------------
# History Encoder
# ---------------------------------------------------------------------------


def test_history_encoder_output_shape():
    encoder = HistoryEncoder(d_model=D_MODEL, n_layers=2, n_heads=3, d_ffn=768)
    batch_size, n_events = 2, 5
    usr_emb = torch.randn(batch_size, D_MODEL)
    evt_embs = torch.randn(batch_size, n_events, D_MODEL)
    time_to_last_event = torch.rand(batch_size, n_events) * 100

    z_h = encoder(usr_emb, evt_embs, time_to_last_event)
    assert z_h.shape == (batch_size, D_MODEL)


def test_history_encoder_respects_padding_mask():
    encoder = HistoryEncoder(d_model=D_MODEL, n_layers=2, n_heads=3, d_ffn=768)
    encoder.eval()
    batch_size, n_events = 2, 5
    usr_emb = torch.randn(batch_size, D_MODEL)
    evt_embs = torch.randn(batch_size, n_events, D_MODEL)
    time_to_last_event = torch.rand(batch_size, n_events) * 100

    padding_mask = torch.zeros(batch_size, n_events, dtype=torch.bool)
    padding_mask[:, -1] = True  # last event is padding

    z_h = encoder(usr_emb, evt_embs, time_to_last_event, padding_mask)
    assert z_h.shape == (batch_size, D_MODEL)


# ---------------------------------------------------------------------------
# Full PRAGMA-Mini
# ---------------------------------------------------------------------------


def _dummy_batch(batch_size=2, n_events=3, n_fields=9, config=None):
    config = config or PRAGMAMiniConfig()
    return {
        "profile_key_ids": torch.zeros(batch_size, 1, dtype=torch.long),
        "profile_value_ids": torch.zeros(batch_size, 1, dtype=torch.long),
        "profile_within_field_pos": torch.zeros(batch_size, 1, dtype=torch.long),
        "profile_time_deltas": torch.zeros(batch_size, 1),
        "event_key_ids": torch.randint(0, config.key_vocab_size, (batch_size, n_events, n_fields)),
        "event_value_ids": torch.randint(
            0, config.value_vocab_size, (batch_size, n_events, n_fields)
        ),
        "event_within_field_pos": torch.zeros(batch_size, n_events, n_fields, dtype=torch.long),
        "event_calendar": torch.randn(batch_size, n_events, 6),
        "event_time_deltas": torch.rand(batch_size, n_events) * 100,
        "event_padding_mask": torch.zeros(batch_size, n_events, dtype=torch.bool),
    }


def test_pragma_mini_forward_shapes():
    config = PRAGMAMiniConfig()
    model = PRAGMAMini(config)
    batch = _dummy_batch(config=config)

    out = model(batch)
    batch_size, n_events = 2, 3
    assert out["z_h"].shape == (batch_size, config.d_model)
    assert out["usr_emb"].shape == (batch_size, config.d_model)
    assert out["evt_embs"].shape == (batch_size, n_events, config.d_model)
    assert out["mlm_logits"].shape == (batch_size, n_events, 9, config.value_vocab_size)


def test_pragma_mini_config_from_yaml():
    config = PRAGMAMiniConfig.from_yaml("configs/pragma_s.yaml")
    assert config.d_model == 192
    assert config.d_ffn == 768
    assert config.n_heads == 3
    assert config.profile_layers == 1
    assert config.event_layers == 5
    assert config.history_layers == 2
    assert config.key_vocab_size == 64
    assert config.value_vocab_size == 28672

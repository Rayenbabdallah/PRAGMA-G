"""Embedding layers for PRAGMA-Mini (paper §2.3.1).

Combines key, value, within-field positional, calendar, and rotary temporal
embeddings into the token representations consumed by the encoder branches.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.tokenizer.vocab import MAX_VALUE_TOKENS_PER_FIELD


class KeyEmbedding(nn.Module):
    """One embedding per key-vocabulary token."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, key_ids: Tensor) -> Tensor:
        return self.embed(key_ids)


class ValueEmbedding(nn.Module):
    """Value-vocabulary embedding plus within-field positional embedding.

    The within-field position distinguishes the multiple tokens that can
    represent a single multi-token (e.g. textual) field value.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_value_tokens_per_field: int = MAX_VALUE_TOKENS_PER_FIELD,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.within_field_pos = nn.Embedding(max_value_tokens_per_field, d_model)

    def forward(self, value_ids: Tensor, within_field_positions: Tensor) -> Tensor:
        return self.embed(value_ids) + self.within_field_pos(within_field_positions)


class TokenEmbedding(nn.Module):
    """Combined key + value token embedding (paper §2.3.1)."""

    def __init__(
        self,
        key_vocab_size: int,
        value_vocab_size: int,
        d_model: int,
        max_value_tokens_per_field: int = MAX_VALUE_TOKENS_PER_FIELD,
    ):
        super().__init__()
        self.key_embed = KeyEmbedding(key_vocab_size, d_model)
        self.value_embed = ValueEmbedding(value_vocab_size, d_model, max_value_tokens_per_field)

    def forward(
        self, key_ids: Tensor, value_ids: Tensor, within_field_positions: Tensor
    ) -> Tensor:
        return self.key_embed(key_ids) + self.value_embed(value_ids, within_field_positions)


class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) driven by continuous time deltas.

    Unlike standard RoPE (integer token positions), PRAGMA uses the
    log-transformed time-since-last-event as the rotation angle input,
    so temporal gaps directly modulate the embedding geometry.
    """

    inv_freq: Tensor

    def __init__(self, d_model: int, base: float = 10000.0):
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model must be even for RoPE, got {d_model}")
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Apply rotary embedding to `x` using `positions` as the angle input.

        x: (B, T, D), positions: (B, T) -> (B, T, D)
        """
        freqs = positions.unsqueeze(-1) * self.inv_freq  # (B, T, D/2)
        cos = torch.repeat_interleave(freqs.cos(), 2, dim=-1)
        sin = torch.repeat_interleave(freqs.sin(), 2, dim=-1)
        x_rotated = self._rotate_every_two(x)
        return x * cos + x_rotated * sin

    @staticmethod
    def _rotate_every_two(x: Tensor) -> Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        return torch.stack([-x2, x1], dim=-1).reshape_as(x)


class CalendarEmbedding(nn.Module):
    """Projects the 6-dim periodic calendar feature vector to `d_model`."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(6, d_model)

    def forward(self, calendar_features: Tensor) -> Tensor:
        return self.proj(calendar_features)

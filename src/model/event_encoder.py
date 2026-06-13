"""Event Encoder (PRAGMA paper §2.3.3).

Encodes each event record independently (no cross-event attention) into an
[EVT] embedding, then adds a projection of the event's calendar features.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class EventEncoder(nn.Module):
    """Per-event Transformer encoder producing (B, T_events, d_model) [EVT] embeddings.

    PRAGMA-S config: d_model=192, n_layers=5, n_heads=3, d_ffn=768.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 5,
        n_heads: int = 3,
        d_ffn: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.evt_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.evt_token, std=0.02)
        self.within_event_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,
                n_heads,
                d_ffn,
                dropout,
                activation="gelu",
                norm_first=True,
                batch_first=True,
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.calendar_proj = nn.Linear(6, d_model)

    def forward(self, event_tokens: Tensor, calendar_features: Tensor) -> tuple[Tensor, Tensor]:
        """
        event_tokens: (B, T_events, T_fields, d_model) — embedded key-value tokens per field
        calendar_features: (B, T_events, 6) — periodic calendar features
        Returns:
            evt_embs: (B, T_events, d_model) — [EVT] embeddings + calendar projection
            field_hidden: (B, T_events, T_fields, d_model) — contextualised per-field states
                (used by the MLM head during pre-training)
        """
        batch_size, n_events, n_fields, d_model = event_tokens.shape

        x = event_tokens.reshape(batch_size * n_events, n_fields, d_model)
        evt = self.evt_token.expand(batch_size * n_events, -1, -1)
        x = torch.cat([evt, x], dim=1)
        x = self.within_event_transformer(x)
        evt_emb = x[:, 0, :].view(batch_size, n_events, d_model)
        field_hidden = x[:, 1:, :].reshape(batch_size, n_events, n_fields, d_model)

        cal = self.calendar_proj(calendar_features)
        return evt_emb + cal, field_hidden

"""History Encoder (PRAGMA paper §2.3.4).

Contextualises the full sequence [USR, EVT_1, ..., EVT_T] using rotary
temporal position embeddings keyed on time-to-last-event, producing the
final record-level embedding z_h.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.embeddings import RotaryPositionEmbedding


class HistoryEncoder(nn.Module):
    """Transformer encoder producing the record-level embedding z_h: (B, d_model).

    PRAGMA-S config: d_model=192, n_layers=2, n_heads=3, d_ffn=768.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 2,
        n_heads: int = 3,
        d_ffn: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.rope = RotaryPositionEmbedding(d_model)
        self.transformer = nn.TransformerEncoder(
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

    def forward(
        self,
        usr_emb: Tensor,
        evt_embs: Tensor,
        time_to_last_event: Tensor,
        event_padding_mask: Tensor | None = None,
    ) -> Tensor:
        """
        usr_emb: (B, d_model) — [USR] embedding from the Profile State Encoder
        evt_embs: (B, T_events, d_model) — [EVT] embeddings from the Event Encoder
        time_to_last_event: (B, T_events) — log-transformed time deltas between events
        event_padding_mask: (B, T_events) bool, True at padded event positions
        Returns: (B, d_model) — record-level embedding z_h
        """
        x = torch.cat([usr_emb.unsqueeze(1), evt_embs], dim=1)

        batch_size = x.shape[0]
        zeros = torch.zeros(batch_size, 1, device=x.device, dtype=time_to_last_event.dtype)
        times = torch.cat([zeros, time_to_last_event], dim=1)
        x = self.rope(x, times)

        key_padding_mask = None
        if event_padding_mask is not None:
            usr_mask = torch.zeros(
                batch_size, 1, dtype=torch.bool, device=event_padding_mask.device
            )
            key_padding_mask = torch.cat([usr_mask, event_padding_mask], dim=1)

        z = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return z[:, 0, :]

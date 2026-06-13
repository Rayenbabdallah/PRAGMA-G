"""Profile State Encoder (PRAGMA paper §2.3.2).

Encodes static/slow-changing account attributes (profile-state key-value
tokens) into a single [USR] embedding via a pre-norm Transformer encoder
with rotary temporal position embeddings.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn

from src.model.embeddings import RotaryPositionEmbedding


class ProfileStateEncoder(nn.Module):
    """Transformer encoder producing a (B, d_model) [USR] embedding.

    PRAGMA-S config: d_model=192, n_layers=1, n_heads=3, d_ffn=768.
    """

    def __init__(
        self,
        d_model: int = 192,
        n_layers: int = 1,
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
        self.usr_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.usr_token, std=0.02)

    def forward(self, profile_tokens: Tensor, time_deltas: Tensor) -> Tensor:
        """
        profile_tokens: (B, T_profile, d_model) — embedded profile key-value tokens
        time_deltas: (B, T_profile) — log-transformed time since life-long events
        Returns: (B, d_model) — [USR] token representation
        """
        batch_size = profile_tokens.shape[0]
        usr = self.usr_token.expand(batch_size, -1, -1)
        x = torch.cat([usr, profile_tokens], dim=1)

        zeros = torch.zeros(batch_size, 1, device=x.device, dtype=time_deltas.dtype)
        times = torch.cat([zeros, time_deltas], dim=1)
        x = self.rope(x, times)

        x = self.transformer(x)
        return x[:, 0, :]

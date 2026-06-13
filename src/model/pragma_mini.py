"""Full PRAGMA-Mini model (paper §2.3): wires the three encoder branches.

`PRAGMAMini.forward(batch)` returns the record-level embedding `z_h` plus the
intermediate [USR]/[EVT] embeddings and MLM logits used for pre-training
(paper §2.3.5).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from torch import Tensor, nn

from src.model.embeddings import TokenEmbedding
from src.model.event_encoder import EventEncoder
from src.model.history_encoder import HistoryEncoder
from src.model.profile_encoder import ProfileStateEncoder


@dataclass
class PRAGMAMiniConfig:
    """Architecture hyperparameters (PRAGMA-S defaults from Table 1)."""

    d_model: int = 192
    d_ffn: int = 768
    n_heads: int = 3
    dropout: float = 0.1
    key_vocab_size: int = 64
    value_vocab_size: int = 28672
    max_value_tokens_per_field: int = 8
    profile_layers: int = 1
    event_layers: int = 5
    history_layers: int = 2

    @classmethod
    def from_yaml(cls, path: str | Path) -> PRAGMAMiniConfig:
        with open(path) as f:
            data: dict[str, Any] = yaml.safe_load(f)
        m = data["model"]
        tok = m["tokenizer"]
        return cls(
            d_model=m["d_model"],
            d_ffn=m["d_ffn"],
            n_heads=m["n_heads"],
            dropout=m["dropout"],
            key_vocab_size=tok["key_vocab_size"],
            value_vocab_size=tok["value_vocab_size"],
            max_value_tokens_per_field=tok["max_value_tokens_per_field"],
            profile_layers=m["profile_encoder"]["n_layers"],
            event_layers=m["event_encoder"]["n_layers"],
            history_layers=m["history_encoder"]["n_layers"],
        )


class PRAGMAMini(nn.Module):
    """Profile State Encoder + Event Encoder + History Encoder, fully wired."""

    def __init__(self, config: PRAGMAMiniConfig):
        super().__init__()
        self.config = config

        self.token_embed = TokenEmbedding(
            config.key_vocab_size,
            config.value_vocab_size,
            config.d_model,
            config.max_value_tokens_per_field,
        )
        self.profile_encoder = ProfileStateEncoder(
            config.d_model, config.profile_layers, config.n_heads, config.d_ffn, config.dropout
        )
        self.event_encoder = EventEncoder(
            config.d_model, config.event_layers, config.n_heads, config.d_ffn, config.dropout
        )
        self.history_encoder = HistoryEncoder(
            config.d_model, config.history_layers, config.n_heads, config.d_ffn, config.dropout
        )
        self.mlm_head = nn.Linear(config.d_model, config.value_vocab_size)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Required batch keys:
            profile_key_ids, profile_value_ids, profile_within_field_pos: (B, T_profile)
            profile_time_deltas: (B, T_profile)
            event_key_ids, event_value_ids, event_within_field_pos: (B, T_events, T_fields)
            event_calendar: (B, T_events, 6)
            event_time_deltas: (B, T_events)
            event_padding_mask: (B, T_events) bool, True at padded positions (optional)

        Returns:
            z_h: (B, d_model) — record-level embedding
            usr_emb: (B, d_model)
            evt_embs: (B, T_events, d_model)
            mlm_logits: (B, T_events, T_fields, value_vocab_size)
        """
        profile_emb = self.token_embed(
            batch["profile_key_ids"],
            batch["profile_value_ids"],
            batch["profile_within_field_pos"],
        )
        usr_emb = self.profile_encoder(profile_emb, batch["profile_time_deltas"])

        event_emb = self.token_embed(
            batch["event_key_ids"], batch["event_value_ids"], batch["event_within_field_pos"]
        )
        evt_embs, field_hidden = self.event_encoder(event_emb, batch["event_calendar"])

        z_h = self.history_encoder(
            usr_emb, evt_embs, batch["event_time_deltas"], batch.get("event_padding_mask")
        )

        mlm_logits = self.mlm_head(field_hidden)

        return {
            "z_h": z_h,
            "usr_emb": usr_emb,
            "evt_embs": evt_embs,
            "mlm_logits": mlm_logits,
        }

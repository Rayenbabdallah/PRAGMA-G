"""MLM pre-training loss (PRAGMA paper §2.3.5).

Randomly mask 15% of value tokens and predict the original value id from the
contextualised per-field representation. Key tokens and the temporal
coordinate are never masked.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class MLMMaskResult:
    masked_value_ids: Tensor
    mask: Tensor  # bool, True where a token was masked and should be predicted


def apply_mlm_masking(
    value_ids: Tensor,
    mask_token_id: int,
    padding_mask: Tensor | None = None,
    mask_prob: float = 0.15,
    generator: torch.Generator | None = None,
) -> MLMMaskResult:
    """Replace `mask_prob` of `value_ids` with `mask_token_id`.

    `padding_mask` is (B, T) bool with True at padded event positions; padded
    positions are broadcast over the field dimension and excluded from masking.
    """
    probs = torch.rand(value_ids.shape, generator=generator, device=value_ids.device)
    mask = probs < mask_prob

    if padding_mask is not None:
        mask = mask & ~padding_mask.unsqueeze(-1)

    masked_value_ids = value_ids.clone()
    masked_value_ids[mask] = mask_token_id
    return MLMMaskResult(masked_value_ids=masked_value_ids, mask=mask)


class MLMLoss(nn.Module):
    """Cross-entropy over the value vocabulary at masked positions only."""

    def __init__(self) -> None:
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, logits: Tensor, value_ids: Tensor, mask: Tensor) -> Tensor:
        """
        logits: (B, T, F, value_vocab_size)
        value_ids: (B, T, F) — original (unmasked) value ids
        mask: (B, T, F) bool — positions to score
        """
        if not bool(mask.any()):
            return logits.sum() * 0.0
        return self.cross_entropy(logits[mask], value_ids[mask])

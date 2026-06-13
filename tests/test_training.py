"""Unit tests for MLM masking, loss, dataset collation, and the pre-train loop
(PLAN.md Week 3)."""
from __future__ import annotations

import torch

from src.model.pragma_mini import PRAGMAMiniConfig
from src.tokenizer.vocab import Vocab
from src.training.dataset import AccountHistoryDataset, collate_histories
from src.training.losses import MLMLoss, apply_mlm_masking
from src.training.pretrain import run_pretrain

# ---------------------------------------------------------------------------
# MLM masking + loss
# ---------------------------------------------------------------------------


def test_apply_mlm_masking_replaces_masked_positions():
    value_ids = torch.randint(0, 100, (4, 5, 9))
    padding_mask = torch.zeros(4, 5, dtype=torch.bool)
    mask_token_id = 2

    result = apply_mlm_masking(value_ids, mask_token_id, padding_mask, mask_prob=1.0)
    assert torch.all(result.masked_value_ids[result.mask] == mask_token_id)
    assert torch.equal(result.mask, torch.ones_like(result.mask))


def test_apply_mlm_masking_excludes_padding():
    value_ids = torch.randint(0, 100, (2, 4, 9))
    padding_mask = torch.zeros(2, 4, dtype=torch.bool)
    padding_mask[:, -1] = True

    result = apply_mlm_masking(value_ids, mask_token_id=2, padding_mask=padding_mask, mask_prob=1.0)
    assert not result.mask[:, -1, :].any()
    assert result.mask[:, :-1, :].all()


def test_mlm_loss_zero_when_no_masked_tokens():
    loss_fn = MLMLoss()
    logits = torch.randn(2, 3, 9, 50)
    value_ids = torch.randint(0, 50, (2, 3, 9))
    mask = torch.zeros(2, 3, 9, dtype=torch.bool)

    loss = loss_fn(logits, value_ids, mask)
    assert torch.equal(loss, torch.tensor(0.0))


def test_mlm_loss_positive_when_masked_tokens_present():
    loss_fn = MLMLoss()
    logits = torch.randn(2, 3, 9, 50)
    value_ids = torch.randint(0, 50, (2, 3, 9))
    mask = torch.zeros(2, 3, 9, dtype=torch.bool)
    mask[0, 0, 0] = True

    loss = loss_fn(logits, value_ids, mask)
    assert loss.item() > 0


# ---------------------------------------------------------------------------
# Dataset + collation
# ---------------------------------------------------------------------------


def test_account_history_dataset_groups_by_sender(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    dataset = AccountHistoryDataset(synthetic_transactions_df, vocab, max_events=64)

    assert len(dataset) == synthetic_transactions_df["Account"].nunique()
    for history in dataset:
        assert 0 < len(history) <= 64


def test_collate_histories_shapes(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    dataset = AccountHistoryDataset(synthetic_transactions_df, vocab, max_events=16)
    batch = [dataset[i] for i in range(4)]

    collated = collate_histories(batch, vocab, max_events=16)
    n_fields = len(vocab.field_vocabs)

    assert collated["event_key_ids"].shape[0] == 4
    assert collated["event_key_ids"].shape[2] == n_fields
    assert collated["event_value_ids"].shape == collated["event_key_ids"].shape
    assert collated["event_calendar"].shape[-1] == 6
    assert collated["event_padding_mask"].dtype == torch.bool


def test_collate_histories_value_ids_within_vocab(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    dataset = AccountHistoryDataset(synthetic_transactions_df, vocab, max_events=16)
    batch = [dataset[i] for i in range(4)]
    collated = collate_histories(batch, vocab, max_events=16)

    assert collated["event_value_ids"].max().item() < vocab.value_vocab_size
    assert collated["event_key_ids"].max().item() < len(vocab.id2key)


# ---------------------------------------------------------------------------
# Pre-train loop smoke test
# ---------------------------------------------------------------------------


def test_run_pretrain_smoke(tmp_path, synthetic_transactions_df, monkeypatch):
    """Tiny end-to-end run: MLM loss is finite and the loop completes without crashing."""
    monkeypatch.chdir(tmp_path)

    config = PRAGMAMiniConfig(
        d_model=32,
        d_ffn=64,
        n_heads=2,
        key_vocab_size=64,
        value_vocab_size=512,
        profile_layers=1,
        event_layers=1,
        history_layers=1,
    )

    small_df = synthetic_transactions_df.iloc[:200]
    losses = run_pretrain(
        config,
        small_df,
        epochs=2,
        batch_size=4,
        max_events=8,
        learning_rate=1e-3,
        mask_prob=0.15,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert len(losses) == 2
    for loss in losses:
        assert loss == loss  # not NaN
        assert loss >= 0

    assert (tmp_path / "checkpoints" / "pragma_mini.pt").exists()

"""Unit tests for the PRAGMA-G fusion classifier and fine-tuning loop (PLAN.md Week 5)."""
from __future__ import annotations

import torch

from src.graph.graph_builder import build_transaction_graph
from src.model.classifier import PRAGMAGClassifier
from src.model.pragma_mini import PRAGMAMiniConfig
from src.tokenizer.vocab import Vocab
from src.training.baseline import build_edge_features_with_accounts, run_xgboost_baseline
from src.training.finetune import run_finetune

D_MODEL = 32

TINY_GRAPH_CONFIG = {
    "hidden_channels": 16,
    "n_layers": 2,
    "aggregation": "mean",
    "dropout": 0.1,
}


def _tiny_config() -> PRAGMAMiniConfig:
    return PRAGMAMiniConfig(
        d_model=D_MODEL,
        d_ffn=64,
        n_heads=2,
        key_vocab_size=64,
        value_vocab_size=512,
        profile_layers=1,
        event_layers=1,
        history_layers=1,
    )


# ---------------------------------------------------------------------------
# PRAGMAGClassifier
# ---------------------------------------------------------------------------


def _make_classifier() -> PRAGMAGClassifier:
    return PRAGMAGClassifier(
        d_model=D_MODEL,
        graph_hidden_channels=TINY_GRAPH_CONFIG["hidden_channels"],
        graph_n_layers=TINY_GRAPH_CONFIG["n_layers"],
        graph_aggregation=TINY_GRAPH_CONFIG["aggregation"],
        dropout=TINY_GRAPH_CONFIG["dropout"],
    )


def test_pragmag_classifier_output_shape(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab, d_model=D_MODEL)

    classifier = _make_classifier()
    z_temporal = torch.randn(data.x.shape[0], D_MODEL)

    logits = classifier(z_temporal, data.edge_index, data.edge_attr)
    n_edges = data.edge_index.shape[1]
    assert logits.shape == (n_edges,)


def test_pragmag_classifier_no_graph_matches_zero_graph_embedding(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab, d_model=D_MODEL)

    classifier = _make_classifier()
    classifier.eval()
    z_temporal = torch.randn(data.x.shape[0], D_MODEL)

    no_graph_logits = classifier(z_temporal, data.edge_index, data.edge_attr, use_graph=False)

    src, dst = data.edge_index[0], data.edge_index[1]
    temporal_edge = z_temporal[src] + z_temporal[dst]
    z_fused = torch.cat([temporal_edge, torch.zeros_like(temporal_edge)], dim=-1)
    expected = classifier.fusion(z_fused).squeeze(-1)

    assert torch.allclose(no_graph_logits, expected)


# ---------------------------------------------------------------------------
# Fine-tuning loop smoke test
# ---------------------------------------------------------------------------


def test_run_finetune_smoke(tmp_path, synthetic_transactions_df, monkeypatch):
    """Tiny end-to-end run: both fine-tuning stages complete and produce finite metrics."""
    monkeypatch.chdir(tmp_path)

    config = _tiny_config()

    metrics = run_finetune(
        config,
        synthetic_transactions_df,
        graph_config=TINY_GRAPH_CONFIG,
        stage_1_epochs=1,
        stage_2_epochs=1,
        learning_rate=1e-3,
        pos_weight=19.0,
        lora_r=4,
        lora_alpha=8,
        max_events=8,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    for key in ("pr_auc", "roc_auc", "precision_at_recall_0.5", "cost"):
        assert key in metrics
        assert metrics[key] == metrics[key]  # not NaN

    assert (tmp_path / "checkpoints" / "pragma_mini_lora.pt").exists()
    assert (tmp_path / "checkpoints" / "classifier.pt").exists()


def test_run_finetune_no_graph_smoke(tmp_path, synthetic_transactions_df, monkeypatch):
    """PRAGMA-Mini-only baseline (no GraphSAGE) also runs end to end."""
    monkeypatch.chdir(tmp_path)

    config = _tiny_config()

    metrics = run_finetune(
        config,
        synthetic_transactions_df,
        graph_config=TINY_GRAPH_CONFIG,
        stage_1_epochs=1,
        stage_2_epochs=1,
        learning_rate=1e-3,
        pos_weight=19.0,
        lora_r=4,
        lora_alpha=8,
        max_events=8,
        checkpoint_dir=tmp_path / "checkpoints",
        use_graph=False,
    )

    assert metrics["pr_auc"] == metrics["pr_auc"]  # not NaN


# ---------------------------------------------------------------------------
# XGBoost baseline
# ---------------------------------------------------------------------------


def test_build_edge_features_with_accounts_shape(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab, d_model=D_MODEL)

    features = build_edge_features_with_accounts(synthetic_transactions_df, data)
    n_edges = data.edge_index.shape[1]
    assert features.shape == (n_edges, data.edge_attr.shape[1] + 12)


def test_run_xgboost_baseline_smoke(synthetic_transactions_df):
    vocab = Vocab().build(synthetic_transactions_df)
    data = build_transaction_graph(synthetic_transactions_df, vocab=vocab, d_model=D_MODEL)

    result = run_xgboost_baseline(synthetic_transactions_df, data)
    assert "pr_auc" in result["test"]
    assert result["test"]["pr_auc"] == result["test"]["pr_auc"]  # not NaN

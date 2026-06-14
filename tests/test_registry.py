"""Tests for the MLflow model registry helpers (PLAN.md Week 9)."""
from __future__ import annotations

from pathlib import Path

import torch

from src.model.classifier import PRAGMAGClassifier
from src.model.pragma_mini import PRAGMAMiniConfig
from src.training.finetune import run_finetune
from src.training.registry import load_registry_model

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


def test_load_registry_model_missing_returns_none(tmp_path: Path) -> None:
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    result = load_registry_model("does-not-exist", stage="Production", tracking_uri=tracking_uri)
    assert result is None


def test_register_and_load_checkpoint(
    tmp_path: Path, synthetic_transactions_df, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    config = _tiny_config()

    run_finetune(
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
        tracking_uri=tracking_uri,
        register_model=True,
        model_registry_name="test-pragma-g-aml",
        registry_stage="Staging",
    )

    # Not yet in Production.
    assert (
        load_registry_model("test-pragma-g-aml", stage="Production", tracking_uri=tracking_uri)
        is None
    )

    loaded = load_registry_model("test-pragma-g-aml", stage="Staging", tracking_uri=tracking_uri)
    assert loaded is not None
    pragma_mini, classifier, version = loaded
    assert version == "1"
    assert isinstance(classifier, PRAGMAGClassifier)

    # Loaded models are ready for inference.
    pragma_mini.eval()
    classifier.eval()
    with torch.no_grad():
        z = torch.randn(3, D_MODEL)
        edge_index = torch.tensor([[0, 1], [1, 2]])
        edge_attr = torch.zeros(2, 4)
        logits = classifier(z, edge_index, edge_attr)
    assert logits.shape == (2,)

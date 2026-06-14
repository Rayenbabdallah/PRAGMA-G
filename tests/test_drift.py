"""Tests for Evidently drift monitoring (PLAN.md Week 7)."""
from __future__ import annotations

from pathlib import Path

from src.monitoring.drift import DRIFT_COLUMNS, build_reference_dataset, run_drift_check
from src.monitoring.metrics import DRIFT_DETECTED
from src.training.dataset import make_synthetic_transactions_df


def test_build_reference_dataset() -> None:
    reference = build_reference_dataset(n=200)
    assert list(reference.columns) == DRIFT_COLUMNS
    assert len(reference) == 200


def test_run_drift_check_no_drift(tmp_path: Path) -> None:
    reference = build_reference_dataset(n=500)
    current = make_synthetic_transactions_df(n=500, seed=123)[DRIFT_COLUMNS]

    result = run_drift_check(reference, current, report_dir=tmp_path)

    assert result["dataset_drift"] is False
    assert 0.0 <= result["drift_share"] <= 1.0
    assert result["n_columns"] == len(DRIFT_COLUMNS)
    assert list(tmp_path.glob("drift_*.html"))
    assert DRIFT_DETECTED._value.get() == 0.0


def test_run_drift_check_detects_drift(tmp_path: Path) -> None:
    reference = build_reference_dataset(n=500)
    current = reference.copy()
    current["Amount Paid"] = current["Amount Paid"] * 1000.0
    current["Amount Received"] = current["Amount Received"] * 1000.0
    current["Payment Format"] = "Bitcoin"
    current["Payment Currency"] = "Bitcoin"

    result = run_drift_check(reference, current, report_dir=tmp_path)

    assert result["dataset_drift"] is True
    assert DRIFT_DETECTED._value.get() == 1.0

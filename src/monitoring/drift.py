"""Evidently drift monitoring for PRAGMA-G serving (PLAN.md Week 7).

Compares the feature distribution of recently-scored live transactions
against a reference distribution using Evidently's `DataDriftPreset`.
Intended to be triggered periodically (every `monitoring.drift_check_interval`
requests, per `configs/pragma_s.yaml`) by `scripts/stream_consumer.py`, or
ad-hoc/cron via a small wrapper script.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.monitoring.metrics import DRIFT_DETECTED
from src.training.dataset import make_synthetic_transactions_df

# Raw transaction fields whose distributions are compared. These are the
# fields PRAGMA-G consumes directly (DATASETS.md schema), excluding
# identifiers (Account, bank ids) and the label (Is Laundering).
DRIFT_COLUMNS = [
    "Amount Paid",
    "Amount Received",
    "Payment Format",
    "Payment Currency",
    "Receiving Currency",
]


def build_reference_dataset(n: int = 2000) -> pd.DataFrame:
    """Reference distribution for drift comparison.

    Uses the synthetic IBM-AML-shaped generator as a stand-in for a stored
    sample of training data, so drift checks run without requiring the
    (gitignored) IBM AML dataset to be present.
    """
    return make_synthetic_transactions_df(n=n)[DRIFT_COLUMNS]


def run_drift_check(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    report_dir: Path | str = "monitoring/reports",
) -> dict[str, Any]:
    """Runs Evidently's `DataDriftPreset` on `current` vs `reference`.

    Saves an HTML report to `report_dir`, updates the `pragma_g_drift_detected`
    Prometheus gauge, and returns the dataset-level drift summary.
    """
    report = Report(metrics=[DataDriftPreset()])
    report.run(
        reference_data=reference[DRIFT_COLUMNS], current_data=current[DRIFT_COLUMNS]
    )

    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report.save_html(str(report_dir / f"drift_{timestamp}.html"))

    drift_result = report.as_dict()["metrics"][0]["result"]
    dataset_drift = bool(drift_result["dataset_drift"])
    DRIFT_DETECTED.set(1.0 if dataset_drift else 0.0)

    return {
        "dataset_drift": dataset_drift,
        "drift_share": drift_result["share_of_drifted_columns"],
        "n_drifted_columns": drift_result["number_of_drifted_columns"],
        "n_columns": drift_result["number_of_columns"],
    }

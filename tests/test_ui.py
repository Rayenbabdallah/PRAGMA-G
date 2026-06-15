"""Tests for the Gradio What-If/explainability UI (PLAN.md Week 10)."""
from __future__ import annotations

import gradio as gr

from src.ui.app import build_app, latest_drift_report


def test_build_app_returns_blocks() -> None:
    demo = build_app()
    assert isinstance(demo, gr.Blocks)


def test_latest_drift_report_placeholder_when_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    html = latest_drift_report()
    assert "No drift reports found" in html

"""Gradio What-If simulator + explainability UI for PRAGMA-G (PLAN.md Week 10).

A thin client over the FastAPI serving layer (`src.api.main`) — talks to it
over HTTP at `PRAGMA_G_API_URL` (default `http://localhost:8000`), so it can
run as a separate container/HF Space alongside the API (see
`docker-compose.yml`'s `ui` service).

Panels:
  1. Score + Explain — enter a transaction, get a score + top-5 SHAP values.
  2. What-If — compare the score for an original vs modified transaction.
  3. Transaction Graph — 1-hop neighbourhood + per-edge risk scores
     (`/explain/graph`), rendered with pyvis.
  4. Drift Monitor — embeds the latest Evidently drift report written by
     `src.monitoring.drift.run_drift_check`.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
import httpx
from pyvis.network import Network

API_URL = os.environ.get("PRAGMA_G_API_URL", "http://localhost:8000")
DRIFT_REPORT_DIR = Path("monitoring/reports")

PAYMENT_FORMATS = ["Wire", "ACH", "Cheque", "Credit Card", "Cash", "Reinvestment"]


def _build_payload(
    account_id: str,
    amount: float,
    currency: str,
    payment_format: str,
    counterparty_account: str,
    timestamp: str,
) -> dict[str, Any]:
    return {
        "account_id": account_id,
        "events": [
            {
                "type": "wire",
                "amount": amount,
                "currency": currency,
                "amount_received": amount,
                "receiving_currency": currency,
                "payment_format": payment_format,
                "counterparty_account": counterparty_account or None,
                "timestamp": timestamp,
            }
        ],
    }


def score_account(
    account_id: str,
    amount: float,
    currency: str,
    payment_format: str,
    counterparty_account: str,
    timestamp: str,
) -> tuple[float, str, dict[str, float]]:
    """Calls `/score` and returns `(score, decision, top-5 SHAP values)`."""
    payload = _build_payload(
        account_id, amount, currency, payment_format, counterparty_account, timestamp
    )
    resp = httpx.post(f"{API_URL}/score", json=payload, timeout=30.0)
    resp.raise_for_status()
    body = resp.json()
    return body["score"], body["decision"], body["shap_values"]


def whatif_score(
    account_id: str,
    amount: float,
    currency: str,
    payment_format: str,
    counterparty_account: str,
    timestamp: str,
    new_amount: float,
    new_payment_format: str,
    new_counterparty_account: str,
) -> tuple[float, float, float]:
    """Scores the original and modified transaction and returns `(original,
    modified, delta)` AML risk scores."""
    original_score, _, _ = score_account(
        account_id, amount, currency, payment_format, counterparty_account, timestamp
    )
    new_score, _, _ = score_account(
        account_id, new_amount, currency, new_payment_format, new_counterparty_account, timestamp
    )
    return original_score, new_score, round(new_score - original_score, 4)


def render_graph(
    account_id: str,
    amount: float,
    currency: str,
    payment_format: str,
    counterparty_account: str,
    timestamp: str,
) -> str:
    """Calls `/explain/graph` and renders the 1-hop neighbourhood with pyvis."""
    payload = _build_payload(
        account_id, amount, currency, payment_format, counterparty_account, timestamp
    )
    resp = httpx.post(f"{API_URL}/explain/graph", json=payload, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    net = Network(height="400px", width="100%", directed=True, notebook=False)
    for node in data["nodes"]:
        is_root = node == data["account_id"]
        net.add_node(node, label=node, color="#e74c3c" if is_root else "#3498db")
    for edge in data["edges"]:
        net.add_edge(
            edge["source"],
            edge["target"],
            value=max(edge["weight"], 0.05),
            title=f"score={edge['score']}, weight={edge['weight']}",
        )
    return net.generate_html(notebook=False)


def latest_drift_report() -> str:
    """Returns the most recently written Evidently drift report's HTML, or a
    placeholder if none have been generated yet."""
    if DRIFT_REPORT_DIR.exists():
        reports = sorted(DRIFT_REPORT_DIR.glob("drift_*.html"))
        if reports:
            return reports[-1].read_text()
    return (
        "<p>No drift reports found yet. Run <code>scripts/stream_consumer.py</code> "
        "(or <code>src.monitoring.drift.run_drift_check</code> directly) to generate one.</p>"
    )


def build_app() -> gr.Blocks:
    default_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")

    with gr.Blocks(title="PRAGMA-G — AML Explainability Demo") as demo:
        gr.Markdown("# PRAGMA-G — AML Risk Explainability Demo")
        gr.Markdown(
            f"Connected to the PRAGMA-G API at `{API_URL}`. Graph-Temporal extension of "
            "Revolut's PRAGMA foundation model (arXiv:2604.08649)."
        )

        with gr.Tab("Score + Explain"):
            with gr.Row():
                with gr.Column():
                    account_id = gr.Textbox(label="Account ID", value="ACC_1234")
                    amount = gr.Number(label="Amount", value=9500.0)
                    currency = gr.Textbox(label="Currency", value="US Dollar")
                    payment_format = gr.Dropdown(
                        label="Payment Format", choices=PAYMENT_FORMATS, value="Wire"
                    )
                    counterparty_account = gr.Textbox(
                        label="Counterparty Account", value="ACC_5678"
                    )
                    timestamp = gr.Textbox(label="Timestamp (ISO 8601)", value=default_timestamp)
                    score_btn = gr.Button("Score", variant="primary")
                with gr.Column():
                    score_out = gr.Number(label="AML Risk Score")
                    decision_out = gr.Textbox(label="Decision")
                    shap_out = gr.JSON(label="Top-5 SHAP Attributions")

            score_btn.click(
                score_account,
                inputs=[
                    account_id, amount, currency, payment_format, counterparty_account, timestamp,
                ],
                outputs=[score_out, decision_out, shap_out],
            )

        with gr.Tab("What-If"):
            gr.Markdown("Modify a transaction and see how the AML risk score changes.")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Original transaction**")
                    wi_account_id = gr.Textbox(label="Account ID", value="ACC_1234")
                    wi_amount = gr.Number(label="Amount", value=100.0)
                    wi_currency = gr.Textbox(label="Currency", value="US Dollar")
                    wi_payment_format = gr.Dropdown(
                        label="Payment Format", choices=PAYMENT_FORMATS, value="Wire"
                    )
                    wi_counterparty = gr.Textbox(label="Counterparty Account", value="ACC_5678")
                    wi_timestamp = gr.Textbox(
                        label="Timestamp (ISO 8601)", value=default_timestamp
                    )
                with gr.Column():
                    gr.Markdown("**Modified transaction**")
                    new_amount = gr.Number(label="New Amount", value=10000.0)
                    new_payment_format = gr.Dropdown(
                        label="New Payment Format", choices=PAYMENT_FORMATS, value="Wire"
                    )
                    new_counterparty = gr.Textbox(
                        label="New Counterparty Account", value="ACC_5678"
                    )
            whatif_btn = gr.Button("Compare", variant="primary")
            with gr.Row():
                orig_score_out = gr.Number(label="Original Score")
                new_score_out = gr.Number(label="New Score")
                delta_out = gr.Number(label="Score Δ")

            whatif_btn.click(
                whatif_score,
                inputs=[
                    wi_account_id, wi_amount, wi_currency, wi_payment_format,
                    wi_counterparty, wi_timestamp,
                    new_amount, new_payment_format, new_counterparty,
                ],
                outputs=[orig_score_out, new_score_out, delta_out],
            )

        with gr.Tab("Transaction Graph"):
            gr.Markdown(
                "Visualise the account's 1-hop transaction-graph neighbourhood, with "
                "a risk score for each transaction (edge)."
            )
            with gr.Row():
                g_account_id = gr.Textbox(label="Account ID", value="ACC_1234")
                g_amount = gr.Number(label="Amount", value=9500.0)
                g_currency = gr.Textbox(label="Currency", value="US Dollar")
                g_payment_format = gr.Dropdown(
                    label="Payment Format", choices=PAYMENT_FORMATS, value="Wire"
                )
                g_counterparty = gr.Textbox(label="Counterparty Account", value="ACC_5678")
                g_timestamp = gr.Textbox(label="Timestamp (ISO 8601)", value=default_timestamp)
            graph_btn = gr.Button("Visualise", variant="primary")
            graph_out = gr.HTML()

            graph_btn.click(
                render_graph,
                inputs=[
                    g_account_id, g_amount, g_currency, g_payment_format,
                    g_counterparty, g_timestamp,
                ],
                outputs=[graph_out],
            )

        with gr.Tab("Drift Monitor"):
            gr.Markdown("Latest Evidently drift report from live scoring traffic.")
            drift_btn = gr.Button("Load latest report")
            drift_out = gr.HTML()
            drift_btn.click(latest_drift_report, outputs=[drift_out])

    return demo


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7860)

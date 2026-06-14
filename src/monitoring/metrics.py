"""Prometheus metrics for PRAGMA-G serving (PLAN.md Week 6/7/9).

`record_request` is called once per `/score` and `/whatif` request. The
`pragma_g_drift_detected` gauge is set by `src.monitoring.drift.run_drift_check`,
called periodically by `scripts/stream_consumer.py`. `MODEL_VERSION_REQUESTS`
is incremented per request with the `model_version` label of the model that
served it, for the `/score?model=v1|v2` A/B scaffold (PLAN.md Week 9).
"""
from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

REQUEST_COUNT = Counter(
    "pragma_g_requests_total", "Total number of /score requests served"
)

REQUEST_LATENCY = Histogram(
    "pragma_g_latency_seconds", "End-to-end /score request latency in seconds"
)

SCORE_HISTOGRAM = Histogram(
    "pragma_g_score_histogram",
    "Distribution of AML risk scores returned by /score",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

DRIFT_DETECTED = Gauge(
    "pragma_g_drift_detected",
    "1 if the most recent Evidently drift check flagged dataset-level drift, else 0",
)

MODEL_VERSION_REQUESTS = Counter(
    "pragma_g_model_version_requests_total",
    "Total number of requests served, by model version (A/B routing)",
    ["model_version"],
)


def record_request(latency_ms: float, score: float) -> None:
    """Records one scored request's latency (ms) and risk score."""
    REQUEST_COUNT.inc()
    REQUEST_LATENCY.observe(latency_ms / 1000.0)
    SCORE_HISTOGRAM.observe(score)


def record_model_version(model_version: str) -> None:
    """Records which model version served a request (A/B routing)."""
    MODEL_VERSION_REQUESTS.labels(model_version=model_version).inc()

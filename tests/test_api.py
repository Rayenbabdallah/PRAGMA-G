"""Tests for the PRAGMA-G FastAPI serving layer (PLAN.md Week 6)."""
from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from src.api.main import app

VALID_PAYLOAD = {
    "account_id": "ACC_1234",
    "profile": {"plan": "standard", "region": "uk", "balance_quantile": 0.6},
    "events": [
        {
            "type": "wire",
            "amount": 9500.00,
            "currency": "US Dollar",
            "amount_received": 9500.00,
            "receiving_currency": "US Dollar",
            "payment_format": "Wire",
            "counterparty_account": "ACC_5678",
            "counterparty_bank": "20",
            "timestamp": "2026-06-01T14:32:00",
        }
    ],
}


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(app) as c:
        yield c


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_version" in body


def test_score(client: TestClient) -> None:
    response = client.post("/score", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert body["account_id"] == "ACC_1234"
    assert 0.0 <= body["score"] <= 1.0
    assert body["decision"] in {"flag", "review", "clear"}
    assert len(body["shap_values"]) == 5
    assert body["graph_neighbours"] == 1
    assert body["latency_ms"] >= 0


def test_score_no_counterparty(client: TestClient) -> None:
    payload = {
        "account_id": "ACC_0001",
        "events": [
            {
                "type": "card_payment",
                "amount": 42.0,
                "currency": "Euro",
                "timestamp": "2026-01-01T00:00:00",
            }
        ],
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    assert response.json()["graph_neighbours"] == 0


def test_score_requires_events(client: TestClient) -> None:
    response = client.post("/score", json={"account_id": "ACC_0001", "events": []})
    assert response.status_code == 422


def test_explain(client: TestClient) -> None:
    response = client.post("/explain", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body["all_shap_values"]) == {
        "amount_paid",
        "payment_format",
        "receiving_currency",
        "payment_currency",
        "graph_fan_in_ratio",
        "temporal_velocity",
    }
    assert body["graph_neighbourhood"] == [{"account_id": "ACC_5678", "relation": "counterparty"}]


def test_whatif(client: TestClient) -> None:
    response = client.post("/whatif", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert 0.0 <= response.json()["score"] <= 1.0


def test_metrics(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert b"pragma_g_requests_total" in response.content


def test_score_model_v2(client: TestClient) -> None:
    """`/score?model=v2` routes to the `v2` model and is reflected in the response."""
    response = client.post("/score", json=VALID_PAYLOAD, params={"model": "v2"})
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["score"] <= 1.0
    assert "v2" in body["model_version"]


def test_score_unknown_model(client: TestClient) -> None:
    response = client.post("/score", json=VALID_PAYLOAD, params={"model": "v3"})
    assert response.status_code == 400


def test_metrics_model_version_label(client: TestClient) -> None:
    client.post("/score", json=VALID_PAYLOAD, params={"model": "v1"})
    response = client.get("/metrics")
    assert b"pragma_g_model_version_requests_total" in response.content

"""Request and response schemas for PRAGMA-G API."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class EventRecord(BaseModel):
    """A single banking event (transaction, app event, communication)."""
    type: str = Field(..., description="Event type: wire, card_payment, p2p_transfer, etc.")
    amount: float | None = Field(None, description="Amount in payment currency")
    currency: str | None = Field(None, description="Payment currency code (ISO 4217)")
    amount_received: float | None = Field(None, description="Amount in receiving currency")
    receiving_currency: str | None = Field(None, description="Receiving currency code")
    payment_format: str | None = Field(None, description="Wire, Cheque, Credit Card, etc.")
    counterparty_bank: str | None = Field(None, description="Counterparty bank identifier")
    counterparty_account: str | None = Field(None, description="Counterparty account ID")
    timestamp: datetime = Field(..., description="Event datetime (ISO 8601)")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "wire",
                "amount": 9500.00,
                "currency": "USD",
                "payment_format": "Wire",
                "counterparty_bank": "BankB",
                "counterparty_account": "ACC_5678",
                "timestamp": "2026-06-01T14:32:00",
            }
        }


class ProfileState(BaseModel):
    """Static/slow-changing account attributes (profile branch of PRAGMA)."""
    plan: str | None = Field(None, description="Account plan: standard, premium, metal")
    region: str | None = Field(None, description="Service region code")
    balance_quantile: float | None = Field(
        None, ge=0.0, le=1.0,
        description="Account balance percentile [0, 1]"
    )
    account_age_days: int | None = Field(None, ge=0, description="Days since account creation")
    currency: str | None = Field(None, description="Primary account currency")


class TransactionRequest(BaseModel):
    """Request to score an account for AML risk."""
    account_id: str = Field(..., description="Account identifier")
    profile: ProfileState = Field(default_factory=ProfileState)
    events: list[EventRecord] = Field(
        ..., min_length=1,
        description="Recent event history (most recent last)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "account_id": "ACC_1234",
                "profile": {
                    "plan": "standard",
                    "region": "uk",
                    "balance_quantile": 0.6,
                    "account_age_days": 120,
                },
                "events": [
                    {
                        "type": "wire",
                        "amount": 9500.00,
                        "currency": "USD",
                        "payment_format": "Wire",
                        "counterparty_account": "ACC_5678",
                        "timestamp": "2026-06-01T14:32:00",
                    }
                ],
            }
        }


class ScoreResponse(BaseModel):
    """AML risk score with decision and explainability."""
    account_id: str
    score: float = Field(..., ge=0.0, le=1.0, description="AML risk score ∈ [0, 1]")
    decision: str = Field(..., description="flag | review | clear")
    threshold_version: str = Field(..., description="Threshold config version used")
    shap_values: dict[str, float] = Field(
        ..., description="Top-5 SHAP feature attributions"
    )
    graph_neighbours: int = Field(
        ..., description="Number of accounts in 1-hop transaction graph neighbourhood"
    )
    latency_ms: float = Field(..., description="End-to-end inference latency in ms")
    model_version: str = Field(..., description="Model version from MLflow registry")

    class Config:
        json_schema_extra = {
            "example": {
                "account_id": "ACC_1234",
                "score": 0.847,
                "decision": "flag",
                "threshold_version": "v1.0",
                "shap_values": {
                    "amount_paid": 0.31,
                    "payment_format": 0.18,
                    "graph_fan_in_ratio": 0.15,
                    "temporal_velocity": 0.12,
                    "receiving_currency": 0.09,
                },
                "graph_neighbours": 3,
                "latency_ms": 47.3,
                "model_version": "pragma-g-aml-v1",
            }
        }


class ExplainResponse(BaseModel):
    """Detailed explainability response (all SHAP values + graph neighbourhood)."""
    account_id: str
    score: float
    all_shap_values: dict[str, float] = Field(
        ..., description="All feature SHAP attributions (not just top-5)"
    )
    graph_neighbourhood: list[dict[str, Any]] = Field(
        ..., description="2-hop graph neighbourhood with per-node scores"
    )
    model_version: str

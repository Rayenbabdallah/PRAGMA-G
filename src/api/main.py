"""PRAGMA-G FastAPI serving layer."""
from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from src.api.model_loader import ModelLoader
from src.api.schemas import ExplainResponse, ScoreResponse, TransactionRequest
from src.monitoring.metrics import record_model_version, record_request

MODEL_VERSIONS = {"v1", "v2"}

# Singleton model loader
_loader: ModelLoader | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load model on startup, clean up on shutdown."""
    global _loader
    app.state.start_time = time.time()
    logger.info("Loading PRAGMA-G model...")
    _loader = ModelLoader()
    _loader.load()
    _loader.warmup(n=10)
    logger.info(f"Model ready. Version: {_loader.model_version}")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="PRAGMA-G",
    description=(
        "Graph-Temporal extension of Revolut's PRAGMA foundation model. "
        "Fixes the 47.1% AML performance gap identified in arXiv:2604.08649 "
        "by adding GraphSAGE over the account transaction graph."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check — always returns 200 if the service is up."""
    return {
        "status": "ok",
        "model_version": _loader.model_version if _loader else "not_loaded",
        "uptime_seconds": time.time() - app.state.start_time
            if hasattr(app.state, "start_time") else 0,
    }


@app.post("/score", response_model=ScoreResponse)
async def score(request: TransactionRequest, model: str = "v1") -> ScoreResponse:
    """
    Score an account for AML risk.

    Returns a risk score ∈ [0, 1] with:
    - decision: flag (>0.70) / review (0.30–0.70) / clear (<0.30)
    - shap_values: top-5 feature attributions
    - latency_ms: end-to-end inference time

    `model`: `v1` (default, Production) or `v2` (Staging) — A/B routing scaffold.
    """
    if _loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if model not in MODEL_VERSIONS:
        raise HTTPException(status_code=400, detail=f"Unknown model version: {model!r}")

    t0 = time.perf_counter()
    try:
        result = _loader.score(request, model=model)
    except Exception as e:
        logger.error(f"Scoring error for account {request.account_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.perf_counter() - t0) * 1000
    record_request(latency_ms, result.score)
    record_model_version(result.model_version)

    return ScoreResponse(
        account_id=request.account_id,
        score=result.score,
        decision=result.decision,
        threshold_version=result.threshold_version,
        shap_values=result.shap_values,
        graph_neighbours=result.graph_neighbours,
        latency_ms=round(latency_ms, 1),
        model_version=result.model_version,
    )


@app.post("/explain", response_model=ExplainResponse)
async def explain(request: TransactionRequest, model: str = "v1") -> ExplainResponse:
    """
    Detailed SHAP explanation for an account's AML score.
    Returns all feature attributions, not just top-5.
    """
    if _loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if model not in MODEL_VERSIONS:
        raise HTTPException(status_code=400, detail=f"Unknown model version: {model!r}")

    result = _loader.explain(request, model=model)
    return ExplainResponse(
        account_id=request.account_id,
        score=result.score,
        all_shap_values=result.all_shap_values,
        graph_neighbourhood=result.graph_neighbourhood,
        model_version=result.model_version,
    )


@app.post("/whatif", response_model=ScoreResponse)
async def whatif(request: TransactionRequest, model: str = "v1") -> ScoreResponse:
    """
    What-If analysis: modify a transaction and see how the score changes.
    Used by compliance analysts to understand model decisions.
    Same as /score but logs that this was a counterfactual query.
    """
    if _loader is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if model not in MODEL_VERSIONS:
        raise HTTPException(status_code=400, detail=f"Unknown model version: {model!r}")

    t0 = time.perf_counter()
    result = _loader.score(request, counterfactual=True, model=model)
    latency_ms = (time.perf_counter() - t0) * 1000

    return ScoreResponse(
        account_id=request.account_id,
        score=result.score,
        decision=result.decision,
        threshold_version=result.threshold_version,
        shap_values=result.shap_values,
        graph_neighbours=result.graph_neighbours,
        latency_ms=round(latency_ms, 1),
        model_version=result.model_version,
    )


@app.get("/metrics")
async def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

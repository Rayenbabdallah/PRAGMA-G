"""MLflow model registry helpers for PRAGMA-G (PLAN.md Week 9).

`register_checkpoint` is called by `src.training.finetune` after a fine-tuning
run to register the trained `pragma_mini`/`classifier` pair as a new version
of the `mlflow.model_registry_name` registered model (default stage:
"Staging"). `load_registry_model` is called by `src.api.model_loader.ModelLoader`
to load the `Production`/`Staging`-stage version at serving time.

`load_registry_model` returns `None` (rather than raising) if the tracking
server is unreachable or no version exists in the requested stage, so the API
can fall back to a local checkpoint or fresh weights — see `ModelLoader.load`.
"""
from __future__ import annotations

import mlflow
from mlflow.tracking import MlflowClient
from torch import nn


def register_checkpoint(
    run_id: str,
    model_name: str,
    tracking_uri: str | None = None,
    stage: str = "Staging",
) -> str:
    """Registers `runs:/{run_id}/pragma_mini` as a new version of `model_name`
    and transitions it to `stage`. Returns the new version number as a string.

    Assumes `mlflow.pytorch.log_model` was called for both `pragma_mini` and
    `classifier` (under those artifact paths) within the run `run_id`.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    model_version = mlflow.register_model(f"runs:/{run_id}/pragma_mini", model_name)
    MlflowClient().transition_model_version_stage(
        name=model_name, version=model_version.version, stage=stage
    )
    return str(model_version.version)


def load_registry_model(
    model_name: str,
    stage: str = "Production",
    tracking_uri: str | None = None,
) -> tuple[nn.Module, nn.Module, str] | None:
    """Loads the `{stage}`-stage `pragma_mini` + `classifier` pair for
    `model_name` from the MLflow model registry.

    Returns `(pragma_mini, classifier, version)`, or `None` if the tracking
    server is unreachable or no version exists in `stage`.
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    try:
        client = MlflowClient()
        versions = client.get_latest_versions(model_name, stages=[stage])
        if not versions:
            return None
        version = versions[0]
        pragma_mini = mlflow.pytorch.load_model(f"models:/{model_name}/{stage}")
        classifier = mlflow.pytorch.load_model(f"runs:/{version.run_id}/classifier")
        return pragma_mini, classifier, str(version.version)
    except Exception:
        return None

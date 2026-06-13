"""AML classification metrics (ARCHITECTURE.md §5, PLAN.md Week 5).

PR-AUC is the primary metric under class imbalance (~5% illicit); ROC-AUC,
precision@recall=0.5, and a false-negative-weighted cost metric are tracked
as secondary diagnostics.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def precision_at_recall(
    y_true: np.ndarray, y_score: np.ndarray, recall_target: float = 0.5
) -> float:
    """Highest precision achieved at recall >= `recall_target`."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    eligible = precision[recall >= recall_target]
    if eligible.size == 0:
        return 0.0
    return float(eligible.max())


def cost_based_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    cost_fp: float = 1.0,
    cost_fn: float = 100.0,
) -> float:
    """`cost = FP * cost_fp + FN * cost_fn` at a fixed decision threshold."""
    y_pred = (y_score >= threshold).astype(int)
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return float(fp * cost_fp + fn * cost_fn)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    """Computes PR-AUC, ROC-AUC, precision@recall=0.5, and the cost-based metric.

    Returns `pr_auc=0.0`/`roc_auc=0.5` if `y_true` has only one class (degenerate
    on tiny synthetic batches) instead of raising.
    """
    if len(np.unique(y_true)) < 2:
        return {
            "pr_auc": 0.0,
            "roc_auc": 0.5,
            "precision_at_recall_0.5": 0.0,
            "cost": cost_based_metric(y_true, y_score),
        }
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "precision_at_recall_0.5": precision_at_recall(y_true, y_score),
        "cost": cost_based_metric(y_true, y_score),
    }

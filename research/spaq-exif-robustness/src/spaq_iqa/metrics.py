"""Regression metrics used by every SPAQ experiment."""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    targets = np.asarray(targets, dtype=float).reshape(-1)
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if targets.shape != predictions.shape or targets.size < 2:
        raise ValueError("Targets and predictions must have equal shape and at least two values")
    if not np.isfinite(targets).all() or not np.isfinite(predictions).all():
        raise ValueError("Metrics require finite values")
    return {
        "srcc": float(spearmanr(targets, predictions).statistic),
        "plcc": float(pearsonr(targets, predictions).statistic),
        "mae": float(np.mean(np.abs(targets - predictions))),
    }

"""Competition metric and regression scores."""

from __future__ import annotations

import numpy as np


def compute_sw_score(y_true: float, y_pred: float) -> float:
    """Score one prediction: full credit within 5 % error, none beyond 20 %."""
    relative_error = abs(y_true - y_pred) / y_true
    if relative_error < 0.05:
        return 100.0
    if relative_error > 0.20:
        return 0.0
    return 100 - (relative_error - 0.05) / 0.15 * 100


def sw_scores(y_true, y_pred) -> np.ndarray:
    """Per-sample scores for two equally shaped arrays."""
    return np.array([compute_sw_score(t, p) for t, p in zip(y_true, y_pred)], dtype=np.float64)


def evaluate_sw_score(y_true, y_pred) -> float:
    """Mean SW score, accepting scalars or 1-D arrays."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 0 and y_pred.ndim == 0:
        return compute_sw_score(float(y_true), float(y_pred))
    if y_true.ndim == 1 and y_pred.ndim == 1:
        return float(np.mean(sw_scores(y_true, y_pred)))
    raise ValueError("y_true and y_pred must both be scalars or 1-D arrays")

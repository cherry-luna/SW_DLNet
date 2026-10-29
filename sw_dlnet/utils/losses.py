"""Training objectives."""
from __future__ import annotations

import torch


def mape_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    epsilon: float = 1e-8,
) -> torch.Tensor:
    """Mean absolute percentage error in percent.

    The ground truth sits in the denominator, matching the competition
    metric; ``epsilon`` guards against a zero width.
    """
    return torch.mean(torch.abs((y_true - y_pred) / (y_true + epsilon))) * 100
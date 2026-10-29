"""Competition metric and loss behaviour."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sw_dlnet.utils import (
    compute_sw_score,
    evaluate_sw_score,
    mape_loss,
    sw_scores,
)


@pytest.mark.parametrize(
    ("y_true", "y_pred", "expected"),
    [
        (1.0, 1.0, 100.0),
        (1.0, 1.04, 100.0),
        (1.0, 1.10, 100.0 - (0.10 - 0.05) / 0.15 * 100),
        (1.0, 1.20, 0.0),
        (1.0, 1.30, 0.0),
        (0.5, 0.55, 100.0 - (0.10 - 0.05) / 0.15 * 100),
    ],
)
def test_compute_sw_score(y_true, y_pred, expected):
    assert compute_sw_score(y_true, y_pred) == pytest.approx(expected)


def test_sw_scores_matches_elementwise_calls():
    y_true = np.array([0.5, 0.8, 1.0])
    y_pred = np.array([0.51, 0.7, 1.25])
    expected = [compute_sw_score(t, p) for t, p in zip(y_true, y_pred)]
    np.testing.assert_allclose(sw_scores(y_true, y_pred), expected)


def test_evaluate_sw_score_averages_samples():
    y_true = np.array([0.5, 0.8, 1.0])
    y_pred = np.array([0.51, 0.7, 1.25])
    assert evaluate_sw_score(y_true, y_pred) == pytest.approx(
        float(np.mean([compute_sw_score(t, p) for t, p in zip(y_true, y_pred)]))
    )


def test_evaluate_sw_score_accepts_scalars():
    assert evaluate_sw_score(1.0, 1.0) == 100.0
    assert evaluate_sw_score(np.float32(1.0), np.float32(1.0)) == 100.0


def test_evaluate_sw_score_rejects_2d_input():
    with pytest.raises(ValueError):
        evaluate_sw_score(np.zeros((2, 2)), np.zeros((2, 2)))


def test_mape_loss_uses_ground_truth_in_denominator():
    y_true = torch.tensor([1.0, 2.0])
    y_pred = torch.tensor([1.1, 1.8])
    expected = (0.1 / 1.0 + 0.2 / 2.0) / 2 * 100
    assert mape_loss(y_pred, y_true).item() == pytest.approx(expected, rel=1e-5)


def test_mape_loss_is_zero_for_perfect_predictions():
    y = torch.tensor([0.4, 0.9])
    assert mape_loss(y, y).item() == pytest.approx(0.0, abs=1e-6)

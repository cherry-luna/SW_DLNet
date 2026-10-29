"""Phase-correction behaviour."""

from __future__ import annotations

import numpy as np

from sw_dlnet.data import correct_iq_phase


def _random_signal(seed: int = 0, length: int = 256) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return rng.normal(size=length), rng.normal(size=length)


def _estimated_phase(i: np.ndarray, q: np.ndarray) -> float:
    complex_pilot = i[:100] + 1j * q[:100]
    return float(np.angle(np.sum(complex_pilot * np.conj(complex_pilot[0]))))


def test_matches_the_documented_rotation_matrix():
    i, q = _random_signal()
    phase = _estimated_phase(i, q)
    rotation = np.array([[np.cos(phase), np.sin(phase)], [-np.sin(phase), np.cos(phase)]])
    expected = rotation @ np.stack([i, q], axis=0)

    corrected_i, corrected_q = correct_iq_phase((i, q))
    np.testing.assert_allclose(corrected_i, expected[0], atol=1e-12)
    np.testing.assert_allclose(corrected_q, expected[1], atol=1e-12)


def test_rotation_preserves_magnitude():
    i, q = _random_signal(seed=1)
    corrected_i, corrected_q = correct_iq_phase((i, q))
    np.testing.assert_allclose(np.hypot(corrected_i, corrected_q), np.hypot(i, q), atol=1e-12)


def test_zero_pilot_leaves_signal_untouched():
    i = np.zeros(200)
    q = np.zeros(200)
    q[150:] = 1.5
    corrected_i, corrected_q = correct_iq_phase((i, q))
    np.testing.assert_allclose(corrected_i, i, atol=1e-12)
    np.testing.assert_allclose(corrected_q, q, atol=1e-12)


def test_tuple_in_returns_tuple_out():
    i, q = _random_signal(seed=2)
    result = correct_iq_phase((i, q))
    assert isinstance(result, tuple) and len(result) == 2
    assert all(isinstance(part, np.ndarray) for part in result)


def test_array_in_returns_array_out():
    i, q = _random_signal(seed=3)
    result = correct_iq_phase(np.stack([i, q], axis=0))
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, i.size)

"""I/Q preprocessing utilities."""
from __future__ import annotations

import numpy as np


def correct_iq_phase(iq_signal):
    """Estimate the common phase from the first 100 samples and rotate it out."""
    if isinstance(iq_signal, tuple):
        iq = np.stack(iq_signal, axis=0)
    else:
        iq = iq_signal

    pilot = iq[:, :100]
    complex_pilot = pilot[0] + 1j * pilot[1]

    if np.any(np.abs(complex_pilot) > 1e-6):
        phase = np.angle(np.sum(complex_pilot * np.conj(complex_pilot[0])))
    else:
        phase = 0.0

    rotation = np.array(
        [
            [np.cos(phase), np.sin(phase)],
            [-np.sin(phase), np.cos(phase)],
        ]
    )
    corrected = rotation @ iq

    if isinstance(iq_signal, tuple):
        return corrected[0], corrected[1]
    return corrected
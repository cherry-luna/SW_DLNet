import numpy as np


def correct_iq_phase(iq_signal):
    """基于前导训练序列的相位估计"""
    # 取前100个采样点估计相位
    pilot_samples = iq_signal[:, :100]
    complex_pilot = pilot_samples[0] + 1j * pilot_samples[1]

    if np.any(np.abs(complex_pilot) > 1e-6):
        # 使用最小二乘估计
        phase = np.angle(np.sum(complex_pilot * np.conj(complex_pilot[0])))
    else:
        phase = 0.0

    rotation_matrix = np.array([
        [np.cos(phase), np.sin(phase)],
        [-np.sin(phase), np.cos(phase)]
    ])
    return rotation_matrix @ iq_signal

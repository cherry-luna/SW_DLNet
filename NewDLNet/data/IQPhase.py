import numpy as np


def correct_iq_phase(iq_signal):
    """基于前导训练序列的相位估计"""
    if isinstance(iq_signal, tuple):
        # 将元组 (i_data, q_data) 转换为二维数组 [[i_data], [q_data]]
        iq = np.stack(iq_signal, axis=0)  # 形状 (2, N)
    else:
        iq = iq_signal

    # 取前100个采样点估计相位
    pilot_samples = iq[:, :100]
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

    if isinstance(iq_signal, tuple):
        corrected_iq = rotation_matrix @ iq
        return corrected_iq[0], corrected_iq[1]
    else:
        return rotation_matrix @ iq


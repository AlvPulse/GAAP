import numpy as np
from beamformer.utils import oversampled_fft

def get_metrics(c_n, task, element, oversample_factor=8):
    """
    Returns (null_depth_db, main_beam_gain_db, ptnr_db)
    """
    N = len(c_n)
    max_ideal_power_db = 20 * np.log10(N)

    u_grid, F = oversampled_fft(c_n, oversample_factor)
    power_db = 20 * np.log10(np.abs(F) + 1e-12)

    # Null Depth (absolute)
    null_depth_abs = -40
    if hasattr(task, 'null_angles') and task.null_angles:
        null_power = []
        for angle in task.null_angles:
            idx = np.argmin(np.abs(u_grid - angle))
            null_power.append(power_db[idx])
        null_depth_abs = np.mean(null_power)

    # Gain (absolute)
    main_beam_gain_abs = max_ideal_power_db
    if task.target_angles:
        target_power = []
        for angle in task.target_angles:
            idx = np.argmin(np.abs(u_grid - angle))
            target_power.append(power_db[idx])
        main_beam_gain_abs = np.mean(target_power)

    # PTNR = Gain - Null (in dB)
    ptnr = main_beam_gain_abs - null_depth_abs

    return null_depth_abs, main_beam_gain_abs, ptnr

def count_flips(V_curr, V_prev, threshold=2.0):
    if V_prev is None:
        return 0
    return np.sum(np.abs(V_curr - V_prev) > threshold)

def get_sll(c_n, task, oversample_factor=8):
    """
    Safely calculates the Side Lobe Level (SLL) in dB relative to the peak array factor.
    Returns absolute max SLL level found outside the main beam region.
    """
    N = len(c_n)
    u_grid, F = oversampled_fft(c_n, oversample_factor)
    power_db = 20 * np.log10(np.abs(F) + 1e-12)

    # Mask out the main beam (assume roughly +/- 0.1 around targets)
    mask = np.ones_like(u_grid, dtype=bool)
    if task.target_angles:
        for angle in task.target_angles:
            mask &= (np.abs(u_grid - angle) > 0.15)

    if np.any(mask):
        return np.max(power_db[mask])
    return -float('inf')

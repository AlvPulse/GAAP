import numpy as np
from .element_model import ElementData
from typing import List

def scan_offset_feasibility(target_weights: np.ndarray, element: ElementData, n_steps=100, projection_method='euclidean', w_phase=1.0, w_amp=0.5):
    """
    Diagnostic geometry scan (Stage A).
    Sweeps global offset delta from 0 to 2pi and evaluates geometry metrics.

    Returns a dictionary of metrics for visualization and analysis.
    """
    deltas = np.linspace(0, 2*np.pi, n_steps)
    N = len(target_weights)

    residuals = np.zeros(n_steps)
    avg_ILs = np.zeros(n_steps)
    branches = np.zeros((n_steps, N)) # Track selected voltages to infer branches

    for i, delta in enumerate(deltas):
        rotated_targets = target_weights * np.exp(1j * delta)

        sum_sq_err = 0.0
        sum_amp = 0.0

        for n in range(N):
            best_V, best_c = element.project(rotated_targets[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
            sum_sq_err += np.abs(rotated_targets[n] - best_c)**2
            sum_amp += np.abs(best_c)
            branches[i, n] = best_V

        residuals[i] = sum_sq_err
        avg_ILs[i] = sum_amp / N

    return {
        'deltas': deltas,
        'residuals': residuals,
        'average_amplitudes': avg_ILs,
        'selected_voltages': branches
    }

def compute_aperture_potential(c_n):
    """
    E_eff: Returns the average achieved amplitude (0 to 1).
    A proxy for total array gain capacity given hardware constraints.
    """
    return np.mean(np.abs(c_n))

def count_branch_flips(c_n_k, c_n_k_minus_1, V_k, V_k_minus_1):
    """
    Correctly estimates if elements jumped between physical branches on the manifold.
    A branch flip occurs when there is a significant jump in control voltage,
    but the actual achieved phase did NOT change significantly (i.e. the algorithm
    jumped to an alias phase point on the manifold).
    """
    if c_n_k_minus_1 is None or V_k_minus_1 is None:
        return 0

    voltage_diffs = np.abs(V_k - V_k_minus_1)

    # Calculate shortest angular distance to see if phase remained roughly same
    phase_k = np.angle(c_n_k)
    phase_k_minus = np.angle(c_n_k_minus_1)
    phase_diffs = np.abs(np.angle(np.exp(1j * (phase_k - phase_k_minus))))

    # A branch flip is defined as: large voltage jump (>2.0V) but small phase change (<0.5 rad)
    # meaning we hit an aliased point.
    flips = np.sum((voltage_diffs > 2.0) & (phase_diffs < 0.5))
    return flips

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

def count_branch_flips(voltages_k, voltages_k_minus_1, threshold=5.0):
    """
    Estimates if elements jumped between branches.
    Since phase wraps and is non-monotonic, a large jump in voltage for a
    small change in target phase implies a branch flip.
    """
    if voltages_k_minus_1 is None:
        return 0

    diffs = np.abs(voltages_k - voltages_k_minus_1)
    return np.sum(diffs > threshold)

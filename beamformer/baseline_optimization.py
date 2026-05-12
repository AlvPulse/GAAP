import numpy as np
from .element_model import ElementData
from .geometry import scan_offset_feasibility

def optimize_offset_only(target_weights: np.ndarray, element: ElementData, n_steps=200):
    """
    Optimizes the global offset delta to minimize insertion loss (maximize amplitude)
    without performing any iterative pattern projection.

    This provides a strong baseline to compare AP against.

    Returns:
        best_delta: The optimal offset.
        best_c_n: The achieved complex weights.
        best_V_n: The achieved voltages.
    """
    metrics = scan_offset_feasibility(target_weights, element, n_steps=n_steps)

    # We want to minimize the residual (which directly correlates with mapping
    # elements away from insertion loss hotspots).
    best_idx = np.argmin(metrics['residuals'])
    best_delta = metrics['deltas'][best_idx]

    N = len(target_weights)
    rotated_targets = target_weights * np.exp(1j * best_delta)

    best_V_n = np.zeros(N)
    best_c_n = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        best_V_n[n], best_c_n[n] = element.project(rotated_targets[n])

    return best_delta, best_c_n, best_V_n

import numpy as np
from typing import Tuple, Optional

from .utils import oversampled_fft, oversampled_ifft
from .pattern_projection import pattern_project, Task
from .element_model import ElementData
from .debug_utils import DebugLogger
from .cost_functions import evaluate_pattern_cost

def optimize_beam_ap(
    task: Task,
    element: ElementData,
    N: int,
    initial_weights: np.ndarray,
    initial_delta: float = 0.0,
    K_max: int = 50,
    tol: float = 1e-4,
    oversample_factor: int = 8,
    debug_dir: Optional[str] = None,
    use_pattern_cost: bool = False,
    projection_method: str = 'euclidean',
    w_phase: float = 1.0,
    w_amp: float = 0.5,
    enable_ap: bool = True,
    enable_local_refinement: bool = False,
    **kwargs
) -> Tuple[np.ndarray, float, np.ndarray, dict]:
    """
    Measured-manifold Alternating Projection with Geometry-Aware Offset Control.

    Returns:
        V_n: Array of selected voltages
        phi_0: Final global phase offset
        c_n: Achieved complex weights
        history: Diagnostics dictionary for visualization
    """

    # Init
    c_n = initial_weights.copy()
    delta = initial_delta

    V_n = np.zeros(N)

    # Determine optional baseline if provided for debug
    baseline_weights = kwargs.get('baseline_weights', None)

    # Warm start projection onto hardware
    for n in range(N):
        V_n[n], c_n[n] = element.project(c_n[n] * np.exp(1j * delta), method=projection_method, w_phase=w_phase, w_amp=w_amp)

    logger = None
    if debug_dir:
        logger = DebugLogger(debug_dir, task, element, N)
        # Convert ideal to coherent hardware weights for baseline
        coh_V = np.zeros(N)
        coh_c = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            coh_V[n], coh_c[n] = element.project(initial_weights[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
        logger.log_initial_state(coh_c, coh_V, baseline_weights=baseline_weights)
        logger.log_iteration(delta, float('inf'), c_n, V_n)

    history = {
        'residuals': [],
        'deltas': [],
        'voltages': [],
        'taus': [],
        'alphas': []
    }

    # Default initial damping
    tau = 0.7
    alpha = 0.7

    best_metric = float('inf')
    best_V_n = V_n.copy()
    best_delta = delta
    best_c_n = c_n.copy()

    from .geometry import compute_aperture_potential, count_branch_flips

    for k in range(K_max):
        # 1. Pattern Domain
        if enable_ap:
            # FFT (De-rotate delta so the pattern is anchored relative to the physical array)
            # The pattern should not continuously spin by delta every iteration.
            u_grid, F = oversampled_fft(c_n * np.exp(-1j * delta), oversample_factor)

            # Project pattern (enforcing task constraints like SLL and nulls)
            F_prime = pattern_project(F, u_grid, task, N)

            # Damped update in pattern domain (AP standard damping)
            F_prime_damped = (1 - tau) * F + tau * F_prime

            c_cand = oversampled_ifft(F_prime_damped, N)
        else:
            # If AP is disabled, we just try to fit the un-updated ideal weights
            # (only useful for baseline testing where delta changes but pattern does not)
            c_cand = initial_weights

        # 2. Hardware Domain Projection
        # The offset delta controls geometry by rotating the target BEFORE manifold projection.
        rotated_cand = c_cand * np.exp(1j * delta)

        V_cand = np.zeros(N)
        c_proj = np.zeros(N, dtype=np.complex128)

        current_residual = 0.0
        for n in range(N):
            V_cand[n], c_proj[n] = element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
            current_residual += np.abs(rotated_cand[n] - c_proj[n])**2

        # Damping is static since delta is fixed in Stage 0

        # 3. Damping / Stabilization
        # Damped update in hardware domain to stabilize branch jumps
        V_n = (1 - alpha) * V_n + alpha * V_cand

        # Final c_n evaluated at new voltages BEFORE local refinement
        for n in range(N):
            c_n[n] = element.get_complex_weight(V_n[n])

        # Evaluate current cost *before* checking tracking, and *before* refinement
        # (or after refinement depending on what we want to track as best, but typically we want the refined pattern)
        # We will do refinement next, so we update c_n again if refinement happens.

        # 4. Local Refinement (Architectural Hook)
        # Run AFTER AP projection and damping so we don't destroy the local repairs
        # by immediately hitting them with the Euclidean AP constraint in the same step.
        if enable_local_refinement:
            from .local_refinement import refine_local_active_set
            # We would initialize and pass af_cache here, but for now we just pass None
            V_n = refine_local_active_set(V_n, task, element, af_cache=None)

            # Re-evaluate c_n after refinement
            for n in range(N):
                c_n[n] = element.get_complex_weight(V_n[n])

        current_pattern_cost = evaluate_pattern_cost(c_n, task, oversample_factor)

        # 5. Tracking and stopping criteria
        # Store tracking info
        history['residuals'].append(current_residual)
        history['deltas'].append(delta)
        history['voltages'].append(V_n.copy())

        # NOTE: We need history to store 'weights' because we use it for branch flips
        if 'weights' not in history:
            history['weights'] = []
        history['weights'].append(c_n.copy())

        history['taus'].append(tau)
        history['alphas'].append(alpha)

        if logger:
            logger.log_iteration(delta, current_residual, c_n, V_n)

        # Track the best actual geometry/pattern quality, not just the lowest projection mismatch
        metric_to_track = current_pattern_cost if use_pattern_cost else current_residual
        if metric_to_track < best_metric:
            best_metric = metric_to_track
            best_V_n = V_n.copy()
            best_delta = delta
            best_c_n = c_n.copy()

        # Convergence check
        if k > 0:
            volt_change = np.max(np.abs(history['voltages'][-1] - history['voltages'][-2]))
            if volt_change < tol and current_residual < tol:
                break

    if logger:
        logger.generate_report()

    return best_V_n, best_delta, best_c_n, history

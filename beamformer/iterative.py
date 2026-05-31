import numpy as np
from typing import Tuple, Optional

from .utils import oversampled_fft, oversampled_ifft
from .pattern_projection import pattern_project, Task
from .element_model import ElementData
from .offset_controller import GeometryAwareOffsetController
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
    enable_offset: bool = True,
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
    controller = GeometryAwareOffsetController(initial_delta=delta, use_pattern_cost=use_pattern_cost)

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

    # NEW STEPPED APPROACH:
    # Instead of fully projecting and losing the beam, we will maintain a stronger
    # anchor to the ideal weights.

    best_residual = float('inf')
    best_V_n = V_n.copy()
    best_delta = delta
    best_c_n = c_n.copy()

    for k in range(K_max):
        # 1. Pattern Domain
        if enable_ap:
            # FFT of rotated weights
            rotated_c_n = c_n * np.exp(1j * delta)
            u_grid, F = oversampled_fft(rotated_c_n, oversample_factor)

            # Project pattern (enforcing task constraints like SLL and nulls)
            F_prime = pattern_project(F, u_grid, task, N)

            # We apply the alpha step in the pattern domain: we blend the ideal pattern
            # with the current pattern to stay closer to the ideal shape instead of wildly projecting.
            # Create ideal pattern for reference
            _, F_ideal = oversampled_fft(initial_weights * np.exp(1j * delta), oversample_factor)

        # Stepped pattern update towards ideal shape
            F_prime = (1 - alpha) * F_prime 

            # Damped update in pattern domain (AP standard damping)
            F_prime_damped = (1 - tau) * F + tau * F_prime

            c_cand = oversampled_ifft(F_prime_damped, N)
        else:
            # If AP is disabled, we just try to fit the un-updated ideal weights
            # (only useful for baseline testing where delta changes but pattern does not)
            c_cand = initial_weights * np.exp(1j * delta)

        # 2. Hardware Domain Projection
        V_cand = np.zeros(N)
        c_proj = np.zeros(N, dtype=np.complex128)

        current_residual = 0.0
        for n in range(N):
            V_cand[n], c_proj[n] = element.project(c_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
            current_residual += np.abs(c_cand[n] - c_proj[n])**2

        # 3. Local Refinement (Architectural Hook)
        if enable_local_refinement:
            from .local_refinement import refine_local_active_set
            # We would initialize and pass af_cache here, but for now we just pass None
            # until we fully implement the inner loop in later stages.
            V_cand = refine_local_active_set(V_cand, task, element, af_cache=None)

        # 4. Geometry-Aware Offset Control
        metric_for_controller = current_residual
        if use_pattern_cost:
            # We want to test if the achieved c_n so far is better.
            metric_for_controller = evaluate_pattern_cost(c_n, task, oversample_factor)

        delta_new, suggested_tau, suggested_alpha = controller.update(metric_for_controller, V_cand, k)

        if enable_offset:
            delta = delta_new
            tau = suggested_tau
            alpha = suggested_alpha

        # 5. Damping / Stabilization
        # Update voltages directly to candidates (damping is handled in pattern domain)
        V_n = V_cand

        # Final c_n evaluated at new voltages
        for n in range(N):
            c_n[n] = element.get_complex_weight(V_n[n])

        # 4. Tracking and stopping criteria
        history['residuals'].append(current_residual)
        history['deltas'].append(delta)
        history['voltages'].append(V_n.copy())
        history['taus'].append(tau)
        history['alphas'].append(alpha)

        if logger:
            logger.log_iteration(delta, current_residual, c_n, V_n)

        if current_residual < best_residual:
            best_residual = current_residual
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

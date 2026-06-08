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

    # Init controllers
    mab_controller = None
    if kwargs.get('enable_mab_hopping', False):
        from .mab_controller import UCBPhaseController
        mab_controller = UCBPhaseController(n_arms=kwargs.get('mab_arms', 16), exploration_weight=kwargs.get('mab_exploration', 2.0))

    current_sa_cost = float('inf')

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

        if current_sa_cost == float('inf'):
            current_sa_cost = current_residual

        # INNER LOOP: Local Refinement inside AP
        # We perform local refinement iteratively during the AP run, ensuring we refine
        # the weights obtained in the current geometry basin BEFORE calculating the cost.
        if enable_local_refinement:
            from .local_refinement import refine_local_active_set
            from .incremental_af import IncrementalAFCache

            # Temporary re-evaluation for baseline
            for n in range(N):
                c_n[n] = element.get_complex_weight(V_n[n])

            af_cache = IncrementalAFCache(N, task)
            af_cache.initialize(c_n)

            # Use current dynamic step size if tracked
            current_step_size = kwargs.get('current_step_size', kwargs.get('step_size', 0.05))

            V_n = refine_local_active_set(
                V_n,
                task,
                element,
                af_cache=af_cache,
                k_active=kwargs.get('k_active', 8),
                refinement_steps=kwargs.get('refinement_steps_inner', 3),
                step_size=current_step_size
            )

        # Final c_n evaluated at new voltages
        for n in range(N):
            c_n[n] = element.get_complex_weight(V_n[n])

        current_pattern_cost = evaluate_pattern_cost(c_n, task, oversample_factor)

        # 4. Tracking and stopping criteria
        # Store tracking info
        history['residuals'].append(current_residual)
        history['deltas'].append(delta)
        history['voltages'].append(V_n.copy())

        if 'weights' not in history:
            history['weights'] = []
        history['weights'].append(c_n.copy())

        history['taus'].append(tau)
        history['alphas'].append(alpha)

        if logger:
            logger.log_iteration(delta, current_residual, c_n, V_n)

        metric_to_track = current_pattern_cost if use_pattern_cost else current_residual
        if metric_to_track < best_metric:
            best_metric = metric_to_track
            best_V_n = V_n.copy()
            best_delta = delta
            best_c_n = c_n.copy()

        # Convergence check & Event-Triggered Basin Reselection
        if k > 0:
            volt_change = np.max(np.abs(history['voltages'][-1] - history['voltages'][-2]))

            # Stagnation detection based on PTNR cost, NOT just projection residual
            # because deep null quality can change even when residual doesn't
            stagnated = False

            if 'costs' not in history:
                history['costs'] = []
            history['costs'].append(current_pattern_cost)

            patience = kwargs.get('patience', 5)
            eps = kwargs.get('stagnation_eps', 1e-2)

            if k >= patience:
                # We also want to check if cost isn't improving OR if it's getting worse
                recent_cost_change = history['costs'][-patience] - history['costs'][-1]
                if recent_cost_change < eps:
                    stagnated = True
            elif volt_change < eps and abs(history['residuals'][-1] - history['residuals'][-2]) < eps:
                stagnated = True

            # To avoid rapid hopping, require a minimum number of iterations in the current basin
            min_basin_iters = kwargs.get('min_basin_iters', 3)
            iters_in_basin = k - kwargs.get('last_hop_k', 0)
            if mab_controller is not None:
                # Update bandit with current cost
                current_arm = int((delta / (2*np.pi)) * mab_controller.n_arms) % mab_controller.n_arms
                mab_controller.update(current_arm, current_pattern_cost if use_pattern_cost else current_residual)

                # We hop based on bandit only when stagnated to allow local settling
                if stagnated and iters_in_basin >= min_basin_iters:
                    next_arm = mab_controller.select_arm()
                    if next_arm != current_arm:
                        new_delta = mab_controller.get_delta_for_arm(next_arm)
                        # Add a tiny bit of jitter to not hit the exact same point
                        new_delta += np.random.uniform(-0.1, 0.1)
                        new_delta = new_delta % (2*np.pi)

                        rotated_cand = c_n * np.exp(1j * (new_delta - delta))
                        for n in range(N):
                            V_n[n], c_n[n] = element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
                        delta = new_delta
                        kwargs['current_step_size'] = kwargs.get('step_size', 0.05)
                        kwargs['last_hop_k'] = k
                        stagnated = False # Reset stagnation

            elif kwargs.get('enable_sa_hopping', False):
                from .simulated_annealing_hopping import update_sa_hopping
                if iters_in_basin >= 1: # SA hops more frequently, evaluates cost explicitly
                    new_delta, new_V, new_c, new_cost, accepted = update_sa_hopping(
                        delta, current_sa_cost, c_n, element, N, task, k, K_max,
                        projection_method, w_phase, w_amp,
                        initial_temp=kwargs.get('sa_temp', 1.0),
                        cooling_rate=kwargs.get('sa_cooling', 0.85)
                    )
                    if accepted:
                        delta = new_delta
                        V_n = new_V
                        c_n = new_c
                        current_sa_cost = new_cost
                        kwargs['current_step_size'] = kwargs.get('step_size', 0.05)
                        kwargs['last_hop_k'] = k
                        stagnated = False

            if stagnated and iters_in_basin >= min_basin_iters:
                if kwargs.get('enable_event_triggered_offset', False):
                    from .event_triggered_offset import trigger_basin_reselection

                    new_delta, new_V_n, new_c_n, changed = trigger_basin_reselection(
                        task, element, N, c_n, delta, current_pattern_cost,
                        n_candidates=kwargs.get('basin_candidates', 8),
                        probe_lr_steps=kwargs.get('probe_lr_steps', 3),
                        k_active=kwargs.get('k_active', 8),
                        step_size=kwargs.get('step_size', 0.05),
                        projection_method=projection_method,
                        w_phase=w_phase,
                        w_amp=w_amp
                    )

                    if changed:
                        delta = new_delta
                        V_n = new_V_n
                        c_n = new_c_n

                        # Reset adaptive step size globally
                        kwargs['current_step_size'] = kwargs.get('step_size', 0.05)
                        kwargs['last_hop_k'] = k

                elif kwargs.get('enable_random_hopping', False):
                    # Random basin hopping fallback
                    new_delta = np.random.uniform(0, 2 * np.pi)

                    # Project directly without probe
                    rotated_cand = c_n * np.exp(1j * new_delta)
                    for n in range(N):
                        V_n[n], c_n[n] = element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
                    delta = new_delta

                    kwargs['current_step_size'] = kwargs.get('step_size', 0.05)
                    kwargs['last_hop_k'] = k


            if volt_change < tol and current_residual < tol and not stagnated:
                # Only break if we didn't just hop
                break

    # STAGE 2: Local Refinement (Final Cleanup)
    if enable_local_refinement and kwargs.get('final_cleanup', True):
        from .local_refinement import refine_local_active_set
        from .incremental_af import IncrementalAFCache

        af_cache = IncrementalAFCache(N, task)
        af_cache.initialize(best_c_n)

        best_V_n = refine_local_active_set(
            best_V_n,
            task,
            element,
            af_cache=af_cache,
            k_active=kwargs.get('k_active', 8),
            refinement_steps=kwargs.get('refinement_steps', 10),
            step_size=kwargs.get('step_size', 0.05)
        )

        for n in range(N):
            best_c_n[n] = element.get_complex_weight(best_V_n[n])

    if logger:
        logger.generate_report()

    return best_V_n, best_delta, best_c_n, history

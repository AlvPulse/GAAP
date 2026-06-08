import numpy as np
from typing import Tuple, Optional
from .pattern_projection import Task
from .element_model import ElementData
from .incremental_af import IncrementalAFCache
from .local_refinement import refine_local_active_set

def trigger_basin_reselection(
    task: Task,
    element: ElementData,
    N: int,
    current_weights: np.ndarray,
    current_delta: float,
    current_cost: float,
    n_candidates: int = 8,
    probe_lr_steps: int = 5,
    k_active: int = 8,
    step_size: float = 0.05,
    projection_method: str = 'euclidean',
    w_phase: float = 1.0,
    w_amp: float = 0.5
) -> Tuple[float, np.ndarray, np.ndarray, bool]:
    """
    Event-triggered basin reselection with short-horizon probe rollout.

    Triggered when the AP+LR loop stagnates. It proposes `n_candidates`
    offset deltas, does a fast projection and short LR probe, and selects
    the delta that yields the best expected cost.

    Returns:
        best_delta: The new chosen offset delta.
        best_V_n: The resulting voltages from the probe.
        best_c_n: The resulting complex weights from the probe.
        changed: True if a new basin was selected.
    """

    # Generate candidate deltas (propose around previous offset with jitter)
    candidate_deltas = [current_delta]
    for _ in range(n_candidates - 1):
        # 50% chance of random hopping, 50% chance of local jitter around current delta
        if np.random.rand() > 0.5:
            candidate_deltas.append(np.random.uniform(0, 2 * np.pi))
        else:
            jitter = np.random.normal(0, np.pi/4)
            candidate_deltas.append((current_delta + jitter) % (2 * np.pi))

    # Redefine evaluate_ptnr_cost internally so we can compare apples to apples
    def evaluate_ptnr_cost(c_eval):
        from .utils import oversampled_fft
        u_grid, F = oversampled_fft(c_eval, 8)
        power_db = 20 * np.log10(np.abs(F) + 1e-12)

        cost_val = 0.0
        max_ideal_power_db = 20 * np.log10(N)
        if task.target_angles:
            target_power = []
            for angle in task.target_angles:
                idx = np.argmin(np.abs(u_grid - angle))
                target_power.append(power_db[idx])
            avg_target_power = np.mean(target_power)

            gain_loss = max_ideal_power_db - avg_target_power
            if gain_loss > 3.0:
                cost_val += 1e6 * gain_loss
            else:
                cost_val += 10.0 * gain_loss

        if hasattr(task, 'null_angles') and task.null_angles:
            null_power = []
            for angle in task.null_angles:
                idx = np.argmin(np.abs(u_grid - angle))
                null_power.append(power_db[idx])
            avg_null_power = np.mean(null_power)
            cost_val += avg_null_power

        return cost_val

    best_candidate_delta = current_delta
    best_candidate_cost = evaluate_ptnr_cost(current_weights)
    best_candidate_V_n = None
    best_candidate_c_n = current_weights.copy()

    for delta in candidate_deltas:
        # 1. Project current weights onto the new rotated manifold basin
        # It's crucial to evaluate candidates relative to the current coordinate frame.
        # So we unrotate the current frame and apply the new delta.
        delta_rel = (delta - current_delta + np.pi) % (2 * np.pi) - np.pi
        rotated_cand = current_weights * np.exp(1j * delta_rel)

        V_cand = np.zeros(N)
        c_cand = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            V_cand[n], c_cand[n] = element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)

        # 2. Short-horizon Local Refinement Probe
        if probe_lr_steps > 0 and hasattr(task, 'null_angles') and len(task.null_angles) > 0:
            af_cache = IncrementalAFCache(N, task)
            af_cache.initialize(c_cand)

            V_probe = refine_local_active_set(
                V_cand, task, element, af_cache=af_cache,
                k_active=k_active, refinement_steps=probe_lr_steps, step_size=step_size
            )

            c_probe = np.zeros(N, dtype=np.complex128)
            for n in range(N):
                c_probe[n] = element.get_complex_weight(V_probe[n])

        else:
            V_probe = V_cand
            c_probe = c_cand

        # 3. Evaluate Cost (we want to minimize cost)
        cost = evaluate_ptnr_cost(c_probe)

        if cost < best_candidate_cost:
            best_candidate_cost = cost
            best_candidate_delta = delta
            best_candidate_V_n = V_probe
            best_candidate_c_n = c_probe

    changed = (best_candidate_delta != current_delta)

    if not changed:
        # Provide current weights fallback
        best_candidate_V_n = np.zeros(N)
        for n in range(N):
            # approximate voltage
            best_candidate_V_n[n], _ = element.project(current_weights[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)

    return best_candidate_delta, best_candidate_V_n, best_candidate_c_n, changed

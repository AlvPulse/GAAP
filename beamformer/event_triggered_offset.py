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

    # Generate candidate deltas (uniformly distributed)
    candidate_deltas = np.linspace(0, 2 * np.pi, n_candidates, endpoint=False)

    best_candidate_delta = current_delta
    best_candidate_cost = current_cost
    best_candidate_V_n = None
    best_candidate_c_n = current_weights.copy()

    for delta in candidate_deltas:
        # 1. Project current weights onto the new rotated manifold basin
        rotated_cand = current_weights * np.exp(1j * delta)

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
        # Reuse IncrementalAFCache cost logic or simple evaluate
        # Peak-to-Null Ratio cost (inverse)
        from .cost_functions import evaluate_pattern_cost
        # Let's use evaluate_pattern_cost or a simplified PTNR if nulls exist

        cost = evaluate_pattern_cost(c_probe, task, oversample_factor=8)

        # We penalize gain drop? evaluate_pattern_cost might handle it, but let's
        # ensure it heavily penalizes if it loses main lobe.
        # Actually evaluate_pattern_cost in cost_functions.py might just be Euclidean distance to ideal.
        # Let's check cost_functions.py to be sure.

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

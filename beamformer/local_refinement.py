import numpy as np

from .geometry import compute_local_mobility
from .synthesis import get_steering_vector

def refine_local_active_set(V_cand, task, element, af_cache, k_active=8, refinement_steps=10, step_size=0.05):
    """
    Stage 2: Geometry-Aware Local Refinement.
    Selects the most "mobile" elements (high M_n) and performs small manifold-tangent
    corrections to improve nulls without significantly degrading the main lobe gain.

    Args:
        V_cand: Array of N candidate voltages from hardware projection.
        task: The task definition (nulls, targets).
        element: The hardware manifold data.
        af_cache: The IncrementalAFCache maintaining the array factor at sparse angles.
        k_active: Number of elements to include in the active set.

    Returns:
        V_refined: Array of N refined voltages.
    """
    if not hasattr(task, 'null_angles') or len(task.null_angles) == 0:
        # Refinement is primarily for deep null recovery
        return V_cand.copy()

    V_refined = V_cand.copy()

    # Calculate mobility for all elements
    M_n = compute_local_mobility(V_refined, element)

    # Select top-K most mobile elements
    active_set = np.argsort(M_n)[-k_active:]

    # Evaluate current c_n
    c_n = np.zeros(len(V_refined), dtype=np.complex128)
    for n in range(len(V_refined)):
        c_n[n] = element.get_complex_weight(V_refined[n])

    # We will perturb voltages in the active set using a greedy approach
    for step in range(refinement_steps):
        for idx in active_set:
            v_curr = V_refined[idx]

            # Evaluate +step
            v_plus = np.clip(v_curr + step_size, element.v_min, element.v_max)
            c_plus = element.get_complex_weight(v_plus)

            # Evaluate -step
            v_minus = np.clip(v_curr - step_size, element.v_min, element.v_max)
            c_minus = element.get_complex_weight(v_minus)

            # Simple greedy evaluation of null depth
            # (In a fully cached implementation, we'd use af_cache here.
            # For robustness and since N is small enough, we do a quick local eval)

            def evaluate_nulls(c_cand_array):
                cost = 0.0
                for nu in task.null_angles:
                    sv = get_steering_vector(len(V_refined), nu)
                    null_power = np.abs(np.sum(c_cand_array * np.conj(sv)))
                    cost += null_power
                return cost

            c_test_curr = c_n.copy()
            cost_curr = evaluate_nulls(c_test_curr)

            c_test_plus = c_n.copy()
            c_test_plus[idx] = c_plus
            cost_plus = evaluate_nulls(c_test_plus)

            c_test_minus = c_n.copy()
            c_test_minus[idx] = c_minus
            cost_minus = evaluate_nulls(c_test_minus)

            # Pick best
            if cost_plus < cost_curr and cost_plus < cost_minus:
                V_refined[idx] = v_plus
                c_n[idx] = c_plus
            elif cost_minus < cost_curr and cost_minus < cost_plus:
                V_refined[idx] = v_minus
                c_n[idx] = c_minus

    return V_refined

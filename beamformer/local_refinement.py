import numpy as np

from .geometry import compute_local_mobility
from .synthesis import get_steering_vector

def refine_local_active_set(V_cand, task, element, af_cache, k_active=8, refinement_steps=10, step_size=0.05, adaptive_decay=0.9, lr_objective='ptnr_constrained', autopsy_callback=None):
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
        step_size: Initial step size for voltage perturbation.
        adaptive_decay: Decay factor for step size upon success.
        lr_objective: The objective function ('null_only' or 'ptnr_constrained').

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

    current_step_sizes = np.ones(len(V_refined)) * step_size

    # We will perturb voltages in the active set using a greedy approach
    for step in range(refinement_steps):
        for idx in active_set:
            v_curr = V_refined[idx]
            current_step = current_step_sizes[idx]

            # Evaluate +step
            v_plus = np.clip(v_curr + current_step, element.v_min, element.v_max)
            c_plus = element.get_complex_weight(v_plus)

            # Evaluate -step
            v_minus = np.clip(v_curr - current_step, element.v_min, element.v_max)
            c_minus = element.get_complex_weight(v_minus)

            if af_cache is not None:
                # Get baseline peak power for 3dB drop check
                angles, af_vals_base = af_cache.get_af()
                baseline_peak_power = 0.0
                if task.target_angles:
                    for t in task.target_angles:
                        t_idx = np.argmin(np.abs(angles - t))
                        baseline_peak_power += np.abs(af_vals_base[t_idx])**2

                # O(K) Incremental Cache evaluation
                def evaluate_incremental(c_test):
                    af_cache.update(idx, c_test)
                    angles, af_vals = af_cache.get_af()

                    # 1. Null Power
                    null_power = 0.0
                    for nu in task.null_angles:
                        angle_idx = np.argmin(np.abs(angles - nu))
                        null_power += np.abs(af_vals[angle_idx])**2

                    # 2. Peak Power
                    peak_power = 0.0
                    cost= 0
                    if task.target_angles:
                        for t in task.target_angles:
                            t_idx = np.argmin(np.abs(angles - t))
                            peak_power += np.abs(af_vals[t_idx])**2

                        if peak_power < 0.7 * baseline_peak_power:
                            cost += 10*(baseline_peak_power- peak_power) # severe penalty
                    else:
                        peak_power = 1.0 # fallback
                    ptnr = peak_power / (null_power + 1e-10)
                    cost += null_power # Minimize null power directly

                                        # 3. Peak-to-Null Ratio (we want to maximize this, so cost is inverse)
                        # Use a small epsilon to avoid division by zero
                    


                        # 4. Gain Penalty (Penalize heavily if peak drops by > 0.5dB from baseline)
                        # 0.5 dB drop corresponds to ~0.89 in linear power ratio
                       

                    # Revert cache state
                    af_cache.update(idx, c_n[idx])
                    return cost

                cost_curr = evaluate_incremental(c_n[idx])
                cost_plus = evaluate_incremental(c_plus)
                cost_minus = evaluate_incremental(c_minus)

            else:
                # Fallback brute-force evaluation O(N)

                # Baseline peak
                baseline_peak_power = 0.0
                if task.target_angles:
                    for t in task.target_angles:
                        sv = get_steering_vector(len(V_refined), t)
                        baseline_peak_power += np.abs(np.sum(c_n * np.conj(sv)))**2

                def evaluate_metrics(c_cand_array):
                    null_power = 0.0
                    for nu in task.null_angles:
                        sv = get_steering_vector(len(V_refined), nu)
                        null_power += np.abs(np.sum(c_cand_array * np.conj(sv)))**2

                    peak_power = 0.0
                    if task.target_angles:
                        for t in task.target_angles:
                            sv = get_steering_vector(len(V_refined), t)
                            peak_power += np.abs(np.sum(c_cand_array * np.conj(sv)))**2
                    else:
                        peak_power = 1.0

                    ptnr = peak_power / (null_power + 1e-10)
                    cost = -ptnr

                    if peak_power < 0.5 * baseline_peak_power:
                        cost += 1e6

                    return cost

                c_test_curr = c_n.copy()
                cost_curr = evaluate_metrics(c_test_curr)

                c_test_plus = c_n.copy()
                c_test_plus[idx] = c_plus
                cost_plus = evaluate_metrics(c_test_plus)

                c_test_minus = c_n.copy()
                c_test_minus[idx] = c_minus
                cost_minus = evaluate_metrics(c_test_minus)

            # Pick best
            if cost_plus < cost_curr and cost_plus < cost_minus:
                V_refined[idx] = v_plus
                if af_cache is not None:
                    af_cache.update(idx, c_plus)
                c_n[idx] = c_plus
                current_step_sizes[idx] *= adaptive_decay # Optional: decay step size on success or just keep it
            elif cost_minus < cost_curr and cost_minus < cost_plus:
                V_refined[idx] = v_minus
                if af_cache is not None:
                    af_cache.update(idx, c_minus)
                c_n[idx] = c_minus
                current_step_sizes[idx] *= adaptive_decay
            else:
                # If neither step improved, reduce the step size for next time
                current_step_sizes[idx] *= 0.5

        if autopsy_callback and (step + 1) % 5 == 0:
            c_post_lr = np.zeros(len(V_refined), dtype=np.complex128)
            for n in range(len(V_refined)):
                c_post_lr[n] = element.get_complex_weight(V_refined[n])
            autopsy_callback(f'Post-LR Step {step+1}', c_post_lr)

    return V_refined

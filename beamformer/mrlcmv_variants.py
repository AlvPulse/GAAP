import time
import numpy as np
from typing import Dict, Any, List

from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.MR_LCMV_certified import get_counters, reset_counters, _bump_eval, _bump_glcp, _af
from beamformer.coherent_lcmv import _retract

# We decouple GLCP acceptance to use the worst null instead of squared sum.

def build_A_matrix(nulls, N, ridge=1e-6):
    if not nulls:
        return np.zeros((N, 0), complex), np.zeros((0, 0), complex)
    n = np.arange(N)
    A = np.column_stack([np.exp(1j * np.pi * n * um) for um in nulls])
    G = np.conj(A).T @ A
    G += ridge * np.trace(G).real / max(A.shape[1], 1) * np.eye(A.shape[1])
    return A, np.linalg.inv(G)


def gain_locked_polish_minimax(c_n, V_n, task, element, N, null_angles, rounds=6, initial_gain=None, gain_floor_ratio=0.995):
    """
    Minimax GLCP variant tracking Worst-Null depth.
    Strict residual descent: rejects trades of deep nulls for shallower ones.
    """
    if not null_angles:
        return V_n.copy(), c_n.copy()

    if hasattr(element, "get_c_grid_all"):
        c_grid, V_grid = element.get_c_grid_all(), element.get_v_grid_all()
    else:
        c_grid, V_grid = np.asarray(element.c_grid), np.asarray(element.V_grid)
    u_t = task.target_angles[0]

    # Calculate baseline
    current_c = c_n.copy()
    current_V = V_n.copy()

    _bump_eval()
    baseline_gain_lin = abs(_af(current_c, u_t, N)) / N
    min_gain = (initial_gain if initial_gain else baseline_gain_lin) * gain_floor_ratio

    def get_worst_null(c):
        worst = 0.0
        for nu in null_angles:
            val = abs(_af(c, nu, N))
            if val > worst:
                worst = val
        return worst

    _bump_eval()
    best_worst_null = get_worst_null(current_c)

    # GLCP iterative sweeps
    for r in range(rounds):
        improved_any = False
        for n in range(N):
            best_c_for_n = current_c[n]
            best_V_for_n = current_V[n]
            local_best_worst_null = best_worst_null

            for vg, cg in zip(V_grid, c_grid):
                _bump_glcp() # Count as incremental search candidate
                trial_c = current_c.copy()
                trial_c[n] = cg

                # Check gain floor
                # incremental evaluation
                g_trial = abs(_af(trial_c, u_t, N)) / N
                if g_trial < min_gain:
                    continue

                # Check worst null improvement
                wn_trial = get_worst_null(trial_c)
                if wn_trial < local_best_worst_null - 1e-12: # strict improvement
                    local_best_worst_null = wn_trial
                    best_c_for_n = cg
                    best_V_for_n = vg
                    improved_any = True

            if improved_any:
                current_c[n] = best_c_for_n
                current_V[n] = best_V_for_n
                best_worst_null = local_best_worst_null

        if not improved_any:
            break

    return current_V, current_c

def beamform_mrlcmv_variants(
    task: Task,
    element: Any,
    N: int,
    mode: str = "adaptive",
    max_dual_steps: int = 128,
    null_target_db: float = -60.0,
    gauge_samples: int = 72,
    glcp_gain_floor: float = 0.995,
    patience: int = 5,
    stagnation_tol_db: float = 0.1
) -> Dict[str, Any]:
    """
    Unified MR-LCMV solver executing one of the 3 requested philosophies:
      - 'fixed': exactly max_dual_steps, ignores convergence.
      - 'certified': certified monotonic improvement/feasibility.
      - 'adaptive': practical numerical convergence (stagnation/target reaching).
    """
    reset_counters()
    t0 = time.perf_counter()

    u_t = task.target_angles[0]
    nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []

    n_idx = np.arange(N)
    s_t = np.exp(-1j * np.pi * n_idx * u_t)

    A, Ginv = build_A_matrix(nulls, N)

    # Optional handling of per-element manifolds vs unified.
    # Assuming standard unified element_model API for now.
    V_grid, c_grid = np.asarray(element.V_grid), np.asarray(element.c_grid)

    best_global_ptnr = -np.inf
    best_global_res = None

    psis = np.linspace(0, 2*np.pi, gauge_samples, endpoint=False)

    for psi in psis:
        tgt0 = np.exp(1j * psi) * s_t
        mu = np.zeros(len(nulls), dtype=complex)

        c_n, V_n, _ = _retract(tgt0, c_grid, V_grid)
        initial_gain = abs(_af(c_n, u_t, N)) / N

        hist = []
        best_c, best_V = c_n.copy(), V_n.copy()

        _bump_eval()
        best_worst_null_db = 20 * np.log10(max([abs(_af(c_n, nu, N)) for nu in nulls]) + 1e-12) if nulls else 0

        stagnation_counter = 0
        termination_reason = "max_steps_reached"

        k = 0
        for k in range(max_dual_steps):
            target = tgt0 - np.conj(A) @ mu
            c_trial, V_trial, _ = _retract(target, c_grid, V_grid)

            leak = A.T @ c_trial
            _bump_eval()
            wn_db = 20 * np.log10(max([abs(_af(c_trial, nu, N)) for nu in nulls]) + 1e-12) if nulls else 0

            improved = wn_db < best_worst_null_db - stagnation_tol_db

            if improved:
                best_worst_null_db = wn_db
                best_c, best_V = c_trial.copy(), V_trial.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            hist.append({
                "iteration": k,
                "worst_null_db": wn_db,
                "improvement_db": best_worst_null_db - wn_db,
                "state_changed": improved,
                "target_reached": wn_db <= null_target_db,
            })

            # --- Check Philosophies ---
            if mode == "fixed":
                pass # Always run to max_dual_steps
            elif mode == "certified":
                # Only stop if monotonic improvement breaks/certifiably stagnated
                if not improved and stagnation_counter >= 1:
                    termination_reason = "certified_stagnation"
                    break
            elif mode == "adaptive":
                if wn_db <= null_target_db:
                    termination_reason = "target_reached"
                    break
                if stagnation_counter >= patience:
                    termination_reason = "adaptive_stagnation"
                    break

            # Gauss-Newton Dual Step
            mu = mu + Ginv @ leak

        # Refine best candidate found in this gauge
        V_glcp, c_glcp = gain_locked_polish_minimax(
            best_c, best_V, task, element, N, nulls,
            initial_gain=initial_gain, gain_floor_ratio=glcp_gain_floor
        )

        # Score final PTNR for the gauge
        _bump_eval()
        g_db = 20 * np.log10(abs(_af(c_glcp, u_t, N)) / N + 1e-12)
        wn_db = 20 * np.log10(max([abs(_af(c_glcp, nu, N)) for nu in nulls]) + 1e-12) if nulls else 0
        ptnr = g_db - wn_db

        if ptnr > best_global_ptnr:
            best_global_ptnr = ptnr
            best_global_res = {
                "voltages": V_glcp,
                "weights": c_glcp,
                "info": {
                    "convergence": {
                        "history": hist,
                        "iterations_used": k + 1,
                        "target_reached": wn_db <= null_target_db,
                        "termination_reason": termination_reason,
                        "null_certificate_dB": best_worst_null_db
                    },
                    "final_worst_null_db": wn_db,
                    "final_gain_db": g_db
                }
            }

    full_evals, glcp_updates = get_counters()
    best_global_res["info"]["evals"] = full_evals
    best_global_res["info"]["glcp_updates"] = glcp_updates
    best_global_res["info"]["runtime_ms"] = (time.perf_counter() - t0) * 1000.0

    return best_global_res

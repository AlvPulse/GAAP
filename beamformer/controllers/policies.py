import numpy as np

# POLICY A: Fixed Baseline
def policy_A_fixed(t, current_delta, **kwargs):
    return current_delta

# POLICY B: Random Basin Hopping
def policy_B_random(t, current_delta, **kwargs):
    if np.random.rand() < 0.2:
        return np.random.uniform(0, 2*np.pi)
    return current_delta

# POLICY C: Trajectory-Aware Periodic Scan
# It triggers every M steps. We scan locally around the historical best null offset.
def policy_C_periodic_scan(t, current_delta, M, history_null, history_gain, best_delta_null, best_delta_gain, scan_idx, **kwargs):
    if t % M == 0 and t >= M:
        recent_null = np.mean(history_null[-M:])
        recent_gain = np.mean(history_gain[-M:])

        deficit_null = recent_null - (-40) # We want null to be -40 or lower
        deficit_gain = 20*np.log10(kwargs.get('N', 64)) - recent_gain # Max possible gain minus actual

        if deficit_null > deficit_gain:
            # Null bottleneck
            phi_scan = [-30, -15, 0, 15, 30]
            idx = scan_idx % len(phi_scan)
            shift = np.deg2rad(phi_scan[idx])
            new_delta = (best_delta_null + shift) % (2*np.pi)
            return new_delta, scan_idx + 1
        else:
            # Gain bottleneck
            return best_delta_gain, scan_idx
    return current_delta, scan_idx

# POLICY D: Stagnation-Triggered Trajectory Escapement
def policy_D_stagnation(t, current_delta, W, history_null, history_gain, history_flips, history_c, **kwargs):
    if t >= W:
        sigma_null = np.std(history_null[-W:])
        sigma_gain = np.std(history_gain[-W:])
        avg_flips = np.mean(history_flips[-W:])

        if (sigma_null < 0.5) and (sigma_gain < 0.2) and (avg_flips <= 1):
            c_t = history_c[-1]
            c_t_W = history_c[-W]

            # calculate mean phase change
            phase_change = np.angle(np.sum(c_t * np.conj(c_t_W)))
            new_delta = (current_delta + phase_change + np.pi/2) % (2*np.pi)
            return new_delta
    return current_delta

# POLICY E: Probe-Based Diversity Optimization
def policy_E_probe(t, current_delta, W, history_null, history_gain, history_flips, current_c, current_V, current_b, element, N, task, **kwargs):
    if t >= W:
        sigma_null = np.std(history_null[-W:])
        sigma_gain = np.std(history_gain[-W:])
        avg_flips = np.mean(history_flips[-W:])

        if (sigma_null < 0.5) and (sigma_gain < 0.2) and (avg_flips <= 1):
            from .core import step_ap_lr

            candidates = np.linspace(0, 2*np.pi, 12, endpoint=False)
            best_cand = current_delta
            best_null_residual = float('inf')

            for cand in candidates:
                cand_c = current_c * np.exp(1j * (cand - current_delta))
                cand_V = np.zeros(N)
                for n in range(N):
                    res = element.project(cand_c[n], method='euclidean')
                    cand_V[n] = res[0]
                    cand_c[n] = res[1]

                # exactly 2 steps of pure AP
                for _ in range(2):
                    cand_c, cand_V, _, _, _, _, b_probe = step_ap_lr(cand_c, cand, cand_V, task, element, refinement_steps_inner=0)

                D = np.sum(b_probe != current_b)

                if 0.1 * N <= D <= 0.4 * N:
                    from .metrics import get_metrics
                    null_depth, _, _ = get_metrics(cand_c, task, element)
                    if null_depth < best_null_residual:
                        best_null_residual = null_depth
                        best_cand = cand

            if best_cand != current_delta:
                return best_cand

    return current_delta

# POLICY F: Simulated Annealing
# Returns proposed delta and a temperature T_t.
# The acceptance logic will run in the main benchmark script.
def policy_F_sa_propose(t, current_delta, T_0, **kwargs):
    T_t = T_0 * (0.85)**t
    new_delta = (current_delta + np.random.normal(0, np.deg2rad(15))) % (2*np.pi)
    return new_delta, T_t

# POLICY G: Kinetic Simulated Annealing (Ablation)
def policy_G_ksa_propose(t, current_delta, T_0, W, history_c, history_null, history_gain, history_flips, **kwargs):
    T_t = T_0 * (0.85)**t

    if t > W and len(history_c) >= W:
        sigma_null = np.std(history_null[-W:])
        sigma_gain = np.std(history_gain[-W:])
        avg_flips = np.mean(history_flips[-W:])

        c_t = history_c[-1]
        c_t_W = history_c[-W]
        phase_velocity = np.angle(np.sum(c_t * np.conj(c_t_W))) / W

        if (sigma_null < 0.5) and (sigma_gain < 0.2) and (avg_flips <= 1):
            # Explode
            new_delta = np.random.uniform(0, 2*np.pi)
        else:
            # Shifted Gaussian
            new_delta = (current_delta + phase_velocity + np.random.normal(0, np.deg2rad(15))) % (2*np.pi)
    else:
        new_delta = (current_delta + np.random.normal(0, np.deg2rad(15))) % (2*np.pi)

    return new_delta, T_t

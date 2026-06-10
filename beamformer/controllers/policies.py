import numpy as np

# POLICY A: Fixed Baseline
def policy_A_fixed(t, current_delta, **kwargs):
    return current_delta, False

# POLICY B: Random Basin Hopping
def policy_B_random(t, current_delta, **kwargs):
    if np.random.rand() < 0.2:
        return np.random.uniform(0, 2*np.pi), True
    return current_delta, False

# POLICY C: Trajectory-Aware Periodic Scan
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
            return new_delta, True, scan_idx + 1
        else:
            # Gain bottleneck
            return best_delta_gain, True, scan_idx
    return current_delta, False, scan_idx

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
            return new_delta, True
    return current_delta, False

# POLICY E: Probe-Based Diversity Optimization
def policy_E_probe(t, current_delta, W, history_null, history_gain, history_flips, current_c, element, N, task, **kwargs):
    if t >= W:
        sigma_null = np.std(history_null[-W:])
        sigma_gain = np.std(history_gain[-W:])
        avg_flips = np.mean(history_flips[-W:])

        if (sigma_null < 0.5) and (sigma_gain < 0.2) and (avg_flips <= 1):
            from .core import step_ap_lr

            candidates = np.linspace(0, 2*np.pi, 12, endpoint=False)
            best_cand = current_delta
            best_null_residual = float('inf')

            current_b = np.zeros(N)
            for n in range(N):
                current_b[n], _ = element.project(current_c[n], method='euclidean')

            for cand in candidates:
                cand_c = current_c * np.exp(1j * cand)
                cand_V = np.zeros(N)
                for n in range(N):
                    cand_V[n], cand_c[n] = element.project(cand_c[n], method='euclidean')

                # 2 steps of pure AP
                for _ in range(2):
                    cand_c, cand_V, _ = step_ap_lr(cand_c, cand, cand_V, task, element, refinement_steps_inner=0)

                b_probe = cand_V
                D = np.sum(np.abs(b_probe - current_b) > 2.0)

                if 0.1 * N <= D <= 0.4 * N:
                    from .metrics import get_metrics
                    null_depth, _, _ = get_metrics(cand_c, task, element)
                    if null_depth < best_null_residual:
                        best_null_residual = null_depth
                        best_cand = cand

            if best_cand != current_delta:
                return best_cand, True

    return current_delta, False

# POLICY F: Simulated Annealing
def policy_F_sa(t, current_delta, current_ptnr, T_0, **kwargs):
    # Propose new
    new_delta = (current_delta + np.random.normal(0, np.deg2rad(15))) % (2*np.pi)

    # We must evaluate it (requires 1 step AP+LR which we will mock by passing back the new delta and letting the loop do it)
    # Actually to do it right, we propose the delta, return it, and the runner evaluates it and accepts/rejects.
    # To keep runner simple, we just return the proposal and a flag saying "evaluate this".
    return new_delta, True

# POLICY G: Momentum-Guided Hamiltonian Annealing
def policy_G_mgha(t, current_delta, W, history_null, history_gain, history_flips, history_c, **kwargs):
    # Similar to D, but adds velocity tracking
    if t >= W:
        sigma_null = np.std(history_null[-W:])
        sigma_gain = np.std(history_gain[-W:])
        avg_flips = np.mean(history_flips[-W:])

        c_t = history_c[-1]
        c_t_W = history_c[-W]
        phase_velocity = np.angle(np.sum(c_t * np.conj(c_t_W))) / W

        if (sigma_null < 0.5) and (sigma_gain < 0.2) and (avg_flips <= 1):
            # Surge kinetic energy
            new_delta = (current_delta + phase_velocity * W + np.random.uniform(np.pi/4, 3*np.pi/4)) % (2*np.pi)
            return new_delta, True
        else:
            # Drift with velocity
            if np.random.rand() < 0.1: # 10% chance to follow momentum
                new_delta = (current_delta + phase_velocity) % (2*np.pi)
                return new_delta, True
    return current_delta, False

# POLICY H: Predictive-Convergence Deadlock Resolution
def policy_H_predictive(t, current_delta, W, history_V, current_V, epsilon_V, B_total, task, element, current_c, current_lr_step, lr_min, tol=-35, epsilon_imp=1.0, M=12, **kwargs):
    if len(history_V) == W:
        V_old = history_V[0]
        if V_old <= 0: V_old = 1e-12
        rho_hat = (current_V / V_old) ** (1/W)
        B_rem = B_total - t
        if B_rem <= 0:
            B_rem = 1
        rho_req = (epsilon_V / current_V) ** (1/B_rem) if current_V > epsilon_V else 0

        if (rho_hat >= min(rho_req, 0.995)) and (current_lr_step <= lr_min * 1.01):
            # Deadlock
            from .core import step_ap_lr
            from .metrics import get_metrics

            candidates = np.linspace(0, 2*np.pi, M, endpoint=False)
            best_delta = current_delta
            best_gamma = float('inf')
            any_safe = False

            # evaluate current objective terms
            null_curr, gain_curr, _ = get_metrics(current_c, task, element)

            for cand in candidates:
                cand_c = current_c * np.exp(1j * cand)
                cand_V = np.zeros(kwargs.get('N', 64))
                for n in range(kwargs.get('N', 64)):
                    cand_V[n], cand_c[n] = element.project(cand_c[n], method='euclidean')

                # single AP
                cand_c, cand_V, res = step_ap_lr(cand_c, cand, cand_V, task, element, refinement_steps_inner=0)

                gamma_tilde = res / max(current_V, 1e-12)

                # check safe improvement
                null_cand, gain_cand, _ = get_metrics(cand_c, task, element)

                safe = True
                if null_curr > tol: # if we are violating null constraint
                    if (null_cand - null_curr) > -epsilon_imp: # negative is better for null depth
                        safe = False

                if safe and gamma_tilde < best_gamma:
                    best_gamma = gamma_tilde
                    best_delta = cand
                    any_safe = True

            if not any_safe:
                golden_ratio = (1 + np.sqrt(5)) / 2
                best_delta = (current_delta + 2*np.pi * golden_ratio) % (2*np.pi)

            return best_delta, True

    return current_delta, False

import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_schelkunoff,synthesize_phase_only_lms
from beamformer.controllers.core import step_ap_lr, apply_phase_jump
from beamformer.controllers.policies import *
from beamformer.controllers.metrics import get_metrics

def run_policy(policy_name, seed, N, T=50):
    np.random.seed(seed)
    element = SyntheticVaractor(beta=1.5, folding=True)

    target_angle = np.random.uniform(-0.4, 0.4)
    nulls = []
    while len(nulls) < 3:
        n_ang = np.random.uniform(-0.9, 0.9)
        if np.abs(n_ang - target_angle) > 0.15:
            nulls.append(n_ang)

    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    # Warm initialization
    w_ideal = synthesize_phase_only_lms(N, task.target_angles[0], task.null_angles)

    V_n = np.zeros(N)
    c_n = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        res = element.project(w_ideal[n])
        if len(res) == 3:
            V_n[n], c_n[n], _ = res
        else:
            V_n[n], c_n[n] = res

    # Initial offset
    delta = 0.0

    # History
    history_null = []
    history_gain = []
    history_ptnr = []
    history_flips = []
    history_c = []
    history_residual = []

    best_delta_null = delta
    best_null = float('inf')
    best_delta_gain = delta
    best_gain = -float('inf')

    scan_idx = 0
    W = 5
    M = 10
    T_0 = 4.328

    start_time = time.time()

    for t in range(1, T + 1):
        # 1. Outer Loop Decision
        delta_old = delta
        propose_delta = delta

        b_t_old = None
        if t > 1:
            current_b = np.zeros(N, dtype=int)
            for n in range(N):
                res = element.project(c_n[n])
                if len(res) == 3:
                    _, _, current_b[n] = res
            b_t_old = current_b

        is_sa = policy_name in ['Policy F', 'Policy G']

        if policy_name == 'Policy A':
            propose_delta = best_delta_gain
        elif policy_name == 'Policy B':
            propose_delta = policy_B_random(t, delta)
        elif policy_name == 'Policy C':
            propose_delta, scan_idx = policy_C_periodic_scan(t, delta, M, history_null, history_gain, best_delta_null, best_delta_gain, scan_idx, N=N)
        elif policy_name == 'Policy D':
            propose_delta = policy_D_stagnation(t, delta, W, history_null, history_gain, history_flips, history_c)
        elif policy_name == 'Policy E':
            current_b = np.zeros(N, dtype=int)
            for n in range(N):
                res = element.project(c_n[n])
                if len(res) == 3:
                    _, _, current_b[n] = res
            propose_delta = policy_E_probe(t, delta, W, history_null, history_gain, history_flips, c_n.copy(), V_n.copy(), current_b, element, N, task)
        elif policy_name == 'Policy F':
            propose_delta, T_t = policy_F_sa_propose(t, delta, T_0)
        elif policy_name == 'Policy G':
            propose_delta, T_t = policy_G_ksa_propose(t, delta, T_0, W, history_c, history_null, history_gain, history_flips)


        if is_sa:
            # We must evaluate it to decide acceptance
            c_cand, V_cand = apply_phase_jump(c_n.copy(), V_n.copy(), delta, propose_delta, element)
            c_cand, V_cand, _, null_cand, gain_cand, ptnr_cand, _ = step_ap_lr(c_cand, propose_delta, V_cand, task, element)

            E_new = -ptnr_cand
            E_old = -history_ptnr[-1] if len(history_ptnr) > 0 else 0

            if E_new < E_old:
                delta = propose_delta
            elif np.random.rand() < np.exp(-(E_new - E_old) / T_t):
                delta = propose_delta
        else:
            delta = propose_delta

        if delta != delta_old:
            c_n, V_n = apply_phase_jump(c_n, V_n, delta_old, delta, element)

        # 2. Inner AP+LR
        c_n, V_n, res, null_depth, gain, ptnr, b_t = step_ap_lr(c_n, delta, V_n, task, element, refinement_steps_inner=20)

        # 3. State update
        flips = 0
        if b_t_old is not None:
            flips = np.sum(b_t != b_t_old)

        history_null.append(null_depth)
        history_gain.append(gain)
        history_ptnr.append(ptnr)
        history_flips.append(flips)
        history_c.append(c_n.copy())
        history_residual.append(res)

        if null_depth < best_null:
            best_null = null_depth
            best_delta_null = delta
        if gain > best_gain:
            best_gain = gain
            best_delta_gain = delta

    wall_clock = time.time() - start_time

    return {
        'policy': policy_name,
        'seed': seed,
        'N': N,
        'null': history_null,
        'gain': history_gain,
        'ptnr': history_ptnr,
        'flips': history_flips,
        'residual': history_residual,
        'wall_clock': wall_clock
    }

def run_main_benchmark():
    os.makedirs('experiments/data', exist_ok=True)

    policies = ['Policy A', 'Policy B', 'Policy C', 'Policy D', 'Policy E', 'Policy F', 'Policy G']
    N = 256
    seeds = 20
    T = 50

    all_results = []

    for policy in policies:
        print(f"Running {policy}...")
        for seed in range(seeds):
            res = run_policy(policy, seed, N, T)
            # Expand to dataframe format for easy iteration plotting
            for t in range(T):
                all_results.append({
                    'Policy': policy,
                    'Seed': seed,
                    'Iteration': t + 1,
                    'NullDepth': res['null'][t],
                    'Gain': res['gain'][t],
                    'PTNR': res['ptnr'][t],
                    'Flips': res['flips'][t],
                    'CumulativeFlips': np.sum(res['flips'][:t+1]),
                    'Residual': res['residual'][t]
                })

    df = pd.DataFrame(all_results)
    df.to_csv('experiments/data/main_benchmark_policies.csv', index=False)
    print("Main benchmark completed.")

def run_scaling_benchmark():
    policies = ['Policy A', 'Policy D', 'Policy E', 'Policy F'] # Subset to save time if needed, or all
    Ns = [64, 256, 1024]
    seeds = 3
    T = 10 # Short horizon for scaling

    all_results = []

    for N in Ns:
        for policy in policies:
            print(f"Scaling check: {policy} at N={N}...")
            for seed in range(seeds):
                res = run_policy(policy, seed, N, T)
                all_results.append({
                    'Policy': policy,
                    'N': N,
                    'Seed': seed,
                    'WallClock_s': res['wall_clock']
                })

    df = pd.DataFrame(all_results)
    df.to_csv('experiments/data/scaling_benchmark_policies.csv', index=False)
    print("Scaling benchmark completed.")

if __name__ == '__main__':
    run_main_benchmark()
    run_scaling_benchmark()

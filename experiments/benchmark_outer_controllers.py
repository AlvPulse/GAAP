import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_pencil, synthesize_schelkunoff
from beamformer.controllers import (
    step_ap_lr, apply_phase_jump, get_metrics, count_flips,
    policy_A_fixed, policy_B_random, policy_C_periodic_scan,
    policy_D_stagnation, policy_E_probe, policy_F_sa, policy_G_mgha,
    policy_H_predictive
)
from beamformer.geometry import count_branch_flips

def run_single_seed(seed, N, policy_name, T=50, W=5):
    np.random.seed(seed)

    # 1. Setup statistically independent task for this seed
    element = SyntheticVaractor(beta=1.5, folding=True)

    target_angle = np.random.uniform(0.1, 0.5)
    # 3 random nulls
    nulls = [np.random.uniform(-0.8, -0.2), np.random.uniform(0.6, 0.9), np.random.uniform(-0.1, 0.0)]
    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    # 2. Init weights (Schelkunoff nulling init)
    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    # Initial offset scan to find baseline opt
    from beamformer.geometry import scan_offset_feasibility
    metrics = scan_offset_feasibility(c_n, element, n_steps=100)
    best_idx = np.argmin(metrics['residuals'])
    delta_gain = metrics['deltas'][best_idx] # proxy for max amplitude
    delta_null = metrics['deltas'][np.random.randint(0, 100)] # simplified placeholder

    current_delta = delta_gain

    # Init hardware
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n], _= element.project(c_n[n] * np.exp(1j * current_delta))

    # Tracker
    history = {
        'null_depth': [],
        'gain': [],
        'ptnr': [],
        'residual': [],
        'flips': [],
        'c_n': []
    }

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    scan_idx = 0
    history_V = []

    # Policy F SA state
    current_ptnr_cost = float('inf')
    T_0 = 10.0 # Will accept 3dB drop initially

    start_time = time.time()

    for t in range(1, T + 1):
        # 1. Step AP + LR
        prev_c = c_n.copy()
        prev_V = V_n.copy()

        c_n, V_n, res, _ = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=lr_step, refinement_steps_inner=20)

        # 2. Metrics
        null_depth, gain, ptnr = get_metrics(c_n, task, element)
        flips = count_flips(V_n, prev_V)

        history['null_depth'].append(null_depth)
        history['gain'].append(gain)
        history['ptnr'].append(ptnr)
        history['residual'].append(res)
        history['flips'].append(flips)
        history['c_n'].append(c_n.copy())

        history_V.append(res)
        if len(history_V) > W:
            history_V.pop(0)

        # LR Step adaptation (standard)
        if t > 1 and res < history['residual'][-2]:
            lr_step = min(lr_step * 1.1, lr_max)
        else:
            lr_step = max(lr_step * 0.5, lr_min)

        # Policy execution
        new_delta = current_delta
        jump_triggered = False

        if policy_name == 'A':
            new_delta, jump_triggered = policy_A_fixed(t, current_delta)
        elif policy_name == 'B':
            new_delta, jump_triggered = policy_B_random(t, current_delta)
        elif policy_name == 'C':
            new_delta, jump_triggered, scan_idx = policy_C_periodic_scan(
                t, current_delta, 10, history['null_depth'], history['gain'], delta_null, delta_gain, scan_idx, N=N
            )
        elif policy_name == 'D':
            new_delta, jump_triggered = policy_D_stagnation(
                t, current_delta, W, history['null_depth'], history['gain'], history['flips'], history['c_n']
            )
        elif policy_name == 'E':
            new_delta, jump_triggered = policy_E_probe(
                t, current_delta, W, history['null_depth'], history['gain'], history['flips'], c_n, element, N, task
            )
        elif policy_name == 'F':
            # SA logic
            new_ptnr_cost = -ptnr
            cand_delta, _ = policy_F_sa(t, current_delta, new_ptnr_cost, T_0)

            # Evaluate acceptance immediately
            cand_c, cand_V = apply_phase_jump(c_n, V_n, current_delta, cand_delta, element)
            cand_c, cand_V, cand_res, _ = step_ap_lr(cand_c, cand_delta, cand_V, task, element, refinement_steps_inner=20)
            _, _, cand_ptnr = get_metrics(cand_c, task, element)
            cand_ptnr_cost = -cand_ptnr

            accept = False
            if cand_ptnr_cost < new_ptnr_cost:
                accept = True
            else:
                T_t = T_0 * (0.85 ** t)
                if T_t > 1e-6:
                    prob = np.exp(-(cand_ptnr_cost - new_ptnr_cost) / T_t)
                    if np.random.rand() < prob:
                        accept = True

            if accept:
                new_delta = cand_delta
                c_n = cand_c
                V_n = cand_V
                lr_step = lr_initial # Reset step size
                # Skip normal jump trigger since we already applied it
        elif policy_name == 'G':
            new_delta, jump_triggered = policy_G_mgha(
                t, current_delta, W, history['null_depth'], history['gain'], history['flips'], history['c_n']
            )
        elif policy_name == 'H':
            new_delta, jump_triggered = policy_H_predictive(
                t, current_delta, W, history_V, res, 1e-4, T, task, element, c_n, lr_step, lr_min, N=N
            )

        if jump_triggered and policy_name != 'F':
            c_n, V_n = apply_phase_jump(c_n, V_n, current_delta, new_delta, element)
            current_delta = new_delta
            lr_step = lr_initial
            history_V.clear()

    exec_time = time.time() - start_time

    return history, exec_time

def run_benchmarks():
    policies = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    num_seeds = 20
    N = 64
    T = 50

    results = {p: [] for p in policies}

    for p in policies:
        print(f"Running Policy {p}...")
        for seed in range(num_seeds):
            hist, _ = run_single_seed(seed, N, p, T=T)
            results[p].append(hist)

    # Save results for plotting
    np.save('experiments/benchmark_results.npy', results, allow_pickle=True)
    print("Benchmarking complete. Results saved.")

if __name__ == '__main__':
    run_benchmarks()

def plot_results(scales, times):
    results = np.load('experiments/benchmark_results.npy', allow_pickle=True).item()
    policies = list(results.keys())

    T = len(results['A'][0]['null_depth'])
    iterations = np.arange(1, T + 1)

    fig, axs = plt.subplots(3, 2, figsize=(15, 15))

    metrics = [
        ('null_depth', 'Null Depth (dB)', axs[0, 0]),
        ('gain', 'Main-Beam Gain (dB)', axs[0, 1]),
        ('ptnr', 'Peak-to-Null Ratio (dB)', axs[1, 0]),
        ('residual', 'Manifold Projection Residual', axs[1, 1])
    ]

    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))

    for i, p in enumerate(policies):
        for metric_key, ylabel, ax in metrics:
            data = np.array([res[metric_key] for res in results[p]])
            mean_val = np.mean(data, axis=0)
            q25 = np.percentile(data, 25, axis=0)
            q75 = np.percentile(data, 75, axis=0)

            ax.plot(iterations, mean_val, label=f'Policy {p}', color=colors[i])
            ax.fill_between(iterations, q25, q75, color=colors[i], alpha=0.1)
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Iteration')
            ax.grid(True)

            if metric_key == 'null_depth' and i == 0:
                ax.axhline(-40, color='r', linestyle='--', label='Target -40dB')

        # Plot 5: Cumulative Branch Flips
        ax = axs[2, 0]
        data = np.array([res['flips'] for res in results[p]])
        cum_flips = np.cumsum(data, axis=1)
        mean_val = np.mean(cum_flips, axis=0)
        q25 = np.percentile(cum_flips, 25, axis=0)
        q75 = np.percentile(cum_flips, 75, axis=0)

        ax.plot(iterations, mean_val, label=f'Policy {p}', color=colors[i])
        ax.fill_between(iterations, q25, q75, color=colors[i], alpha=0.1)
        ax.set_ylabel('Cumulative Branch Flips')
        ax.set_xlabel('Iteration')
        ax.grid(True)


    # Plot 6: Scaling on axs[2, 1]
    ax = axs[2, 1]
    for i, p in enumerate(policies):
        if p in times:
            ax.plot(scales, [t * 1000 for t in times[p]], marker='o', label=f'Policy {p}', color=colors[i])
    ax.set_xlabel('Array Scale N')
    ax.set_ylabel('Wall-clock Time (ms)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xticks(scales)
    ax.set_xticklabels(scales)
    ax.grid(True)

    for ax in axs.flatten():
        ax.legend(loc='best', fontsize='small')

    plt.tight_layout()
    plt.savefig('experiments/benchmark_6panel.png')
    print("Saved 6-panel plot to experiments/benchmark_6panel.png")

def run_scaling_study():
    print("Running scaling study...")
    policies = ['A', 'D', 'H'] # Subset for speed, or all if preferred
    scales = [64, 256, 1024]

    times = {p: [] for p in policies}

    for N in scales:
        print(f"Scale N={N}")
        for p in policies:
            # single run for timing
            _, exec_time = run_single_seed(0, N, p, T=20, W=5)
            times[p].append(exec_time)

    return scales, times

if __name__ == '__main__':
    run_benchmarks()
    scales, times = run_scaling_study()
    plot_results(scales, times)

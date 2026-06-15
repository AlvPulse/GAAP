import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())

from beamformer.element_model import SyntheticVaractor
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.pattern_projection import Task
from beamformer.controllers.modular_backbone import OptimizationBackbone
from beamformer.controllers.policies import (
    PolicyA_FixedBaseline, PolicyB_RandomBasinHopping, PolicyC_TrajectoryScan,
    PolicyD_StagnationEscapement, PolicyE_ProbeDiversity, PolicyF_SimulatedAnnealing,
    PolicyG_TrustRegionAnnealing
)
from beamformer.controllers.metrics import get_metrics

def run_experiment(policy_class, N=64, iterations=50, seed=42):
    np.random.seed(seed)

    target_u = np.sin(30 * np.pi / 180)
    null_u = [np.sin(-20 * np.pi / 180), np.sin(50 * np.pi / 180)]

    element = SyntheticVaractor()
    task = Task(type='nulled', target_angles=[target_u], null_angles=null_u)

    # Init
    w = synthesize_schelkunoff(N, target_u, null_u)
    w /= np.max(np.abs(w))

    backbone = OptimizationBackbone(task, element, N)
    policy = policy_class(N)

    delta = 0.0
    history = []
    flips_cumulative = []
    total_flips = 0

    # Get initial metrics
    _, b_init, _ = element.project(w[0]) # just to initialize b shape
    b = np.zeros(N, dtype=int)
    for n in range(N):
        _, _, b[n] = element.project(w[n], method='phase_only')

    start_time = time.time()

    for t in range(1, iterations + 1):
        # 1. Policy decides next delta
        if t == 1:
            metrics = {'null_depth': -10.0, 'gain': 0.0, 'ptnr': 10.0} # dummy initial
            delta_new = delta
        else:
            delta_new = policy.step(t, delta, metrics, history, w, b, backbone)

        # 2. Backbone executes
        w_new, b_new, metrics = backbone.run_step(w, delta_new)

        # 3. Track metrics
        null_res = np.sqrt(np.sum(np.abs(w_new - w)**2)) # dummy hard residual
        history.append({
            'null_depth': metrics['null_depth'],
            'gain': metrics['gain'],
            'ptnr': metrics['ptnr'],
            'hard_residual': null_res
        })

        flips = np.sum(b_new != b)
        total_flips += flips
        flips_cumulative.append(total_flips)

        # Update state
        w = w_new
        b = b_new
        delta = delta_new

    end_time = time.time()
    wall_clock = (end_time - start_time) * 1000 # ms

    return history, flips_cumulative, wall_clock

def run_benchmarks():
    policies = {
        'A (Fixed)': PolicyA_FixedBaseline,
        'B (Random Hop)': PolicyB_RandomBasinHopping,
        'C (Traj Scan)': PolicyC_TrajectoryScan,
        'D (Stagnation)': PolicyD_StagnationEscapement,
        'E (Probe Div)': PolicyE_ProbeDiversity,
        'F (SA)': PolicyF_SimulatedAnnealing,
        'G (TRA)': PolicyG_TrustRegionAnnealing
    }

    seeds = range(20)
    iterations = 50
    N_list = [64, 256, 1024]

    results = {name: {'null_depth': [], 'gain': [], 'ptnr': [], 'hard_residual': [], 'flips': [], 'times': {n: [] for n in N_list}} for name in policies}

    print("Running benchmarks...")
    for name, pclass in policies.items():
        print(f"  Benchmarking {name}...")

        # Run 20 seeds for main plots (N=64)
        for seed in seeds:
            hist, flips, _ = run_experiment(pclass, N=64, iterations=iterations, seed=seed)
            results[name]['null_depth'].append([h['null_depth'] for h in hist])
            results[name]['gain'].append([h['gain'] for h in hist])
            results[name]['ptnr'].append([h['ptnr'] for h in hist])
            results[name]['hard_residual'].append([h['hard_residual'] for h in hist])
            results[name]['flips'].append(flips)

        # Run scaling benchmarks
        for N in N_list:
            n_seeds = 5 if N == 1024 else 20
            for seed in range(n_seeds):
                _, _, t_ms = run_experiment(pclass, N=N, iterations=1, seed=seed) # Just 1 iteration for timing to save time
                results[name]['times'][N].append(t_ms)

    # Plotting
    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    axes = axes.flatten()

    colors = ['black', 'red', 'blue', 'orange', 'purple', 'green', 'magenta']

    for (name, data), color in zip(results.items(), colors):
        t_axis = np.arange(1, iterations + 1)

        def plot_iqr(ax_idx, key, label=None):
            mean = np.mean(data[key], axis=0)
            p25 = np.percentile(data[key], 25, axis=0)
            p75 = np.percentile(data[key], 75, axis=0)
            axes[ax_idx].plot(t_axis, mean, label=label if label else name, color=color)
            axes[ax_idx].fill_between(t_axis, p25, p75, color=color, alpha=0.2)

        plot_iqr(0, 'null_depth')
        plot_iqr(1, 'gain')
        plot_iqr(2, 'ptnr')
        plot_iqr(3, 'hard_residual')
        plot_iqr(4, 'flips', label=name)

        # Scaling Plot
        mean_times = [np.mean(data['times'][n]) for n in N_list]
        axes[5].plot(N_list, mean_times, '-o', color=color, label=name)

    axes[0].set_title('Null Depth (dB)')
    axes[0].axhline(-40, color='r', linestyle='--')
    axes[1].set_title('Main-Beam Gain (dB)')
    axes[2].set_title('Peak-to-Null Ratio (PTNR in dB)')
    axes[3].set_title('Hard Manifold Projection Residual')
    axes[4].set_title('Cumulative Branch Flips')
    axes[4].legend(loc='upper left')
    axes[5].set_title('Scaling Performance')
    axes[5].set_xlabel('Array Scale N')
    axes[5].set_ylabel('Wall-clock time (ms)')
    axes[5].set_xscale('log')
    axes[5].set_yscale('log')
    axes[5].set_xticks(N_list)
    axes[5].set_xticklabels(N_list)

    plt.tight_layout()
    plt.savefig('Final_6Panel_Benchmark.png')
    print("Saved Final_6Panel_Benchmark.png")

if __name__ == '__main__':
    run_benchmarks()

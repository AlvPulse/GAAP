import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.controllers.core import step_ap_lr, apply_phase_jump
from beamformer.controllers.metrics import get_metrics

from beamformer.controllers.sa_families import (
    FixedOffset, RandomRestart, HillClimbing, BasinHopping,
    SA, VFSA, ASA, HT_ROA, CEM
)

def run_single_seed_sa(seed, N, controller_class, B_total=50):
    np.random.seed(seed)

    element = SyntheticVaractor(beta=1.5, folding=True)
    target_angle = np.random.uniform(0.1, 0.5)
    nulls = [np.random.uniform(-0.8, -0.2), np.random.uniform(0.6, 0.9), np.random.uniform(-0.1, 0.0)]
    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    current_delta = 0.0
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n] = element.project(c_n[n] * np.exp(1j * current_delta))

    history = {
        'ptnr': [],
        'null_depth': [],
        'gain': [],
        'residual': [],
        'delta_diff': [],
        'accept_ratio': []
    }

    controller = controller_class()

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    # Evaluate initial state
    c_n, V_n, res = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=lr_step)
    _, _, current_ptnr = get_metrics(c_n, task, element)
    current_cost = -current_ptnr

    best_cost = current_cost
    best_c = c_n.copy()
    best_V = V_n.copy()
    best_delta = current_delta

    history_res = [res]

    for t in range(1, B_total + 1):
        # 1. Propose new offset
        cand_delta = controller.propose(current_delta)
        delta_diff = min(np.abs(cand_delta - current_delta), 2*np.pi - np.abs(cand_delta - current_delta))

        # 2. Evaluate candidate (Black box evaluation F(delta))
        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)

        # Execute 1 step AP + LR
        cand_c, cand_V, cand_res = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=lr_step)

        null_depth, gain, ptnr = get_metrics(cand_c, task, element)
        cand_cost = -ptnr

        # 3. Accept/Reject
        accepted = controller.accept(current_cost, cand_cost)

        if accepted:
            current_delta = cand_delta
            current_cost = cand_cost
            best_c = cand_c
            best_V = cand_V
            best_delta = cand_delta
            # Reset LR step size on jump
            if delta_diff > 1e-3:
                lr_step = lr_initial

        # Update Controller State
        controller.update_state(accepted, history_r=history_res, lr_step=lr_step, lr_min=lr_min, new_cost=cand_cost)

        # Record metrics (recording the BEST seen state, or current state depending on interpretation.
        # For SA, tracking current state is standard)
        history['ptnr'].append(-current_cost)
        history['null_depth'].append(null_depth if accepted else history['null_depth'][-1] if history['null_depth'] else null_depth)
        history['gain'].append(gain if accepted else history['gain'][-1] if history['gain'] else gain)
        history['residual'].append(cand_res if accepted else history_res[-1])
        history['delta_diff'].append(delta_diff)
        history['accept_ratio'].append(controller.accepted_moves / max(1, controller.total_proposals))

        history_res.append(history['residual'][-1])

        # Adapt LR step based on evaluation
        if t > 1 and cand_res < history_res[-2]:
            lr_step = min(lr_step * 1.1, lr_max)
        else:
            lr_step = max(lr_step * 0.5, lr_min)

    return history

def run_sa_benchmarks():
    controllers = {
        'Fixed': FixedOffset,
        'Random': RandomRestart,
        'HillClimb': HillClimbing,
        'BasinHop': BasinHopping,
        'SA': SA,
        'VFSA': VFSA,
        'ASA': ASA,
        'HT-ROA': HT_ROA,
        'CEM': CEM
    }

    num_seeds = 20
    N = 64
    B_total = 50

    results = {name: [] for name in controllers.keys()}

    for name, cls in controllers.items():
        print(f"Running {name}...")
        for seed in range(num_seeds):
            hist = run_single_seed_sa(seed, N, cls, B_total=B_total)
            results[name].append(hist)

    np.save('experiments/benchmark_sa_results.npy', results, allow_pickle=True)
    print("SA benchmarking complete. Results saved.")



def calculate_higher_level_metrics():
    results = np.load('experiments/benchmark_sa_results.npy', allow_pickle=True).item()

    metrics_summary = {}

    for name, runs in results.items():
        auc_list = []
        ttt_list = []
        me_list = []

        for run in runs:
            ptnr = np.array(run['ptnr'])
            nulls = np.array(run['null_depth'])

            # AUC (Area Under Improvement Curve) - sum of PTNR over time
            auc = np.sum(ptnr)
            auc_list.append(auc)

            # Time-To-Target (Iterations to reach -40dB null, or max budget if not reached)
            reached = np.where(nulls <= -40.0)[0]
            if len(reached) > 0:
                ttt = reached[0] + 1
            else:
                ttt = len(nulls)
            ttt_list.append(ttt)

            # Meta-Efficiency (Total PTNR Improvement / Budget)
            improvement = np.max(ptnr) - ptnr[0]
            me = improvement / len(ptnr)
            me_list.append(me)

        metrics_summary[name] = {
            'AUC_mean': np.mean(auc_list),
            'AUC_std': np.std(auc_list),
            'TTT_mean': np.mean(ttt_list),
            'TTT_std': np.std(ttt_list),
            'ME_mean': np.mean(me_list),
            'ME_std': np.std(me_list)
        }

    # Print nice table
    print(f"{'Algorithm':<15} | {'AUC':<15} | {'TTT (Iters)':<15} | {'Meta-Efficiency':<15}")
    print("-" * 65)
    for name, m in metrics_summary.items():
        print(f"{name:<15} | {m['AUC_mean']:>8.1f} ± {m['AUC_std']:>4.1f} | {m['TTT_mean']:>8.1f} ± {m['TTT_std']:>4.1f} | {m['ME_mean']:>8.3f} ± {m['ME_std']:>4.3f}")

    return metrics_summary

def plot_sa_results():
    results = np.load('experiments/benchmark_sa_results.npy', allow_pickle=True).item()
    metrics = calculate_higher_level_metrics()

    controllers = list(results.keys())
    T = len(results[controllers[0]][0]['ptnr'])
    iterations = np.arange(1, T + 1)

    # 1. Convergence of PTNR
    plt.figure(figsize=(10, 6))
    for name in controllers:
        data = np.array([res['ptnr'] for res in results[name]])
        mean_val = np.mean(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)

        plt.plot(iterations, mean_val, label=name, linewidth=2)
        plt.fill_between(iterations, q25, q75, alpha=0.1)

    plt.xlabel('Evaluations (AP+LR executions)')
    plt.ylabel('Peak-to-Null Ratio (PTNR) [dB]')
    plt.title('Convergence of Online Optimizers')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/sa_convergence_ptnr.png')

    # 2. Acceptance Rate and Jump Size for Key Algorithms
    key_algos = ['BasinHop', 'SA', 'HT-ROA']
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name in key_algos:
        if name not in results: continue

        # Accept Rate
        acc = np.array([res['accept_ratio'] for res in results[name]])
        axs[0].plot(iterations, np.mean(acc, axis=0), label=name)

        # Jump Size
        jumps = np.array([res['delta_diff'] for res in results[name]])
        axs[1].plot(iterations, np.mean(jumps, axis=0), label=name)

    axs[0].set_ylabel('Acceptance Ratio')
    axs[0].set_title('Exploration Mechanics over Time')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].set_ylabel('Mean Phase Jump $|\\Delta\\delta|$ (rad)')
    axs[1].set_xlabel('Evaluations')
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('experiments/sa_mechanics.png')

    # 3. Bar Chart of Meta-Efficiency and Time-To-Target
    fig, ax1 = plt.subplots(figsize=(12, 6))

    x = np.arange(len(controllers))
    width = 0.35

    me_means = [metrics[n]['ME_mean'] for n in controllers]
    me_stds = [metrics[n]['ME_std'] for n in controllers]
    ttt_means = [metrics[n]['TTT_mean'] for n in controllers]
    ttt_stds = [metrics[n]['TTT_std'] for n in controllers]

    color1 = 'tab:blue'
    ax1.set_xlabel('Controller')
    ax1.set_ylabel('Meta-Efficiency (dB/eval)', color=color1)
    ax1.bar(x - width/2, me_means, width, yerr=me_stds, label='Meta-Efficiency', color=color1, capsize=5)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Time-to-Target (-40dB Nulls) [Iters]', color=color2)
    ax2.bar(x + width/2, ttt_means, width, yerr=ttt_stds, label='Time-to-Target', color=color2, capsize=5)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.xticks(x, controllers, rotation=45)
    plt.title('High-Level Performance Metrics')
    fig.tight_layout()
    plt.savefig('experiments/sa_metrics_bar.png')

    print("Plots saved to experiments/sa_convergence_ptnr.png, sa_mechanics.png, sa_metrics_bar.png")

if __name__ == '__main__':
    run_sa_benchmarks()
    plot_sa_results()

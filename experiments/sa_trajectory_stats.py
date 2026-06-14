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
from beamformer.controllers.sa_families import SA, HillClimbing

def run_trajectory_tracking(seed, N, controller_class, B_total=100):
    np.random.seed(seed)

    element = SyntheticVaractor(beta=1.5, folding=True)
    target_angle = 0.3
    nulls = [-0.5, 0.1, 0.7]
    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    current_delta = 0.0
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n] = element.project(c_n[n] * np.exp(1j * current_delta))

    controller = controller_class()

    # Eval initial
    c_n, V_n, res = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=0.05)
    _, _, current_ptnr = get_metrics(c_n, task, element)
    current_cost = -current_ptnr

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta

    history = {
        'deltas': [current_delta],
        'move_types': [] # 0: Rejected, 1: Accept Improving, 2: Accept Worsening
    }

    for t in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)

        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)
        cand_c, cand_V, cand_res = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=0.05)

        _, _, ptnr = get_metrics(cand_c, task, element)
        cand_cost = -ptnr

        accepted = controller.accept(current_cost, cand_cost)

        if accepted:
            if cand_cost < current_cost:
                history['move_types'].append(1) # Improving
            else:
                history['move_types'].append(2) # Worsening

            current_delta = cand_delta
            current_cost = cand_cost
            best_c, best_V, best_delta = cand_c, cand_V, cand_delta
        else:
            history['move_types'].append(0) # Rejected

        history['deltas'].append(current_delta)
        controller.update_state(accepted)

    return history

def plot_trajectories():
    B_total = 100
    N = 64
    seed = 42

    sa_hist = run_trajectory_tracking(seed, N, SA, B_total)
    hc_hist = run_trajectory_tracking(seed, N, HillClimbing, B_total)

    iterations = np.arange(B_total + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12))

    # 1. Delta trajectories
    axs[0].plot(iterations, sa_hist['deltas'], label='SA', marker='.')
    axs[0].plot(iterations, hc_hist['deltas'], label='HillClimbing', marker='.')
    axs[0].set_ylabel('Offset $\\delta$ (rad)')
    axs[0].set_title('Offset Trajectories')
    axs[0].legend()
    axs[0].grid(True)

    # 2. SA Move Stats
    iters_moves = np.arange(1, B_total + 1)
    sa_moves = np.array(sa_hist['move_types'])

    # Scatter plot of moves for SA
    improving = np.where(sa_moves == 1)[0] + 1
    worsening = np.where(sa_moves == 2)[0] + 1
    rejected = np.where(sa_moves == 0)[0] + 1

    axs[1].scatter(improving, np.ones_like(improving), color='green', label='Improving', marker='o')
    axs[1].scatter(worsening, np.ones_like(worsening)*0.5, color='orange', label='Worsening', marker='x')
    axs[1].scatter(rejected, np.zeros_like(rejected), color='red', label='Rejected', marker='.')
    axs[1].set_yticks([0, 0.5, 1])
    axs[1].set_yticklabels(['Rejected', 'Worsening', 'Improving'])
    axs[1].set_title('SA Acceptance Statistics over Time')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Rolling Accept Ratio
    window = 10
    if B_total >= window:
        rolling_improving = np.convolve(sa_moves == 1, np.ones(window)/window, mode='valid')
        rolling_worsening = np.convolve(sa_moves == 2, np.ones(window)/window, mode='valid')
        rolling_rejected = np.convolve(sa_moves == 0, np.ones(window)/window, mode='valid')

        axs[2].plot(np.arange(window, B_total+1), rolling_improving, color='green', label='Improving')
        axs[2].plot(np.arange(window, B_total+1), rolling_worsening, color='orange', label='Worsening')
        axs[2].plot(np.arange(window, B_total+1), rolling_rejected, color='red', label='Rejected')
        axs[2].set_xlabel('Iteration')
        axs[2].set_ylabel(f'Rolling Rate (W={window})')
        axs[2].set_title('SA Acceptance Rates')
        axs[2].legend()
        axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('experiments/sa_trajectory_stats.png')
    print("Saved trajectory stats to experiments/sa_trajectory_stats.png")

if __name__ == '__main__':
    plot_trajectories()

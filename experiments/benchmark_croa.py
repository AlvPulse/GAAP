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
from beamformer.controllers.sa_families import SA, HT_ROA, CROA

def run_deep_dive(seed, N, controller_class, B_total=100):
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

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    c_n, V_n, res = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=lr_step)
    _, _, current_ptnr = get_metrics(c_n, task, element)
    current_cost = -current_ptnr

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta
    history_res = [res]

    history = {
        'ptnr': [current_ptnr],
        'deltas': [current_delta],
        'lr_steps': [lr_step]
    }

    for t in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)
        delta_diff = min(np.abs(cand_delta - current_delta), 2*np.pi - np.abs(cand_delta - current_delta))

        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)
        cand_c, cand_V, cand_res = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=lr_step)

        _, _, ptnr = get_metrics(cand_c, task, element)
        cand_cost = -ptnr

        accepted = controller.accept(current_cost, cand_cost)


        # Calculate diff before state variables update
        old_delta = current_delta

        if accepted:
            current_cost = cand_cost
            best_c, best_V, best_delta = cand_c, cand_V, cand_delta

            if hasattr(controller, 'lr_step'):
                lr_step = controller.lr_step
            else:
                if delta_diff > 1e-3:
                    lr_step = lr_initial

        controller.update_state(
            accepted,
            history_r=history_res,
            lr_step=lr_step,
            lr_min=lr_min,
            new_cost=cand_cost,
            current_delta=old_delta,
            cand_delta=cand_delta,
            new_res=cand_res,
            r_target=1e-4,
            r_0=history_res[0] if history_res else 1.0
        )

        if accepted:
            current_delta = cand_delta

        if hasattr(controller, 'lr_step'):
            lr_step = controller.lr_step

        history['ptnr'].append(-current_cost)
        history['deltas'].append(current_delta)
        history['lr_steps'].append(lr_step)
        history_res.append(cand_res if accepted else history_res[-1])

        if t > 1 and cand_res < history_res[-2]:
            lr_step = min(lr_step * 1.1, lr_max)
        else:
            lr_step = max(lr_step * 0.5, lr_min)

        if hasattr(controller, 'lr_step'):
            controller.lr_step = lr_step

    return history

def plot_croa_comparison():
    B_total = 100
    N = 64
    seed = 42

    sa = run_deep_dive(seed, N, SA, B_total)
    htroa = run_deep_dive(seed, N, HT_ROA, B_total)
    croa = run_deep_dive(seed, N, CROA, B_total)

    iterations = np.arange(B_total + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(iterations, sa['ptnr'], label='Classical SA', marker='.')
    axs[0].plot(iterations, htroa['ptnr'], label='HT-ROA', marker='.')
    axs[0].plot(iterations, croa['ptnr'], label='CROA', marker='.')
    axs[0].set_ylabel('PTNR (dB)')
    axs[0].set_title('PTNR Convergence')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(iterations, sa['deltas'], label='Classical SA', marker='.')
    axs[1].plot(iterations, htroa['deltas'], label='HT-ROA', marker='.')
    axs[1].plot(iterations, croa['deltas'], label='CROA', marker='.')
    axs[1].set_ylabel('Offset $\\\\delta$ (rad)')
    axs[1].set_title('Offset Phase Trajectories')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(iterations, sa['lr_steps'], label='Classical SA (Hard Resets)', marker='.')
    axs[2].plot(iterations, htroa['lr_steps'], label='HT-ROA (Hard Resets)', marker='.')
    axs[2].plot(iterations, croa['lr_steps'], label='CROA (Soft Reannealing)', marker='.')
    axs[2].set_ylabel('LR Step Size $\\\\alpha$')
    axs[2].set_xlabel('Iteration')
    axs[2].set_title('Inner Optimizer Memory Tracking')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('experiments/benchmark_croa.png')
    print("Saved CROA deep dive to experiments/benchmark_croa.png")

if __name__ == '__main__':
    plot_croa_comparison()

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
from beamformer.controllers.sa_families import SA
from beamformer.controllers.trust_region import TR_HC, TR_SA

def run_tr_comparison(seed, N, controller_class, B_total=100):
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

    c_n, V_n, res = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=0.05)
    _, _, current_ptnr = get_metrics(c_n, task, element)
    current_cost = -current_ptnr

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta

    history = {
        'ptnr': [current_ptnr],
        'deltas': [current_delta],
        'TR_size': [getattr(controller, 'TR', getattr(controller, 'sigma', np.pi/12))]
    }

    for t in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)

        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)
        cand_c, cand_V, cand_res = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=0.05)

        _, _, ptnr = get_metrics(cand_c, task, element)
        cand_cost = -ptnr

        accepted = controller.accept(current_cost, cand_cost)

        if accepted:
            current_delta = cand_delta
            current_cost = cand_cost
            best_c, best_V, best_delta = cand_c, cand_V, cand_delta

        controller.update_state(accepted)

        history['ptnr'].append(-current_cost)
        history['deltas'].append(current_delta)
        history['TR_size'].append(getattr(controller, 'TR', getattr(controller, 'sigma', np.pi/12)))

    return history

def plot_debug_tr():
    B_total = 100
    N = 64
    seed = 42

    res_sa = run_tr_comparison(seed, N, SA, B_total)
    res_tr_hc = run_tr_comparison(seed, N, TR_HC, B_total)
    res_tr_sa = run_tr_comparison(seed, N, TR_SA, B_total)

    iterations = np.arange(B_total + 1)

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    axs[0].plot(iterations, res_sa['ptnr'], label='Classical SA', marker='.')
    axs[0].plot(iterations, res_tr_hc['ptnr'], label='Trust Region HC', marker='.')
    axs[0].plot(iterations, res_tr_sa['ptnr'], label='Trust Region SA', marker='.')
    axs[0].set_ylabel('PTNR (dB)')
    axs[0].set_title('PTNR Convergence Comparison')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(iterations, res_sa['deltas'], label='Classical SA', marker='.')
    axs[1].plot(iterations, res_tr_hc['deltas'], label='Trust Region HC', marker='.')
    axs[1].plot(iterations, res_tr_sa['deltas'], label='Trust Region SA', marker='.')
    axs[1].set_ylabel('Offset $\\delta$ (rad)')
    axs[1].set_title('Offset Trajectories')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(iterations, res_sa['TR_size'], label='Classical SA ($\sigma$)', marker='.')
    axs[2].plot(iterations, res_tr_hc['TR_size'], label='Trust Region HC ($\Delta$)', marker='.')
    axs[2].plot(iterations, res_tr_sa['TR_size'], label='Trust Region SA ($\Delta$)', marker='.')
    axs[2].set_ylabel('Proposal Bounds (rad)')
    axs[2].set_xlabel('Iteration')
    axs[2].set_title('Trust Region Size / Proposal StdDev')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('experiments/debug_trust_region.png')
    print("Saved TR debug plots to experiments/debug_trust_region.png")

if __name__ == '__main__':
    plot_debug_tr()

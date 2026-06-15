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
from beamformer.controllers.sa_families import FixedOffset

def sweep_landscape(c_curr, V_curr, delta_curr, task, element, N, num_points=50):
    deltas = np.linspace(0, 2*np.pi, num_points, endpoint=False)
    ptnrs, gains, nulls = [], [], []

    for cand_delta in deltas:
        cand_c, cand_V = apply_phase_jump(c_curr, V_curr, delta_curr, cand_delta, element)
        res_c, res_V, _, _ = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=0.05, refinement_steps_inner=20)
        null_depth, gain, ptnr = get_metrics(res_c, task, element)

        ptnrs.append(ptnr)
        gains.append(gain)
        nulls.append(null_depth)

    return deltas, np.array(ptnrs), np.array(gains), np.array(nulls)

def run_landscape_snapshots():
    np.random.seed(42)
    N = 64
    element = SyntheticVaractor(beta=1.5, folding=True)

    target_angle = 0.3
    nulls = [-0.5, 0.1, 0.7]
    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)
    current_delta = 0.0
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n], _= element.project(c_n[n] * np.exp(1j * current_delta))

    snapshots = [0, 10, 20, 40]
    results = {}

    # We use FixedOffset behavior to naturally evolve the state (AP+LR always evaluates at current_delta)
    c_curr = c_n.copy()
    V_curr = V_n.copy()

    for t in range(max(snapshots) + 1):
        if t in snapshots:
            print(f"Taking snapshot at iteration {t}...")
            deltas, ptnrs, gains, null_depths = sweep_landscape(c_curr, V_curr, current_delta, task, element, N)
            results[t] = {
                'deltas': deltas,
                'ptnr': ptnrs,
                'gain': gains,
                'null': null_depths
            }

        # Evolve state 1 step
        c_curr, V_curr, _, _ = step_ap_lr(c_curr, current_delta, V_curr, task, element, step_size=0.05, refinement_steps_inner=20)

    # Plotting
    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for t in snapshots:
        res = results[t]
        axs[0].plot(res['deltas'], res['gain'], label=f'Iter {t}')
        axs[1].plot(res['deltas'], res['null'], label=f'Iter {t}')
        axs[2].plot(res['deltas'], res['ptnr'], label=f'Iter {t}')

    axs[0].set_ylabel('Gain (dB)')
    axs[0].set_title('Gain Landscape $F(\\delta)$ Evolution')
    axs[0].grid(True); axs[0].legend()

    axs[1].set_ylabel('Null Depth (dB)')
    axs[1].set_title('Null Landscape $F(\\delta)$ Evolution')
    axs[1].grid(True)

    axs[2].set_ylabel('PTNR (dB)')
    axs[2].set_xlabel('Offset $\\delta$ (rad)')
    axs[2].set_title('PTNR Landscape $F(\\delta)$ Evolution')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('experiments/landscape_snapshots.png')
    print("Saved landscape snapshots to experiments/landscape_snapshots.png")

if __name__ == '__main__':
    run_landscape_snapshots()

"""
Experiment: PTNR Gauge Opportunity Loss Scaling.

This script scales the array size (N) and number of nulls (M) to verify
that the ~13 dB opportunity loss between the Gain Gauge and PTNR Gauge
is a fundamental scaling property of the hardware-algorithm co-design,
not an artifact of a specific array size.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.pattern_projection import Task


def run_scaling_experiment():
    np.random.seed(42)
    # element = SyntheticVaractor()
    element= MeasuredVaractor()

    N_values = [16, 32, 64, 128,256,1024]
    null_counts = [1, 2]
    num_trials = 20

    psi_grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    print("N\tNulls\tMedian Opp. Loss (dB)\tMean Opp. Loss (dB)")
    print("-" * 60)

    for N in N_values:
        for M in null_counts:
            opportunity_losses = []

            for _ in range(num_trials):
                target_u = np.random.uniform(-0.5, 0.5)

                nulls = []
                while len(nulls) < M:
                    cand = np.random.uniform(-0.8, 0.8)
                    if abs(cand - target_u) > 0.15 and all(abs(cand - n) > 0.1 for n in nulls):
                        nulls.append(cand)

                task = Task(type='nulled', target_angles=[target_u], null_angles=nulls)

                pre_gains = []
                post_ptnrs = []

                for psi in psi_grid:
                    V_n, c_n, _ = solve_coherent_lcmv(
                        task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
                    )

                    pre_gain = 20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12)
                    pre_gains.append(pre_gain)

                    V_post, c_post = gain_locked_polish(
                        c_n, V_n, task, element, null_angles=nulls, rounds=6, gain_floor=0.995
                    )

                    post_gain = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)

                    # Average null depth for multi-null tasks
                    null_depths = [20 * np.log10(abs(_af(c_post, nu, N)) + 1e-12) for nu in nulls]
                    avg_null_depth = np.mean(null_depths)

                    post_ptnrs.append(post_gain - avg_null_depth)

                best_psi_gain_idx = np.argmax(pre_gains)
                ptnr_if_we_picked_by_gain = post_ptnrs[best_psi_gain_idx]
                actual_max_post_ptnr = np.max(post_ptnrs)

                opportunity_losses.append(actual_max_post_ptnr - ptnr_if_we_picked_by_gain)

            median_loss = np.median(opportunity_losses)
            mean_loss = np.mean(opportunity_losses)
            print(f"{N}\t{M}\t{median_loss:.2f}\t\t\t{mean_loss:.2f}")

if __name__ == "__main__":
    run_scaling_experiment()

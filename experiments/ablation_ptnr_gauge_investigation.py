"""
Investigates the statistical opportunity loss of using a gain gauge vs a PTNR gauge.

Runs a large number of random nulled beamforming tasks and quantifies the median dB loss
encountered when selecting the global offset `psi` based on initial gain vs final PTNR.
This supports the ablation study claims regarding the ~13 dB PTNR improvement.
"""
import os
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv
from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task


def run_experiment(num_trials=50, N=64):
    """Run statistical trials comparing PTNR and gain gauges."""
    np.random.seed(42)
    element = SyntheticVaractor()

    correlations = []
    opportunity_losses = []

    psi_grid = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    for i in range(num_trials):
        target_u = np.random.uniform(-0.5, 0.5)
        null_u = np.random.uniform(-0.8, 0.8)
        while abs(null_u - target_u) < 0.15:
            null_u = np.random.uniform(-0.8, 0.8)

        task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])

        pre_ptnrs = []
        post_ptnrs = []
        pre_gains = []

        for psi in psi_grid:
            V_n, c_n, _ = solve_coherent_lcmv(
                task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
            )

            pre_gain = 20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12)
            pre_null = 20 * np.log10(abs(_af(c_n, null_u, N)) + 1e-12)
            pre_ptnrs.append(pre_gain - pre_null)
            pre_gains.append(pre_gain)

            V_post, c_post = gain_locked_polish(
                c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995
            )

            post_gain = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)
            post_null = 20 * np.log10(abs(_af(c_post, null_u, N)) + 1e-12)
            post_ptnrs.append(post_gain - post_null)

        # 1. Pearson correlation between pre-GLCP PTNR and post-GLCP PTNR
        corr, _ = pearsonr(pre_ptnrs, post_ptnrs)
        correlations.append(corr)

        # 2. Opportunity loss: If we picked psi based on max pre-GLCP gain vs max post-GLCP PTNR
        best_psi_gain_idx = np.argmax(pre_gains)
        ptnr_if_we_picked_by_gain = post_ptnrs[best_psi_gain_idx]
        actual_max_post_ptnr = np.max(post_ptnrs)

        opportunity_losses.append(actual_max_post_ptnr - ptnr_if_we_picked_by_gain)

    print(f"Mean Correlation (pre-GLCP PTNR vs post-GLCP PTNR): {np.mean(correlations):.3f}")
    print(f"Median Opportunity Loss (using gain gauge vs PTNR gauge): {np.median(opportunity_losses):.2f} dB")
    print(f"Mean Opportunity Loss: {np.mean(opportunity_losses):.2f} dB")


if __name__ == "__main__":
    run_experiment()

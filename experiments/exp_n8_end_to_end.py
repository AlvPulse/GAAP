import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, solve_coherent_lcmv, gain_locked_polish
from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task


def run_n8_test():
    np.random.seed(42)
    N = 8
    element = SyntheticVaractor(beta=1.0, folding=True) # Full measured-like VDIL
    psi_grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    results = []

    print(f"Running End-to-End Test for N={N} on severe VDIL hardware...")

    for trial in range(30):
        target_u = np.random.uniform(-0.4, 0.4)
        null_u = np.random.uniform(-0.8, 0.8)
        while abs(null_u - target_u) < 0.25:
            null_u = np.random.uniform(-0.8, 0.8)

        task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])

        pre_gains = []
        post_ptnrs = []
        best_post_null = -10.0

        for psi in psi_grid:
            V_n, c_n, _ = solve_coherent_lcmv(
                task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
            )
            pre_gains.append(20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12))

            V_post, c_post = gain_locked_polish(
                c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995
            )

            g = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)
            n_db = 20 * np.log10(abs(_af(c_post, null_u, N)) + 1e-12)
            post_ptnrs.append(g - n_db)

            if n_db < best_post_null:
                best_post_null = n_db

        ptnr_gain_gauge = post_ptnrs[np.argmax(pre_gains)]
        ptnr_ptnr_gauge = np.max(post_ptnrs)

        results.append({
            "Trial": trial,
            "Target (u)": target_u,
            "Null (u)": null_u,
            "Gain Gauge PTNR (dB)": ptnr_gain_gauge,
            "PTNR Gauge PTNR (dB)": ptnr_ptnr_gauge,
            "Opportunity Loss (dB)": ptnr_ptnr_gauge - ptnr_gain_gauge,
            "Best Null (dB)": best_post_null
        })

    df = pd.DataFrame(results)

    print("\n--- Summary ---")
    print(f"Median Gain Gauge PTNR: {df['Gain Gauge PTNR (dB)'].median():.2f} dB")
    print(f"Median PTNR Gauge PTNR: {df['PTNR Gauge PTNR (dB)'].median():.2f} dB")
    print(f"Median Opportunity Loss: {df['Opportunity Loss (dB)'].median():.2f} dB")
    print(f"Median Achieved Null: {df['Best Null (dB)'].median():.2f} dB")

    df.to_csv("experiments/n8_end_to_end_results.csv", index=False)

if __name__ == "__main__":
    run_n8_test()

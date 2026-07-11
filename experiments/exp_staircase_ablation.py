"""
Experiment: Staircase Ablation (Anchor -> PTNR Gauge -> GLCP)

This script generates the 4-stage sequential ablation requested by the reviewer:
Stage 1: Anchor (Gain Gauge)
Stage 2: Anchor + PTNR Gauge (No Refinement)
Stage 3: Anchor + Gain Gauge + GLCP
Stage 4: Full (Anchor + PTNR Gauge + GLCP)

It runs 50 random tasks and outputs a summary CSV and a boxplot to cleanly
tell the story of how each layer contributes to the final PTNR.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv
from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task

def run_staircase_ablation(num_trials=50, N=64):
    np.random.seed(42)
    element = SyntheticVaractor()
    psi_grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    records = []

    for task_id in range(num_trials):
        target_u = np.random.uniform(-0.5, 0.5)
        null_u = np.random.uniform(-0.8, 0.8)
        while abs(null_u - target_u) < 0.15:
            null_u = np.random.uniform(-0.8, 0.8)

        task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])

        pre_gains = []
        pre_ptnrs = []
        post_ptnrs = []

        # We need the full landscape to simulate the gauges
        for psi in psi_grid:
            V_n, c_n, _ = solve_coherent_lcmv(
                task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
            )
            pre_gain = 20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12)
            pre_null = 20 * np.log10(abs(_af(c_n, null_u, N)) + 1e-12)
            pre_gains.append(pre_gain)
            pre_ptnrs.append(pre_gain - pre_null)

            V_post, c_post = gain_locked_polish(
                c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995
            )
            post_gain = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)
            post_null = 20 * np.log10(abs(_af(c_post, null_u, N)) + 1e-12)
            post_ptnrs.append(post_gain - post_null)

        # The Gauges
        best_gain_idx = np.argmax(pre_gains)
        best_pre_ptnr_idx = np.argmax(pre_ptnrs) # If we gauge by pre-refine PTNR
        best_post_ptnr_idx = np.argmax(post_ptnrs) # The true PTNR Gauge (post-refine)

        # 1. Anchor (Gain Gauge)
        records.append({"Task": task_id, "Stage": "1. Anchor (Gain Gauge)", "PTNR": pre_ptnrs[best_gain_idx]})

        # 2. Anchor + PTNR Gauge (Pre-refine)
        records.append({"Task": task_id, "Stage": "2. Anchor + PTNR Gauge", "PTNR": pre_ptnrs[best_pre_ptnr_idx]})

        # 3. Anchor + Gain Gauge + GLCP
        records.append({"Task": task_id, "Stage": "3. Anchor + GLCP (Gain Gauge)", "PTNR": post_ptnrs[best_gain_idx]})

        # 4. Full (Anchor + GLCP + PTNR Gauge)
        records.append({"Task": task_id, "Stage": "4. Full Controller (PTNR Gauge)", "PTNR": post_ptnrs[best_post_ptnr_idx]})

    return pd.DataFrame(records)

def main():
    print("Running Staircase Ablation...")
    df = run_staircase_ablation(num_trials=50, N=64)

    # Calculate Medians
    medians = df.groupby("Stage")["PTNR"].median().reset_index()
    print("\n--- Sequential Ablation Medians ---")
    print(medians.to_string(index=False))
    medians.to_csv("experiments/staircase_ablation_medians.csv", index=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="Stage", y="PTNR", data=df, palette="Blues", width=0.5)
    sns.stripplot(x="Stage", y="PTNR", data=df, color="black", alpha=0.3, jitter=True)

    plt.title("Sequential Ablation of Beamforming Stages")
    plt.ylabel("Peak-to-Null Ratio (PTNR) [dB]")
    plt.xlabel("")
    plt.xticks(rotation=15)
    plt.grid(axis='y', alpha=0.5)
    plt.tight_layout()
    plt.savefig("experiments/staircase_ablation.png", dpi=300)

if __name__ == "__main__":
    main()

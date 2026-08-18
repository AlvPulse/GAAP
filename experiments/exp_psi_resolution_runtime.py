"""
Experiment: Offset Resolution Runtime vs Performance

Demonstrates that the opportunity loss recovery saturates rapidly with the number
of psi candidates, and that the offset search contributes trivially to the overall
computational runtime.
"""
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.pattern_projection import Task


def run_resolution_test(N=64, num_trials=10):
    np.random.seed(42)
    element = MeasuredVaractor()

    resolutions = [4, 8, 16, 32, 64, 128]

    results = []

    for res in resolutions:
        print(f"Testing Psi Resolution: {res} candidates...")
        psi_grid = np.linspace(0, 2 * np.pi, res, endpoint=False)

        opp_losses = []
        anchor_times = []
        glcp_times = []

        for _ in range(num_trials):
            target_u = np.random.uniform(-0.5, 0.5)
            null_u = np.random.uniform(-0.8, 0.8)
            while abs(null_u - target_u) < 0.15:
                null_u = np.random.uniform(-0.8, 0.8)

            task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])
            pre_gains = []
            post_ptnrs = []

            t0 = time.perf_counter()
            for psi in psi_grid:
                V_n, c_n, _ = solve_coherent_lcmv(
                    task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
                )
                pre_gains.append(20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12))
            t1 = time.perf_counter()
            anchor_times.append((t1 - t0) / res) # Avg time per anchor

            t2 = time.perf_counter()
            # In a real run, we only GLCP the top candidates, but to measure the landscape we GLCP all here.
            # To simulate the actual algorithm's runtime cost, we measure the time for ONE GLCP run.
            V_n_base, c_n_base, _ = solve_coherent_lcmv(task, element, N, n_dual_steps=8, psi_grid=[psi_grid[0]], return_history=True)
            gain_locked_polish(c_n_base, V_n_base, task, element, null_angles=[null_u], rounds=6)
            t3 = time.perf_counter()
            glcp_times.append(t3 - t2)

            # Compute full landscape for accuracy
            for psi in psi_grid:
                V_n, c_n, _ = solve_coherent_lcmv(task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True)
                V_post, c_post = gain_locked_polish(c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995)
                g = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)
                n_db = 20 * np.log10(abs(_af(c_post, null_u, N)) + 1e-12)
                post_ptnrs.append(g - n_db)

            loss = np.max(post_ptnrs) - post_ptnrs[np.argmax(pre_gains)]
            opp_losses.append(loss)

        # In a real deployed system, we evaluate `res` anchors, and run `1` GLCP on the best.
        avg_anchor = np.mean(anchor_times) * 1000
        avg_glcp = np.mean(glcp_times) * 1000
        search_time_ms = avg_anchor * res
        total_time_ms = search_time_ms + avg_glcp
        overhead_pct = (search_time_ms / total_time_ms) * 100

        results.append({
            "Psi Candidates": res,
            "Median Opp. Loss (dB)": np.median(opp_losses),
            "Search Time (ms)": search_time_ms,
            "Total Time (ms)": total_time_ms,
            "Overhead %": overhead_pct
        })

    df = pd.DataFrame(results)
    print("\n--- Resolution vs Runtime ---")
    print(df.to_string(index=False))
    df.to_csv("experiments/psi_resolution_runtime.csv", index=False)

if __name__ == "__main__":
    run_resolution_test()

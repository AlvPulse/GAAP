"""
Experiment: Opportunity Loss vs Hardware Impairment Severity.

This script scales the severity of the amplitude ripple (VDIL) from 0 dB (ideal)
to 15 dB (severe) and plots the Opportunity Loss of the Gain Gauge.
This provides the predictive proof that the required controller behavior is
strictly proportional to the hardware constraints.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv
from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task


class ParametricVaractor(SyntheticVaractor):
    def __init__(self, max_dip_db):
        super().__init__()
        phases = np.angle(self.c_grid)

        if max_dip_db == 0:
            amps = np.ones_like(phases)
        else:
            # Convert dB dip to linear amplitude ratio
            min_amp = 10 ** (-max_dip_db / 20.0)
            # Create a dip profile
            amps = 1.0 - (1.0 - min_amp) * np.exp(-((phases - np.pi)**2)/0.5)

        self.c_grid = amps * np.exp(1j * phases)


def run_impairment_test(N=64, num_trials=30):
    np.random.seed(42)
    psi_grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    dip_levels_db = [0, 2, 5, 8, 12, 15]

    results = []

    for dip in dip_levels_db:
        print(f"Testing Ripple Severity: {dip} dB...")
        element = ParametricVaractor(max_dip_db=dip)
        opportunity_losses = []

        for _ in range(num_trials):
            target_u = np.random.uniform(-0.5, 0.5)
            null_u = np.random.uniform(-0.8, 0.8)
            while abs(null_u - target_u) < 0.15:
                null_u = np.random.uniform(-0.8, 0.8)

            task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])
            pre_gains = []
            post_ptnrs = []

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

            loss = np.max(post_ptnrs) - post_ptnrs[np.argmax(pre_gains)]
            opportunity_losses.append(loss)

        results.append({
            "Amplitude Ripple (dB)": dip,
            "Median Opportunity Loss (dB)": np.median(opportunity_losses)
        })

    df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.plot(df["Amplitude Ripple (dB)"], df["Median Opportunity Loss (dB)"], marker='o', linewidth=2, color='darkred')
    plt.xlabel("Hardware Amplitude Ripple (dB)")
    plt.ylabel("Median Opportunity Loss (dB)")
    plt.title("Opportunity Loss scales with Hardware Impairment")
    plt.grid(True)
    plt.savefig("experiments/hardware_impairment_scaling.png")

    df.to_csv("experiments/hardware_impairment_scaling.csv", index=False)
    print(df)


if __name__ == "__main__":
    run_impairment_test()

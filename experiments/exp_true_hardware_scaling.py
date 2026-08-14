"""
Experiment: True Hardware Scaling (Resolving the Reviewer Contradiction).

The previous impairment script only flattened the amplitude ripple, but left the
phase spacing non-uniform. Because the phase grid was non-uniform, Theorem 1
still applied, and Opportunity Loss remained at ~12 dB.

This script correctly interpolates the hardware manifold from a Severe VDIL
Varactor (non-uniform phase, 15 dB amplitude dip) to a perfectly Ideal Phase
Shifter (uniform phase, 0 dB dip).

We will prove that as the manifold becomes rotationally symmetric (uniform),
the projection commutativity is restored, and Opportunity Loss strictly approaches 0 dB.
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


class TrueParametricVaractor(SyntheticVaractor):
    def __init__(self, alpha):
        """
        alpha: 0.0 = Ideal Uniform Phase Shifter (32 states)
               1.0 = Severe VDIL Varactor (non-uniform phase, deep amplitude dip)
        """
        super().__init__()

        # 1. Base Severe Varactor properties
        severe_phases = np.angle(self.c_grid)
        # Deep notch for amplitude
        severe_amps = 1.0 - 0.82 * np.exp(-((severe_phases - np.pi)**2)/0.5)

        # 2. Base Ideal Phase Shifter properties
        ideal_phases = np.linspace(0, 2 * np.pi, len(self.c_grid), endpoint=False)
        ideal_amps = np.ones_like(ideal_phases)

        # 3. Interpolate both Amplitude AND Phase Uniformity based on Alpha
        interp_phases = (1 - alpha) * ideal_phases + alpha * severe_phases
        interp_amps = (1 - alpha) * ideal_amps + alpha * severe_amps

        self.c_grid = interp_amps * np.exp(1j * interp_phases)


def run_true_scaling_test(N=64, num_trials=30):
    np.random.seed(42)
    psi_grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    alpha_levels = [0.0, 0.2, 0.5, 0.8, 1.0]

    results = []

    for alpha in alpha_levels:
        print(f"Testing Hardware Impairment Alpha: {alpha:.1f} (0=Ideal, 1=Severe)...")
        element = TrueParametricVaractor(alpha=alpha)
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
            "Impairment Alpha": alpha,
            "Median Opportunity Loss (dB)": np.median(opportunity_losses)
        })

    df = pd.DataFrame(results)

    plt.figure(figsize=(8, 5))
    plt.plot(df["Impairment Alpha"], df["Median Opportunity Loss (dB)"], marker='o', linewidth=2, color='purple')
    plt.xlabel("Hardware Impairment Factor (0=Ideal Uniform, 1=Severe VDIL)")
    plt.ylabel("Median Opportunity Loss (dB)")
    plt.title("Opportunity Loss vs True Hardware Non-Uniformity")
    plt.grid(True)
    plt.savefig("experiments/true_hardware_scaling.png")

    df.to_csv("experiments/true_hardware_scaling.csv", index=False)
    print("\n--- Resolved Hardware Scaling Results ---")
    print(df.to_string(index=False))


if __name__ == "__main__":
    run_true_scaling_test()

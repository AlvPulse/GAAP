"""
Experiment: Hardware Diversity (Co-Design Proof)

This script tests the opportunity loss of using the Gain Gauge vs PTNR Gauge
across three distinct hardware profiles:
1. Ideal Phase Shifters (Flat amplitude, 5-bit phase)
2. Mild VDIL Varactor (Slight amplitude/phase coupling)
3. Severe VDIL Varactor (Deep amplitude dips associated with phase changes)

This is the ultimate proof of hardware-algorithm co-design: the benefit of the
PTNR gauge scales directly with the severity of the hardware constraint.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv, solve_coherent_lcmv_convergence
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.pattern_projection import Task


class IdealPhaseShifter:
    """Ideal 5-bit phase shifter (32 states, uniform amplitude)."""
    def __init__(self):
        phases = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        self.c_grid = np.exp(1j * phases)
        self.V_grid = np.arange(32)

class MildVaractor(SyntheticVaractor):
    """Varactor with very minor amplitude loss (max ~1 dB dip)."""
    def __init__(self):
        super().__init__()
        # Reduce the amplitude dip significantly
        phases = np.angle(self.c_grid)
        amps = np.ones_like(phases) - 0.1 * np.sin(phases/2)**2
        self.c_grid = amps * np.exp(1j * phases)

class SevereVaractor(SyntheticVaractor):
    """Varactor with extreme amplitude loss (max ~15 dB dip)."""
    def __init__(self):
        super().__init__()
        phases = np.angle(self.c_grid)
        # Deep notch
        amps = np.ones_like(phases) - 0.82 * np.exp(-((phases - np.pi)**2)/0.5)
        self.c_grid = amps * np.exp(1j * phases)


def run_hardware_test(element, N, num_trials=30):
    np.random.seed(42)
    psi_grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)
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
            # V_n, c_n, _ = solve_coherent_lcmv(
            #     task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
            # )

            V_n, c_n, _ = solve_coherent_lcmv_convergence(
                            task, element, N, psi_grid=[psi], return_history=True
                        )
            pre_gain = 20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12)
            pre_gains.append(pre_gain)

            V_post, c_post = gain_locked_polish(
                c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995
            )
            post_gain = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)
            post_null = 20 * np.log10(abs(_af(c_post, null_u, N)) + 1e-12)
            post_ptnrs.append(post_gain - post_null)

        best_psi_gain_idx = np.argmax(pre_gains)
        ptnr_if_we_picked_by_gain = post_ptnrs[best_psi_gain_idx]
        actual_max_post_ptnr = np.max(post_ptnrs)
        opportunity_losses.append(actual_max_post_ptnr - ptnr_if_we_picked_by_gain)

    return opportunity_losses


def main():
    print("Running Hardware Diversity Experiment (N=64, 1 Null)...")
    N = 64
    hw_profiles = {
        "Ideal (5-bit)": IdealPhaseShifter(),
        "Mild VDIL": MildVaractor(),
        "Severe VDIL": SevereVaractor(),
        "Real_hardware": MeasuredVaractor()
    }

    results = []
    for name, element in hw_profiles.items():
        print(f"Testing {name}...")
        losses = run_hardware_test(element, N, num_trials=30)
        median_loss = np.median(losses)
        mean_loss = np.mean(losses)
        results.append({
            "Hardware Profile": name,
            "Median Opportunity Loss (dB)": median_loss,
            "Mean Opportunity Loss (dB)": mean_loss
        })

    df = pd.DataFrame(results)
    print("\n--- Results ---")
    print(df.to_string(index=False))
    df.to_csv("experiments/hardware_diversity_results.csv", index=False)

if __name__ == "__main__":
    main()

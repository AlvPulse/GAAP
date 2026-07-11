"""
Experiment: 2x2 Factorial Ablation of Hardware Impairments

This experiment decomposes the Opportunity Loss (L) into its core hardware constituents,
verifying Theorem 1:
- Cell A: beta=0, folding=False (Ideal uniform phase shifter) -> L_floor
- Cell B: beta=0, folding=True  (Pure phase folding) -> L_floor + L_folding
- Cell C: beta=1, folding=False (Pure amplitude coupling) -> L_floor + L_amplitude
- Cell D: beta=1, folding=True  (Full VDIL) -> L_total
"""
import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv
from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task


# Custom Varactor to cleanly isolate the effects without the beta-scaling bug
class FactorialVaractor(SyntheticVaractor):
    def __init__(self, beta=1.0, folding_enabled=True):
        super().__init__(beta=beta, folding=folding_enabled)
        # Override the manifold generation to guarantee strict isolation
        v_norm = (self.V_grid - self.v_min) / (self.v_max - self.v_min)

        # NOTE: 2 * np.pi * v_norm * 1.05 goes to 2.1 pi, meaning the grid points wrap around unequally.
        # This causes the phase spacing to be inherently non-uniform even WITHOUT folding.
        # To truly test an Ideal Uniform Element as the baseline (Cell A), the phase must wrap perfectly.
        phi = 2 * np.pi * v_norm

        # 1. Phase Folding Isolation
        if folding_enabled:
            # Here we apply folding without scaling it by beta, so we can test beta=0 but folding=True
            phi = phi + 0.8 * np.sin(2 * np.pi * v_norm * 1.2)

        # 2. Amplitude Coupling Isolation
        # If beta=0, A=1. If beta=1, we have a deep notch
        # We also need to map the insertion loss cleanly onto the 0 to 2pi domain
        # so that without folding, the notch still exists around pi.
        A = 1.0 - beta * 0.8 * np.exp(-1 * ((phi - np.pi)/np.pi)**2)
        A = np.clip(A, 0.05, 1.0)

        self.c_grid = A * np.exp(1j * phi)


def run_condition(beta, folding, N=64, num_trials=50):
    element = FactorialVaractor(beta=beta, folding_enabled=folding)
    psi_grid = np.linspace(0, 2 * np.pi, 24, endpoint=False)

    losses = []
    for _ in range(num_trials):
        target_u = np.random.uniform(-0.5, 0.5)
        null_u = np.random.uniform(-0.8, 0.8)
        while abs(null_u - target_u) < 0.15:
            null_u = np.random.uniform(-0.8, 0.8)

        task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])
        pre_gains = []
        post_ptnrs = []

        # When evaluating the ideal uniform grid, we must be careful with solve_coherent_lcmv
        # The solver uses n_dual_steps which can occasionally overfit the null if the grid is super dense
        # But here we are just measuring the pure Opportunity Loss L
        for psi in psi_grid:
            V_n, c_n, _ = solve_coherent_lcmv(task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True)
            pre_gains.append(20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12))

            V_post, c_post = gain_locked_polish(c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995)
            g = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)
            n_db = 20 * np.log10(abs(_af(c_post, null_u, N)) + 1e-12)
            post_ptnrs.append(g - n_db)

        losses.append(np.max(post_ptnrs) - post_ptnrs[np.argmax(pre_gains)])

    return np.median(losses)

def main():
    np.random.seed(42)
    print("Running 2x2 Factorial Ablation of Hardware Impairments...")

    # Cell A
    print("Cell A (Ideal Element: beta=0, folding=False)...")
    L_A = run_condition(beta=0.0, folding=False)

    # Cell B
    print("Cell B (Pure Phase Folding: beta=0, folding=True)...")
    L_B = run_condition(beta=0.0, folding=True)

    # Cell C
    print("Cell C (Pure Amplitude Coupling: beta=1, folding=False)...")
    L_C = run_condition(beta=1.0, folding=False)

    # Cell D
    print("Cell D (Full VDIL Varactor: beta=1, folding=True)...")
    L_D = run_condition(beta=1.0, folding=True)

    print("\n=== 2x2 Factorial Results (Median Opportunity Loss in dB) ===")
    print(f"Cell A (Ideal Uniform Grid)      : {L_A:.2f} dB  (L_floor)")
    print(f"Cell B (Pure Phase Folding)      : {L_B:.2f} dB")
    print(f"Cell C (Pure Amplitude Coupling) : {L_C:.2f} dB")
    print(f"Cell D (Full VDIL Varactor)      : {L_D:.2f} dB  (L_total)")

    print("\n--- Decomposition ---")
    print(f"L_floor       = {L_A:.2f} dB")
    print(f"L_folding     = {L_B - L_A:.2f} dB")
    print(f"L_amplitude   = {L_C - L_A:.2f} dB")
    print(f"Interaction   = {L_D - (L_A + (L_B - L_A) + (L_C - L_A)):.2f} dB")

    # Save to CSV
    df = pd.DataFrame([
        {"Condition": "A: Ideal", "Beta": 0, "Folding": False, "Median Loss (dB)": L_A},
        {"Condition": "B: Phase Folding Only", "Beta": 0, "Folding": True, "Median Loss (dB)": L_B},
        {"Condition": "C: Amplitude Coupling Only", "Beta": 1, "Folding": False, "Median Loss (dB)": L_C},
        {"Condition": "D: Full VDIL", "Beta": 1, "Folding": True, "Median Loss (dB)": L_D},
    ])
    df.to_csv("experiments/2x2_factorial_ablation.csv", index=False)

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.baseline_optimization import optimize_offset_only
from beamformer.geometry import scan_offset_feasibility, compute_aperture_potential
from beamformer.cost_functions import evaluate_pattern_cost
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.utils import oversampled_fft

def main():
    print("Running Diagnostic Geometry Scan...")

    # 1. Setup Task
    N = 64
    task = Task('nulled', target_angles=[0.2], null_angles=[-0.4, 0.5], sll_ceiling=0.1)

    # Check if measured files exist, otherwise fallback to synthetic
    if os.path.exists('amplitude.mat') and os.path.exists('phase.mat'):
        print("Using measured data: amplitude.mat, phase.mat")
        element = MeasuredVaractor(amp_file='amplitude.mat', phase_file='phase.mat')
    else:
        print("Measured files not found. Using SyntheticVaractor fallback.")
        element = SyntheticVaractor(beta=0.8, folding=True)

    # Initial Ideal Weights for Nulling
    initial_weights = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    # 2. Run Sweep
    n_steps = 180 # 2 degree increments
    deltas = np.linspace(0, 2*np.pi, n_steps)

    e_effs = np.zeros(n_steps)
    costs = np.zeros(n_steps)
    residuals = np.zeros(n_steps)
    null_depths = np.zeros(n_steps)
    gains = np.zeros(n_steps)
    branch_counts = np.zeros(n_steps)

    print(f"Sweeping delta across {n_steps} points...")
    for i, delta in enumerate(deltas):
        # Rotate ideal
        rotated_ideal = initial_weights * np.exp(1j * delta)

        # Project
        c_proj = np.zeros(N, dtype=np.complex128)
        V_proj = np.zeros(N)
        res = 0.0
        for n in range(N):
            V_proj[n], c_proj[n], _= element.project(rotated_ideal[n], method='phase_only')
            res += np.abs(rotated_ideal[n] - c_proj[n])**2

        # De-rotate for pattern evaluation
        c_eval = c_proj * np.exp(-1j * delta)

        # Calculate Metrics
        e_effs[i] = compute_aperture_potential(c_proj)
        costs[i] = evaluate_pattern_cost(c_eval, task, oversample_factor=16)
        residuals[i] = res

        # Pattern metrics
        u, F = oversampled_fft(c_eval, oversample_factor=16)
        power_dB = 20 * np.log10(np.abs(F) + 1e-12)

        # Absolute gain calculation (max ideal = N)
        max_ideal_db = 20 * np.log10(N)

        target_idx = np.argmin(np.abs(u - task.target_angles[0]))
        gains[i] = power_dB[target_idx] - max_ideal_db

        nd = []
        for nu in task.null_angles:
            idx = np.argmin(np.abs(u - nu))
            nd.append(power_dB[idx] - max_ideal_db)
        null_depths[i] = np.mean(nd)

        # Branch proxy: count number of elements forced into a specific voltage band
        # (e.g., V > 7.5 assuming halfway folding)
        branch_counts[i] = np.sum(V_proj > 7.5)

    # 3. Plotting
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

    axs[0].plot(np.rad2deg(deltas), e_effs, 'b')
    axs[0].set_ylabel('Aperture Potential\n$E_{eff}$')
    axs[0].set_title('Does Global Offset Control Geometry? (Static Sweep)')
    axs[0].grid(True)

    axs[1].plot(np.rad2deg(deltas), residuals, 'r')
    axs[1].set_ylabel('Euclidean\nResidual')
    axs[1].grid(True)

    axs[2].plot(np.rad2deg(deltas), null_depths, 'g')
    axs[2].set_ylabel('Avg Null Depth\n(dB)')
    axs[2].grid(True)

    axs[3].plot(np.rad2deg(deltas), branch_counts, 'orange')
    axs[3].set_ylabel('Elements on\nHigh Branch')
    axs[3].grid(True)

    axs[4].plot(np.rad2deg(deltas), gains, 'purple')
    axs[4].set_ylabel('Main Lobe\nGain (dB)')
    axs[4].set_xlabel('Global Offset $\delta$ (Degrees)')
    axs[4].grid(True)

    plt.tight_layout()
    plt.savefig('diagnostic_results.png')
    print("Saved static geometry scan to diagnostic_results.png")

    print("\nDiagnostic Summary:")
    print(f"E_eff Variation:      {np.max(e_effs) - np.min(e_effs):.3f}")
    print(f"Gain Variation:       {np.max(gains) - np.min(gains):.2f} dB")
    print(f"Null Depth Variation: {np.max(null_depths) - np.min(null_depths):.2f} dB")

if __name__ == '__main__':
    main()

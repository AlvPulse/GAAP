import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer.api import beamform
from beamformer.utils import oversampled_fft

def main():
    print("Running Pencil Beam Experiment...")

    # 1. Setup
    N = 64
    task = Task('pencil', target_angles=[0.2], sll_ceiling=0.3) # -20 dB SLL
    element = SyntheticVaractor(beta=0.9, folding=True) # Severe non-linearity

    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_results')

    projection_methods = ['euclidean', 'phase_only', 'weighted']
    results = {}

    # 2. Run Geometry-Aware AP for each projection method
    for method in projection_methods:
        print(f"Starting AP Optimization for method: {method}... (Debugging output to {debug_dir}_{method})")
        res = beamform(
            task,
            element,
            N=N,
            K_max=50,
            debug_dir=f"{debug_dir}_{method}",
            projection_method=method,
            w_phase=1.0,
            w_amp=0.5
        )
        results[method] = res
        print(f"[{method}] Final Residual: {res['history']['residuals'][-1]:.4f}")

    # 3. Analyze Results & Plotting
    plt.figure(figsize=(12, 6))

    u_ideal, F_ideal = oversampled_fft(results['euclidean']['initial_weights'], oversample_factor=16)
    ideal_dB = 20 * np.log10(np.abs(F_ideal) + 1e-12)
    plt.plot(u_ideal, ideal_dB - np.max(ideal_dB), '--', color='gray', label='Coherent Baseline')

    colors = {'euclidean': 'b', 'phase_only': 'r', 'weighted': 'g'}

    for method in projection_methods:
        u, F = oversampled_fft(results[method]['weights'], oversample_factor=16)
        power_dB = 20 * np.log10(np.abs(F) + 1e-12)
        plt.plot(u, power_dB - np.max(power_dB), color=colors[method], label=f'AP ({method})')

    plt.axhline(20*np.log10(task.sll_ceiling), color='k', linestyle=':', label='SLL Ceiling')
    plt.axvline(0.5, color='orange', linestyle=':', label='Target')
    plt.ylim(-40, 5)
    plt.xlim(-1, 1)
    plt.xlabel('u = sin(theta)')
    plt.ylabel('Normalized Pattern (dB)')
    plt.title(f'Pencil Beam (N={N}) Projection Method Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('pencil_beam_result.png')
    print("Saved plot to pencil_beam_result.png")

if __name__ == '__main__':
    main()

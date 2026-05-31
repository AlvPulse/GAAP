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
    print("Running Nulled Beam Experiment Projection Comparison...")

    # 1. Setup
    N = 64
    task = Task('nulled', target_angles=[0.2], null_angles=[-0.4, 0.5])
    element = SyntheticVaractor(beta=0.8, folding=False) # Severe non-linearity

    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_results_null')

    projection_methods = ['euclidean', 'phase_only', 'weighted']
    results = {}

    # 2. Run Geometry-Aware AP for each projection method
    for method in projection_methods:
        print(f"Starting AP Optimization for method: {method}... (Debugging output to {debug_dir}_{method})")
        res = beamform(
            task,
            element,
            N=N,
            K_max=80,
            debug_dir=f"{debug_dir}_{method}",
            null_init_method='schelkunoff',
            use_pattern_cost=True,
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
    plt.plot(u_ideal, ideal_dB - np.max(ideal_dB), '--', color='gray', label='Schelkunoff Init')

    colors = {'euclidean': 'b', 'phase_only': 'r', 'weighted': 'g'}

    for method in projection_methods:
        u, F = oversampled_fft(results[method]['weights'], oversample_factor=16)
        power_dB = 20 * np.log10(np.abs(F) + 1e-12)
        plt.plot(u, power_dB - np.max(power_dB), color=colors[method], label=f'AP ({method})')

    plt.axvline(0.2, color='orange', linestyle=':', label='Target')
    for nu in task.null_angles:
        plt.axvline(nu, color='k', linestyle='-.', label='Null Constraint' if nu == task.null_angles[0] else "")

    plt.ylim(-60, 5)
    plt.xlim(-1, 1)
    plt.xlabel('u = sin(theta)')
    plt.ylabel('Normalized Pattern (dB)')
    plt.title(f'Nulled Beam (N={N}) Projection Method Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('nullforming_result.png')
    print("Saved plot to nullforming_result.png")

if __name__ == '__main__':
    main()

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
    print("Running Nulled Beam Experiment with Cost Function Offset Optimization...")

    # 1. Setup
    N = 64
    task = Task('nulled', target_angles=[0.2], null_angles=[-0.4, 0.5])
    element = SyntheticVaractor(beta=0.8, folding=True) # Severe non-linearity

    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_results_null')

    # 2. Run Geometry-Aware AP
    print(f"Starting AP Optimization... (Debugging output to {debug_dir})")

    # Using schelkunoff init and pattern cost function
    res = beamform(
        task,
        element,
        N=N,
        K_max=80,
        debug_dir=debug_dir,
        null_init_method='schelkunoff',
        use_pattern_cost=True
    )

    # 3. Analyze Results
    print(f"Final Residual: {res['history']['residuals'][-1]:.4f}")

    u, F = oversampled_fft(res['weights'], oversample_factor=16)
    power_dB = 20 * np.log10(np.abs(F) + 1e-12)

    u_ideal, F_ideal = oversampled_fft(res['initial_weights'], oversample_factor=16)
    ideal_dB = 20 * np.log10(np.abs(F_ideal) + 1e-12)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(u_ideal, ideal_dB - np.max(ideal_dB), '--', color='gray', label='Schelkunoff Init')
    plt.plot(u, power_dB - np.max(power_dB), 'b', label='Geometry-Aware AP')

    plt.axvline(0.2, color='green', linestyle=':', label='Target')
    for nu in task.null_angles:
        plt.axvline(nu, color='red', linestyle='-.', label='Null Constraint' if nu == task.null_angles[0] else "")

    plt.ylim(-60, 5)
    plt.xlim(-1, 1)
    plt.xlabel('u = sin(theta)')
    plt.ylabel('Normalized Pattern (dB)')
    plt.title(f'Nulled Beam (N={N}) with VDIL Varactor Constraints (Pattern Cost)')
    plt.legend()
    plt.grid(True)
    plt.savefig('nullforming_result.png')
    print("Saved plot to nullforming_result.png")

if __name__ == '__main__':
    main()

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
    N = 16
    task = Task('pencil', target_angles=[0.2,-0.3], sll_ceiling=0.1) # -20 dB SLL
    element = SyntheticVaractor(beta=0.7, folding=False) # Severe non-linearity

    debug_dir = os.path.join(os.path.dirname(__file__), 'debug_results')

    # 2. Run Geometry-Aware AP
    print(f"Starting AP Optimization... (Debugging output to {debug_dir})")
    res = beamform(task, element, N=N, K_max=550, debug_dir=debug_dir)

    # 3. Analyze Results
    print(f"Final Residual: {res['history']['residuals'][-1]:.4f}")

    u, F = oversampled_fft(res['weights'], oversample_factor=16)
    power_dB = 20 * np.log10(np.abs(F) + 1e-12)

    u_ideal, F_ideal = oversampled_fft(res['initial_weights'], oversample_factor=16)
    ideal_dB = 20 * np.log10(np.abs(F_ideal) + 1e-12)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(u_ideal, ideal_dB - np.max(ideal_dB), '--', color='gray', label='Coherent Baseline')
    plt.plot(u, power_dB - np.max(power_dB), 'b', label='Geometry-Aware AP')
    plt.axhline(20*np.log10(task.sll_ceiling), color='r', linestyle=':', label='SLL Ceiling')
    plt.axvline(0.5, color='green', linestyle=':', label='Target')
    plt.ylim(-40, 5)
    plt.xlim(-1, 1)
    plt.xlabel('u = sin(theta)')
    plt.ylabel('Normalized Pattern (dB)')
    plt.title(f'Pencil Beam (N={N}) with VDIL Varactor Constraints')
    plt.legend()
    plt.grid(True)
    plt.savefig('pencil_beam_result.png')
    print("Saved plot to pencil_beam_result.png")

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer.api import beamform
from beamformer.utils import oversampled_fft

def main():
    print("Running Multi-target Experiment...")
    N = 64
    targets = [-0.6, 0.0, 0.6]
    gains = [1.0, 1.0, 1.0]
    task = Task('multitarget', target_angles=targets, target_gains=gains, sll_ceiling=0.2)
    element = SyntheticVaractor(beta=0.5, folding=True)

    res = beamform(task, element, N=N, K_max=50)

    u, F = oversampled_fft(res['weights'], oversample_factor=16)
    power_dB = 20 * np.log10(np.abs(F) + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(u, power_dB - np.max(power_dB), 'b', label='Geometry-Aware AP')
    for t in targets:
        plt.axvline(t, color='green', linestyle=':', alpha=0.5)
    plt.axhline(20*np.log10(task.sll_ceiling), color='r', linestyle=':', label='SLL Ceiling')
    plt.ylim(-40, 5)
    plt.xlim(-1, 1)
    plt.xlabel('u = sin(theta)')
    plt.ylabel('Normalized Pattern (dB)')
    plt.title(f'Multi-Target Beam (N={N})')
    plt.legend()
    plt.grid(True)
    plt.savefig('multi_target_result.png')
    print("Saved plot to multi_target_result.png")

if __name__ == '__main__':
    main()

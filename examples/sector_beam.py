import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer.api import beamform
from beamformer.utils import oversampled_fft

def main():
    print("Running Sector Beam Experiment...")
    N = 64
    task = Task('sector', sector_u_min=-0.3, sector_u_max=0.3, sll_ceiling=0.1)
    element = SyntheticVaractor(beta=0.5, folding=True)

    res = beamform(task, element, N=N, K_max=50)

    u, F = oversampled_fft(res['weights'], oversample_factor=16)
    power_dB = 20 * np.log10(np.abs(F) + 1e-12)

    plt.figure(figsize=(10, 5))
    plt.plot(u, power_dB - np.max(power_dB), 'b', label='Geometry-Aware AP')
    plt.axvspan(-0.3, 0.3, color='green', alpha=0.1, label='Target Sector')
    plt.axhline(20*np.log10(task.sll_ceiling), color='r', linestyle=':', label='SLL Ceiling')
    plt.ylim(-40, 5)
    plt.xlim(-1, 1)
    plt.xlabel('u = sin(theta)')
    plt.ylabel('Normalized Pattern (dB)')
    plt.title(f'Sector Beam (N={N})')
    plt.legend()
    plt.grid(True)
    plt.savefig('sector_beam_result.png')
    print("Saved plot to sector_beam_result.png")

if __name__ == '__main__':
    main()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('..')

from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor, SyntheticVaractor
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.geometry import compute_aperture_potential, compute_array_controllability
from beamformer.api import beamform
from beamformer.utils import oversampled_fft

def get_element():
    if os.path.exists('amplitude.mat') and os.path.exists('phase.mat'):
        return MeasuredVaractor(amp_file='amplitude.mat', phase_file='phase.mat')
    return SyntheticVaractor(beta=0.8, folding=True)

def plot_smith_chart_manifold(element):
    plt.figure(figsize=(8, 8))
    plt.plot(np.real(element.c_grid), np.imag(element.c_grid), 'k-', linewidth=2, label='Measured Manifold')
    plt.scatter(np.real(element.c_grid[::20]), np.imag(element.c_grid[::20]), c=element.V_grid[::20], cmap='viridis', s=20)
    plt.colorbar(label='Control Voltage (V)')

    # Unit circle reference
    theta = np.linspace(0, 2*np.pi, 100)
    plt.plot(np.cos(theta), np.sin(theta), 'r--', alpha=0.5, label='Ideal Unit Circle')

    plt.axis('equal')
    plt.xlim(-1.2, 1.2)
    plt.ylim(-1.2, 1.2)
    plt.title('Measured Phase-Shifter Manifold (Smith Chart View)')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig1_smith_chart_manifold.png', dpi=300)
    print("Generated: fig1_smith_chart_manifold.png")

def plot_offset_heatmaps(element, N, task, initial_weights):
    n_steps = 100
    deltas = np.linspace(0, 2*np.pi, n_steps)

    e_effs = np.zeros(n_steps)
    controllability = np.zeros(n_steps)

    V_matrix = np.zeros((n_steps, N))

    for i, delta in enumerate(deltas):
        rotated = initial_weights * np.exp(1j * delta)
        c_proj = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            V_matrix[i, n], c_proj[n] = element.project(rotated[n], method='phase_only')

        e_effs[i] = compute_aperture_potential(c_proj)
        controllability[i] = compute_array_controllability(V_matrix[i, :], element)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # Voltage Matrix Heatmap
    im = axs[0].imshow(V_matrix.T, aspect='auto', cmap='magma', extent=[0, 360, N, 1])
    axs[0].set_title('Element Voltage Assignments vs $\delta$')
    axs[0].set_xlabel('Global Offset $\delta$ (Degrees)')
    axs[0].set_ylabel('Element Index')
    fig.colorbar(im, ax=axs[0], label='Voltage (V)')

    # E_eff
    axs[1].plot(np.rad2deg(deltas), e_effs, 'b', linewidth=2)
    axs[1].set_title('Aperture Potential ($E_{eff}$) vs $\delta$')
    axs[1].set_xlabel('Global Offset $\delta$ (Degrees)')
    axs[1].set_ylabel('$E_{eff}$ (Gain Capacity)')
    axs[1].grid(True)

    # Controllability
    axs[2].plot(np.rad2deg(deltas), controllability, 'g', linewidth=2)
    axs[2].set_title('Array Controllability (C) vs $\delta$')
    axs[2].set_xlabel('Global Offset $\delta$ (Degrees)')
    axs[2].set_ylabel('Average Mobility ($M_n$)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig('fig2_offset_heatmaps.png', dpi=300)
    print("Generated: fig2_offset_heatmaps.png")

def plot_benchmark_ablation(element, N, task):
    print("\nRunning Ablation Benchmark...")

    stages = [
        {"name": "Stage 0 (Coherent)", "ap": False, "offset": False, "refine": False},
        {"name": "Stage 1 (AP Feasibility)", "ap": True, "offset": False, "refine": False},
        {"name": "Stage 2 (AP + Geometry Offset)", "ap": True, "offset": True, "refine": False},
        {"name": "Stage 3 (AP + Offset + Refinement)", "ap": True, "offset": True, "refine": True}
    ]

    plt.figure(figsize=(10, 6))
    colors = ['gray', 'blue', 'orange', 'red']

    for i, stage in enumerate(stages):
        res = beamform(
            task, element, N=N, K_max=80,
            enable_ap=stage['ap'],
            enable_offset=stage['offset'],
            enable_local_refinement=stage['refine'],
            projection_method='phase_only',
            null_init_method='schelkunoff'
        )

        u, F = oversampled_fft(res['weights'], oversample_factor=16)
        power_dB = 20 * np.log10(np.abs(F) + 1e-12)

        # Normalize to theoretical ideal
        ideal_no_amp = np.exp(1j * np.angle(res['initial_weights']))
        _, F_ideal = oversampled_fft(ideal_no_amp, oversample_factor=16)
        max_ideal_db = np.max(20 * np.log10(np.abs(F_ideal) + 1e-12))

        plt.plot(u, power_dB - max_ideal_db, color=colors[i], label=stage['name'],
                 linewidth=2 if stage['refine'] else 1.5,
                 linestyle='-' if stage['ap'] else '--')

    plt.axvline(task.target_angles[0], color='k', linestyle=':', label='Target')
    for nu in task.null_angles:
        plt.axvline(nu, color='k', linestyle='-.', label='Null Constraint' if nu == task.null_angles[0] else "")

    plt.ylim(-60, 5)
    plt.xlim(-1, 1)
    plt.xlabel('u = sin(theta)')
    plt.ylabel('Normalized Pattern (dB)')
    plt.title('Ablation Study: Progressive Geometry Conditioning')
    plt.legend()
    plt.grid(True)
    plt.savefig('fig3_ablation_benchmark.png', dpi=300)
    print("Generated: fig3_ablation_benchmark.png")

def main():
    print("Generating Publication Slide Figures...")
    element = get_element()
    N = 64
    task = Task('nulled', target_angles=[0.2], null_angles=[-0.4, 0.5])
    initial_weights = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    plot_smith_chart_manifold(element)
    plot_offset_heatmaps(element, N, task, initial_weights)
    plot_benchmark_ablation(element, N, task)
    print("\nAll figures generated successfully.")

if __name__ == '__main__':
    main()

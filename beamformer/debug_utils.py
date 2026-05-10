import numpy as np
import matplotlib.pyplot as plt
import os
from .utils import oversampled_fft

class DebugLogger:
    def __init__(self, debug_dir, task, element, N):
        self.debug_dir = debug_dir
        self.task = task
        self.element = element
        self.N = N

        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)

        self.history = {
            'deltas': [],
            'residuals': [],
            'patterns': [],
            'weights': [],
            'voltages': []
        }
        self.initial_weights = None
        self.initial_voltages = None

    def log_initial_state(self, initial_weights, initial_voltages):
        self.initial_weights = initial_weights.copy()
        self.initial_voltages = initial_voltages.copy()

    def log_iteration(self, delta, residual, weights, voltages):
        self.history['deltas'].append(delta)
        self.history['residuals'].append(residual)
        self.history['weights'].append(weights.copy())
        self.history['voltages'].append(voltages.copy())

        # Calculate and log pattern
        u, F = oversampled_fft(weights, oversample_factor=16)
        self.history['patterns'].append((u, F))

    def generate_report(self):
        print(f"Generating debug visualisations in {self.debug_dir}...")
        self._plot_varactor_profile()
        self._plot_coherent_vs_ap()
        self._plot_delta_progression()
        self._plot_voltage_distributions()
        self._plot_complex_plane_trajectory()

    def _plot_varactor_profile(self):
        plt.figure(figsize=(8, 6))
        V_sweep = np.linspace(self.element.v_min, self.element.v_max, 500)
        c_sweep = self.element.get_complex_weight(V_sweep)

        plt.plot(np.angle(c_sweep), np.abs(c_sweep), 'b')
        plt.xlabel('Phase (rad)')
        plt.ylabel('Amplitude (Insertion Loss)')
        plt.title('Nonlinear Varactor Profile (Amplitude vs Phase)')
        plt.grid(True)
        plt.savefig(os.path.join(self.debug_dir, '01_varactor_profile.png'))
        plt.close()

    def _plot_coherent_vs_ap(self):
        plt.figure(figsize=(10, 5))

        u, F_initial = oversampled_fft(self.initial_weights, oversample_factor=16)
        u, F_final = self.history['patterns'][-1]

        db_initial = 20 * np.log10(np.abs(F_initial) + 1e-12)
        db_final = 20 * np.log10(np.abs(F_final) + 1e-12)

        # Real pattern without normalization to see true insertion loss
        plt.plot(u, db_initial, '--', color='gray', label='Coherent Baseline')
        plt.plot(u, db_final, 'b', label='Geometry-Aware AP')

        plt.ylim(np.max(db_initial) - 50, np.max(db_initial) + 5)
        plt.xlabel('u = sin(theta)')
        plt.ylabel('Absolute Pattern (dB)')
        plt.title('Pattern Comparison (Showing Absolute Insertion Loss)')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.debug_dir, '02_pattern_comparison.png'))
        plt.close()

    def _plot_delta_progression(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.history['deltas'], marker='o', linestyle='-')
        plt.xlabel('Iteration')
        plt.ylabel('Global Phase Offset $\\delta$ (rad)')
        plt.title('Offset Trajectory Over Time')
        plt.grid(True)
        plt.savefig(os.path.join(self.debug_dir, '03_delta_progression.png'))
        plt.close()

    def _plot_voltage_distributions(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.initial_voltages, 'o--', color='gray', label='Initial (Iteration 0)')
        plt.plot(self.history['voltages'][-1], 'x-', color='blue', label='Final')
        plt.xlabel('Element Index')
        plt.ylabel('Varactor Control Voltage')
        plt.title('Hardware Voltage Distribution')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.debug_dir, '04_voltage_distributions.png'))
        plt.close()

    def _plot_complex_plane_trajectory(self):
        plt.figure(figsize=(8, 8))
        V_sweep = np.linspace(self.element.v_min, self.element.v_max, 500)
        c_sweep = self.element.get_complex_weight(V_sweep)

        plt.plot(np.real(c_sweep), np.imag(c_sweep), color='lightgray', zorder=1, label='Manifold')

        # Plot element 0's trajectory
        elem_idx = self.N // 2
        traj = [w[elem_idx] for w in self.history['weights']]

        plt.scatter(np.real(traj[0]), np.imag(traj[0]), color='red', marker='x', s=100, zorder=3, label='Iter 0')
        plt.scatter(np.real(traj[-1]), np.imag(traj[-1]), color='blue', marker='o', s=100, zorder=4, label='Final')
        plt.plot(np.real(traj), np.imag(traj), 'k:', zorder=2)

        plt.axis('equal')
        plt.xlim(-1.2, 1.2)
        plt.ylim(-1.2, 1.2)
        plt.title(f'Complex Plane Trajectory for Element {elem_idx}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(self.debug_dir, '05_complex_trajectory.png'))
        plt.close()

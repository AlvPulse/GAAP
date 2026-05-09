import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cells
text1 = """# Geometry-Aware Phase-Only Beamforming Sandbox
This notebook provides interactive visualizations for analyzing alternating projection (AP)
convergence dynamics under real, measured hardware constraints (Voltage-Dependent Insertion Loss and non-monotonic phase).

**Core Concept**: The global phase offset $\\delta$ is not naively optimized, but dynamically evolved as a control signal to reshape convergence geometry."""

code1 = """import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

# Import our beamformer codebase
import sys
sys.path.append('..')

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.api import beamform
from beamformer.geometry import scan_offset_feasibility
from beamformer.utils import oversampled_fft

# Ensure plots look good inline
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
"""

text2 = """## 1. Diagnostic Geometry Scan (Feasibility Landscape)
Before running the alternating projection, we compute the diagnostic offset sweeps across $\\delta \\in [0, 2\\pi]$ to identify stable convergence basins."""

code2 = """# Initialize the array
N = 16
task = Task('pencil', target_angles=[0.5], sll_ceiling=0.1)

# Interactive sweeping tool
def plot_feasibility_landscape(beta=1.0, folding=True):
    element = SyntheticVaractor(beta=beta, folding=folding)

    # Use the initial coherent weights for the diagnostic sweep
    from beamformer.synthesis import synthesize_pencil
    w_initial = synthesize_pencil(N, 0.5)

    metrics = scan_offset_feasibility(w_initial, element, n_steps=200)

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(metrics['deltas'], metrics['residuals'], color='red')
    axes[0].set_ylabel('Projection Residual')
    axes[0].set_title('Feasibility Landscape')

    axes[1].plot(metrics['deltas'], metrics['average_amplitudes'], color='green')
    axes[1].set_ylabel('Avg. Amplitude (Gain)')

    # Plot branch selection heatmap
    im = axes[2].imshow(metrics['selected_voltages'].T, aspect='auto',
                        extent=[0, 2*np.pi, 0, N], cmap='viridis', origin='lower')
    axes[2].set_xlabel(r'Global Phase Offset $\\delta$ [rad]')
    axes[2].set_ylabel('Element Index')
    plt.colorbar(im, ax=axes[2], label='Selected Voltage (Branch)')

    plt.tight_layout()
    plt.show()

widgets.interact(plot_feasibility_landscape,
                 beta=widgets.FloatSlider(value=0.8, min=0.0, max=1.5, step=0.1, description='IL Severity'),
                 folding=widgets.Checkbox(value=True, description='Branch Folding'));
"""

text3 = """## 2. Geometry-Aware Alternating Projection Sandbox
Now we execute the AP loop. Watch how the geometry-aware offset controller responds to the initial conditions and steers the algorithm into stable regions."""

code3 = """def run_ap_sandbox(target_angle=0.5, beta=0.8, initial_delta=0.0):
    element = SyntheticVaractor(beta=beta, folding=True)
    task = Task('pencil', target_angles=[target_angle], sll_ceiling=0.15)

    res = beamform(task, element, N=32, initial_delta=initial_delta, K_max=40)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Array Pattern
    u, F = oversampled_fft(res['weights'], oversample_factor=16)
    pattern_dB = 20 * np.log10(np.abs(F) + 1e-12)

    u_ideal, F_ideal = oversampled_fft(res['initial_weights'], oversample_factor=16)
    ideal_dB = 20 * np.log10(np.abs(F_ideal) + 1e-12)

    axes[0].plot(u_ideal, ideal_dB - np.max(ideal_dB), '--', color='gray', label='Ideal Coherent')
    axes[0].plot(u, pattern_dB - np.max(pattern_dB), color='blue', label='Achieved (AP)')
    axes[0].axvline(target_angle, color='red', linestyle=':', label='Target')
    axes[0].set_ylim(-40, 5)
    axes[0].set_xlabel('u (sin theta)')
    axes[0].set_ylabel('Normalized Pattern (dB)')
    axes[0].set_title('Array Factor')
    axes[0].legend()

    # Plot 2: Convergence Dynamics
    hist = res['history']
    ax2 = axes[1]
    color = 'tab:red'
    ax2.set_xlabel('Iteration k')
    ax2.set_ylabel('Projection Residual', color=color)
    ax2.plot(hist['residuals'], color=color, marker='o')
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax2.twinx()
    color = 'tab:blue'
    ax3.set_ylabel('Phase Offset $\\delta_k$', color=color)
    ax3.plot(hist['deltas'], color=color, linestyle='--', marker='x')
    ax3.tick_params(axis='y', labelcolor=color)
    axes[1].set_title('Controller Dynamics')

    # Plot 3: Manifold Projection
    axes[2].scatter(np.real(res['weights']), np.imag(res['weights']), color='purple', s=40, zorder=3, label='Final Weights')

    # Plot full manifold trajectory for reference
    V_sweep = np.linspace(0, 15, 500)
    c_sweep = element.get_complex_weight(V_sweep)
    axes[2].plot(np.real(c_sweep), np.imag(c_sweep), color='lightgray', zorder=1)

    axes[2].axis('equal')
    axes[2].set_xlim(-1.2, 1.2)
    axes[2].set_ylim(-1.2, 1.2)
    axes[2].set_title('Complex Plane Projection')
    axes[2].legend()

    plt.tight_layout()
    plt.show()

widgets.interact(run_ap_sandbox,
                 target_angle=widgets.FloatSlider(value=0.5, min=-0.9, max=0.9, step=0.05),
                 beta=widgets.FloatSlider(value=0.8, min=0.0, max=1.5, step=0.1),
                 initial_delta=widgets.FloatSlider(value=0.0, min=0.0, max=2*np.pi, step=0.1));
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text1),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_markdown_cell(text2),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_markdown_cell(text3),
    nbf.v4.new_code_cell(code3)
]

with open('notebooks/sandbox.ipynb', 'w') as f:
    nbf.write(nb, f)

"""
Investigates the behavior of the PTNR landscape over different global phase offsets (psi).

This script generates a random nulled beamforming task and visualizes the
Peak-to-Null Ratio (PTNR) before and after applying Gain-Locked Coordinate Polish (GLCP)
across a grid of global phase offsets. This demonstrates why the "PTNR gauge" is
superior to the "Gain gauge" for offset selection.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

# Add parent directory to path to import beamformer modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.coherent_lcmv import _af, gain_locked_polish, solve_coherent_lcmv
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.pattern_projection import Task

def set_ieee_style():
    """Applies global Matplotlib settings tailored for IEEE double-column papers."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.titlesize": 9,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "path.simplify": True,
        "figure.dpi": 300,
    })


def main():
    """Run the investigation and generate the plot."""
    np.random.seed(42)
    N = 64
    #element = SyntheticVaractor()
    element = MeasuredVaractor()

    # Generate a random task
    target_u = np.random.uniform(-0.5, 0.5)
    null_u = np.random.uniform(-0.8, 0.8)
    while abs(null_u - target_u) < 0.15:
        null_u = np.random.uniform(-0.8, 0.8)

    task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])

    psi_grid = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    pre_gains = []
    pre_nulls = []
    pre_ptnrs = []

    post_gains = []
    post_nulls = []
    post_ptnrs = []

    for psi in psi_grid:
        # 1. Anchor
        V_n, c_n, info = solve_coherent_lcmv(
            task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
        )

        pre_gain = 20 * np.log10(abs(_af(c_n, target_u, N)) / N + 1e-12)
        pre_null = 20 * np.log10(abs(_af(c_n, null_u, N)) + 1e-12)
        pre_gains.append(pre_gain)
        pre_nulls.append(pre_null)
        pre_ptnrs.append(pre_gain - pre_null)

        # 2. GLCP
        V_post, c_post = gain_locked_polish(
            c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995
        )

        post_gain = 20 * np.log10(abs(_af(c_post, target_u, N)) / N + 1e-12)
        post_null = 20 * np.log10(abs(_af(c_post, null_u, N)) + 1e-12)
        post_gains.append(post_gain)
        post_nulls.append(post_null)
        post_ptnrs.append(post_gain - post_null)

    set_ieee_style()
    plt.figure(figsize=(10, 6))
    plt.plot(psi_grid, pre_ptnrs, label="Pre-GLCP PTNR (Anchor)")
    plt.plot(psi_grid, post_ptnrs, label="Post-GLCP PTNR (Anchor + GLCP)")
    plt.xlabel("Gauge Psi")
    plt.ylabel("PTNR (dB)")
    plt.title("PTNR Landscape vs Gauge Psi")
    plt.legend()
    plt.grid(True)
    plt.savefig("experiments/ptnr_landscape.png")

    corr, _ = pearsonr(pre_ptnrs, post_ptnrs)
    print(f"Correlation between pre-GLCP and post-GLCP PTNR: {corr:.3f}")
    print(f"Max Pre-GLCP PTNR: {np.max(pre_ptnrs):.2f} dB (at psi={psi_grid[np.argmax(pre_ptnrs)]:.2f})")
    print(f"Post-GLCP PTNR at that same psi: {post_ptnrs[np.argmax(pre_ptnrs)]:.2f} dB")
    print(f"Actual Max Post-GLCP PTNR: {np.max(post_ptnrs):.2f} dB (at psi={psi_grid[np.argmax(post_ptnrs)]:.2f})")
    print(f"Difference (Opportunity Loss): {np.max(post_ptnrs) - post_ptnrs[np.argmax(pre_ptnrs)]:.2f} dB")

if __name__ == "__main__":
    main()

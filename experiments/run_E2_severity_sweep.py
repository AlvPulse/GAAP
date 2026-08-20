import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants

os.makedirs("experiments/results", exist_ok=True)
os.makedirs("experiments/figures_measured", exist_ok=True)

N = 64
task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])

# Create a severely interpolated locus wrapper to simulate severity (Alpha)
class AlphaVaractor:
    def __init__(self, alpha):
        meas = MeasuredVaractor(folding=True)
        self.V_grid = meas.V_grid
        self.v_min, self.v_max = meas.v_min, meas.v_max

        # Ideal circle
        c_ideal = np.exp(1j * self.V_grid)
        # Interpolation: alpha=0 -> Ideal, alpha=1 -> Measured
        self.c_grid = (1 - alpha) * c_ideal + alpha * np.asarray(meas.c_grid)

alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

print("Running E2 Severity Sweep (Alpha)...\n")
all_data = []

# Multiple trials using random initial steering shifts (pseudo-random tasks)
for trial in range(5):
    # Create a slightly shifted task to gather scatter data
    t_angle = 0.2 + np.random.uniform(-0.05, 0.05)
    n_angles = [-0.4 + np.random.uniform(-0.05, 0.05), 0.5 + np.random.uniform(-0.05, 0.05)]
    rand_task = Task(f"rand_{trial}", target_angles=[t_angle], null_angles=n_angles)

    for alpha in alphas:
        el = AlphaVaractor(alpha)
        # Adaptive MRLCMV
        res = beamform_mrlcmv_variants(rand_task, el, N, mode="adaptive", max_dual_steps=32, gauge_samples=12)
        n_db = res['info']['final_worst_null_db']

        all_data.append({
            "Trial": trial,
            "Alpha": alpha,
            "Null": n_db,
            "Evals": res['info']['evals']
        })
        print(f"  Alpha {alpha:.2f} | Trial {trial} | Null: {n_db:.2f} dB")

df = pd.DataFrame(all_data)
df.to_csv("experiments/results/E2_severity_sweep.csv", index=False)

# E2 IEEE Scatter Figure
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'legend.fontsize': 7, 'font.family': 'serif', 'pdf.fonttype': 42
})

plt.figure(figsize=(3.5, 2.8))
for alpha in alphas:
    d = df[df['Alpha'] == alpha]
    # Scatter the per-task data
    plt.scatter([alpha]*len(d), d['Null'], alpha=0.6, edgecolors='k')
    # Plot median as a red line/marker
    median = d['Null'].median()
    plt.plot(alpha, median, 'rs', markersize=6)

plt.xlabel(r"Hardware Severity ($\alpha$)")
plt.ylabel("Worst Null Depth (dB)")
plt.grid(True, linestyle='--', alpha=0.5)

# Custom legend
plt.scatter([], [], color='blue', alpha=0.6, edgecolors='k', label='Per-Task Instance')
plt.plot([], [], 'rs', label='Median')
plt.legend(loc='upper right', framealpha=0.9, edgecolor='0.8')
plt.tight_layout(pad=0.5)

plt.savefig("experiments/figures_measured/fig_E2_severity_sweep_IEEE.pdf", format='pdf', bbox_inches='tight')
print("\nSeverity sweep complete. Data saved to CSV and Figure exported.")

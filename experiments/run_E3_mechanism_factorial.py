import sys
import os
import numpy as np
import pandas as pd

sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants

os.makedirs("experiments/results", exist_ok=True)

N = 64
task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])

# E3 Mechanism Factorial Setup
meas = MeasuredVaractor(folding=True)
phases = meas.V_grid
amp_meas = np.abs(meas.c_grid)

# We create 4 variants of the hardware to isolate amplitude vs phase limits
class FactorialVaractor:
    def __init__(self, amp_type, phase_type):
        self.v_min, self.v_max = meas.v_min, meas.v_max
        self.V_grid = meas.V_grid

        # Amplitude Logic
        if amp_type == "Flat":
            a = np.ones_like(phases)
        else: # "VDIL"
            a = amp_meas

        # Phase Logic
        if phase_type == "Uniform":
            # Just use standard 360 degree uniform phase over the states
            p = np.linspace(0, 2*np.pi, len(phases))
        else: # "Measured/Folded"
            p = np.angle(meas.c_grid)

        self.c_grid = a * np.exp(1j * p)

    def get_complex_weight(self, V):
        V = np.clip(V, self.v_min, self.v_max)
        idx = np.searchsorted(self.V_grid, V)
        if idx >= len(self.V_grid): idx = len(self.V_grid) - 1
        return self.c_grid[idx]

print("Running E3 Mechanism Factorial...\n")

configs = [
    ("Flat", "Uniform"),
    ("Flat", "Measured/Folded"),
    ("VDIL", "Uniform"),
    ("VDIL", "Measured/Folded")
]

results = []
for amp, phase in configs:
    el = FactorialVaractor(amp, phase)
    res = beamform_mrlcmv_variants(task, el, N, mode="adaptive", max_dual_steps=32, gauge_samples=12)
    n_db = res['info']['final_worst_null_db']
    g_db = res['info']['final_gain_db']
    results.append({
        "Amplitude": amp,
        "Phase": phase,
        "Gain (dB)": g_db,
        "Null (dB)": n_db,
        "Evals": res['info']['evals']
    })
    print(f"  Amp: {amp:5s} | Phase: {phase:16s} | Null: {n_db:.2f} dB")

df = pd.DataFrame(results)
print("\n=== E3: Mechanism Factorial Results ===")
print(df.to_string(index=False, float_format="%.2f"))

tex_file = "experiments/results/E3_mechanism_factorial.tex"
with open(tex_file, "w") as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{E3: Mechanism Factorial (Amplitude vs Phase Limits)}" + "\n")
    f.write(r"\begin{tabular}{llcc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Amplitude & Phase & Gain (dB) & Null (dB) \\" + "\n")
    f.write(r"\midrule" + "\n")
    for _, r in df.iterrows():
        f.write(f"{r['Amplitude']} & {r['Phase']} & {r['Gain (dB)']:.2f} & {r['Null (dB)']:.1f} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")

print(f"\nSaved to {tex_file}")

import sys
import os
import numpy as np
import pandas as pd
import time

sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants

os.makedirs("experiments/results", exist_ok=True)

N = 64
task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)

print("Running E7 Psi Resolution Saturation...")

candidates = [1, 4, 8, 16, 32, 64]
results = []

for c in candidates:
    t0 = time.time()
    res = beamform_mrlcmv_variants(task, el, N, mode="adaptive", max_dual_steps=32, gauge_samples=c)
    ms = (time.time() - t0) * 1000
    n_db = res['info']['final_worst_null_db']

    results.append({
        "Candidates": c,
        "Null (dB)": n_db,
        "Search Time (ms)": ms,
        "Opportunity Loss (dB)": n_db - (-70.0) # approx absolute floor
    })
    print(f"  Candidates {c:2d} | Null: {n_db:.2f} dB | {ms:.1f} ms")

df = pd.DataFrame(results)
print("\n=== E7: Psi Resolution Saturation Results ===")
print(df.to_string(index=False, float_format="%.2f"))

tex_file = "experiments/results/E7_psi_resolution.tex"
with open(tex_file, "w") as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{E7: Gauge Resolution vs. Recovery}" + "\n")
    f.write(r"\begin{tabular}{lccc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Candidates & Null (dB) & Search Time (ms) & Opp. Loss (dB) \\" + "\n")
    f.write(r"\midrule" + "\n")
    for _, r in df.iterrows():
        f.write(f"{r['Candidates']} & {r['Null (dB)']:.1f} & {r['Search Time (ms)']:.1f} & {r['Opportunity Loss (dB)']:.1f} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")
print(f"\nSaved to {tex_file}")

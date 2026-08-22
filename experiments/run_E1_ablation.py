import sys
import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Any

sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants
from beamformer.baselines import Problem
from beamformer.MR_LCMV_certified import _af

os.makedirs("experiments/results", exist_ok=True)

# E1 Protocol Setup
N = 64
task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
element = MeasuredVaractor(folding=True)
u_t = task.target_angles[0]
nulls = list(task.null_angles)

def base_kwargs(**overrides):
    kw = dict(
        mode="adaptive", max_dual_steps=32,
        gauge_samples=24, glcp_gain_floor=0.995,
        stagnation_tol_db=0.1, patience=5,
        null_target_db=-60.0
    )
    kw.update(overrides)
    return kw

configs = {
    "A0_full": {"desc": "Full System (MR-LCMV+GLCP+PTNR Gauge)", "kw": base_kwargs()},
    "A1_no_gauge": {"desc": "No Gauge (psi=0)", "kw": base_kwargs(gauge_samples=1)},
    "A2_no_glcp": {"desc": "No GLCP (Dual correction only)", "kw": base_kwargs(glcp_gain_floor=1.0)}, # We'll handle no glcp via rounds=0 in the variant soon
    "A3_no_gain_lock": {"desc": "No Gain Lock (GLCP accepts pure null drops)", "kw": base_kwargs(glcp_gain_floor=0.0)},
    "A4_no_lcmv": {"desc": "No LCMV Dual (Retraction + GLCP only)", "kw": base_kwargs(max_dual_steps=0)}
}

print("Running E1 Canonical Ablation...\n")
results = []
for cid, cfg in configs.items():
    print(f"  Testing {cid}...")
    t0 = time.time()

    # We will just run the new variant wrapper
    kw = cfg["kw"]

    # Hack for no GLCP: we set the gain floor to impossible (2.0) or we just bypass it.
    if cid == "A2_no_glcp":
        kw["glcp_gain_floor"] = 2.0  # Prevents any GLCP moves

    res = beamform_mrlcmv_variants(task, element, N, **kw)

    time_ms = (time.time() - t0) * 1000
    g_db = res['info']['final_gain_db']
    n_db = res['info']['final_worst_null_db']
    oracle = res['info']['evals']
    glcp = res['info']['glcp_updates']

    results.append({
        "Config": cid,
        "Description": cfg["desc"],
        "Gain(dB)": g_db,
        "Null(dB)": n_db,
        "N_AF": oracle,
        "N_Cand": glcp,
        "Time(ms)": time_ms
    })

df = pd.DataFrame(results)
print("\n=== E1: Canonical Ablation Results ===")
print(df.to_string(index=False, float_format="%.2f"))

# Generate LaTeX Table
tex_file = "experiments/results/E1_ablation_table.tex"
with open(tex_file, "w") as f:
    f.write(r"\begin{table}[h]" + "\n")
    f.write(r"\centering" + "\n")
    f.write(r"\caption{E1: MR-LCMV Component Ablation (N=64, Measured Varactor)}" + "\n")
    f.write(r"\begin{tabular}{llcccc}" + "\n")
    f.write(r"\toprule" + "\n")
    f.write(r"Config & Description & Gain (dB) & Null (dB) & $N_{AF}$ & $N_{cand}$ \\" + "\n")
    f.write(r"\midrule" + "\n")
    for _, r in df.iterrows():
        f.write(f"{r['Config'].replace('_', '\\_')} & {r['Description']} & {r['Gain(dB)']:.2f} & {r['Null(dB)']:.1f} & {r['N_AF']:.0f} & {r['N_Cand']:.0f} \\\\\n")
    f.write(r"\bottomrule" + "\n")
    f.write(r"\end{tabular}" + "\n")
    f.write(r"\end{table}" + "\n")

print(f"\nLaTeX table saved to {tex_file}")

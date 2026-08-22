import sys
sys.path.append('.')
import pandas as pd
import time
import os

from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants

os.makedirs("experiments/results", exist_ok=True)
task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)

N_VALUES = [64, 128, 256, 512, 1024]
K_FIXED = [8, 32]

all_data = []

for N in N_VALUES:
    print(f"Running N={N}...")

    # Long Run Ref
    ref = beamform_mrlcmv_variants(task, el, N, mode="certified", max_dual_steps=256, gauge_samples=8)
    ref_null = ref['info']['final_worst_null_db']

    all_data.append({
        "N": N, "Mode": "Long-run Ref", "K": 256, "Null": ref_null,
        "Gap": 0.0, "Evals": ref['info']['evals'], "GLCP": ref['info']['glcp_updates'], "Time": ref['info']['runtime_ms']
    })

    # Fixed loops
    for k in K_FIXED:
        fixed = beamform_mrlcmv_variants(task, el, N, mode="fixed", max_dual_steps=k, gauge_samples=8)
        all_data.append({
            "N": N, "Mode": f"Fixed-{k}", "K": k, "Null": fixed['info']['final_worst_null_db'],
            "Gap": ref_null - fixed['info']['final_worst_null_db'],
            "Evals": fixed['info']['evals'], "GLCP": fixed['info']['glcp_updates'], "Time": fixed['info']['runtime_ms']
        })

    # Certified
    cert = beamform_mrlcmv_variants(task, el, N, mode="certified", max_dual_steps=128, gauge_samples=8)
    all_data.append({
        "N": N, "Mode": "Certified", "K": cert['info']['convergence']['iterations_used'], "Null": cert['info']['final_worst_null_db'],
        "Gap": ref_null - cert['info']['final_worst_null_db'],
        "Evals": cert['info']['evals'], "GLCP": cert['info']['glcp_updates'], "Time": cert['info']['runtime_ms']
    })

    # Adaptive
    adapt = beamform_mrlcmv_variants(task, el, N, mode="adaptive", max_dual_steps=128, gauge_samples=8)
    all_data.append({
        "N": N, "Mode": "Adaptive", "K": adapt['info']['convergence']['iterations_used'], "Null": adapt['info']['final_worst_null_db'],
        "Gap": ref_null - adapt['info']['final_worst_null_db'],
        "Evals": adapt['info']['evals'], "GLCP": adapt['info']['glcp_updates'], "Time": adapt['info']['runtime_ms']
    })

df = pd.DataFrame(all_data)
df.to_csv("experiments/data/convergence_scaling.csv", index=False)
print("Real data saved.")

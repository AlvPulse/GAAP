import sys
sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants
import time

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)

for N in [64, 256]:
    print(f"\n--- N={N} ---")
    t0 = time.time()
    # Certified (Long run ref)
    ref = beamform_mrlcmv_variants(task, el, N, mode="certified", max_dual_steps=256, gauge_samples=8)
    print(f"Ref K=256   : Null={ref['info']['final_worst_null_db']:.2f}, Evals={ref['info']['evals']}, GLCP={ref['info']['glcp_updates']}")

    # Adaptive
    adapt = beamform_mrlcmv_variants(task, el, N, mode="adaptive", max_dual_steps=128, gauge_samples=8)
    gap = ref['info']['final_worst_null_db'] - adapt['info']['final_worst_null_db']
    print(f"Adaptive    : Null={adapt['info']['final_worst_null_db']:.2f}, Evals={adapt['info']['evals']}, GLCP={adapt['info']['glcp_updates']}, Gap to ref: {gap:.2f}dB")

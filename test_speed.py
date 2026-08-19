from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants
import time

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)

t0 = time.time()
res = beamform_mrlcmv_variants(task, el, 64, mode='adaptive', max_dual_steps=32, gauge_samples=4)
print(f"Time: {time.time() - t0:.2f}s, Evals: {res['info'].get('evals')}, GLCP: {res['info'].get('glcp_updates')}")

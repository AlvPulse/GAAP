import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.api import beamform

N = 64
element = SyntheticVaractor(beta=1.5, folding=True)
task = Task(type='nulled', target_angles=[0.4], null_angles=[-0.3, 0.2, 0.6], sll_ceiling=0.1)
K_max = 50

res_event = beamform(
    task, element, N,
    K_max=K_max,
    enable_ap=True,
    enable_offset=True,
    enable_local_refinement=True,
    enable_event_triggered_offset=True,
    basin_candidates=8,
    probe_lr_steps=3,
    k_active=16,
    refinement_steps=5,
    step_size=0.1,
    patience=2,  # Faster triggering for test
    min_basin_iters=2
)

print("Final Delta:", res_event['history']['deltas'][-1])
print("Final Residual:", res_event['history']['residuals'][-1])
print("Final Cost:", res_event['history']['costs'][-1])
print("Number of Deltas tracked:", len(set(res_event['history']['deltas'])))

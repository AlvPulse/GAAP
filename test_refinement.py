import numpy as np
from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.local_refinement import refine_local_active_set
from beamformer.incremental_af import IncrementalAFCache

N = 16
element = SyntheticVaractor()
task = Task(type='nulled', target_angles=[0.0], null_angles=[0.5])
# Dummy weights
V_cand = np.linspace(0, 10, N)

af_cache = IncrementalAFCache(N, task)
initial_weights = np.array([element.get_complex_weight(v) for v in V_cand])
af_cache.initialize(initial_weights)

V_ref = refine_local_active_set(V_cand, task, element, af_cache, k_active=4, refinement_steps=2, step_size=0.1)

print("Refinement completed successfully without syntax errors.")

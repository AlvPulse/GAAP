import sys
sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants
import numpy as np
import time

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)
N = 64

# Test Gauge Symmetry-Breaking
print("--- Test 3: Gauge Symmetry-Breaking ---")
# We will do a manual sweep
res = beamform_mrlcmv_variants(task, el, N, mode="adaptive", max_dual_steps=32, gauge_samples=72)
print("Sweep done. Check PTNR vs Gain variance in gauge search...")

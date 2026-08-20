import sys
sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants
import numpy as np

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)
N = 64

# Run with normal floor
res_floor = beamform_mrlcmv_variants(task, el, N, mode="adaptive", max_dual_steps=128, glcp_gain_floor=0.995, gauge_samples=72)
# Run without floor
from beamformer.baselines import solve_cd, Problem
prob = Problem(task, el, N, discrete=True)
rng = np.random.default_rng(0)
prob.rho = 0.0 # Remove gain penalty from objective entirely
V_cd = solve_cd(prob, rng, sweeps=10, grid=360)
c_cd = prob.weights(V_cd)
from beamformer.MR_LCMV_certified import _af
res_nofloor = {"info": {"final_gain_db": 20*np.log10(abs(_af(c_cd, 0.2, N))/N), "final_worst_null_db": 20*np.log10(max([abs(_af(c_cd, nu, N)) for nu in [-0.4, 0.5]]))}}


print(f"Floor ON : Gain={res_floor['info']['final_gain_db']:.2f} dB, Null={res_floor['info']['final_worst_null_db']:.2f} dB")
print(f"Floor OFF: Gain={res_nofloor['info']['final_gain_db']:.2f} dB, Null={res_nofloor['info']['final_worst_null_db']:.2f} dB")

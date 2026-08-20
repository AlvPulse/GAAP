import sys
sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.baselines import solve_admm, solve_ours_variant, Problem
import numpy as np

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])

# Ideal circle vs Discrete measured grid
class IdealCircleElement:
    def __init__(self):
        self.v_min, self.v_max = -np.pi, np.pi
        phases = np.linspace(-np.pi, np.pi, 3600)
        self.V_grid = phases
        self.c_grid = np.exp(1j * phases)
    def get_complex_weight(self, V):
        return np.exp(1j * V)

el_ideal = IdealCircleElement()
el_meas = MeasuredVaractor(folding=True)
rng = np.random.default_rng(0)

def score(c):
    u_t = task.target_angles[0]
    g = 20 * np.log10(abs(np.sum(c * np.exp(1j*np.pi*np.arange(64)*u_t)))/64 + 1e-12)
    n1 = 20 * np.log10(abs(np.sum(c * np.exp(1j*np.pi*np.arange(64)*-0.4))) + 1e-12)
    n2 = 20 * np.log10(abs(np.sum(c * np.exp(1j*np.pi*np.arange(64)*0.5))) + 1e-12)
    wn = max(n1, n2)
    return g, wn

prob_ideal = Problem(task, el_ideal, 64, discrete=False)
prob_meas = Problem(task, el_meas, 64, discrete=True)

# 1. ADMM on Ideal
V_admm_ideal = solve_admm(prob_ideal, rng, max_iter=200, gain_w=0.01)
c_admm_ideal = prob_ideal.weights(V_admm_ideal)
g, wn = score(c_admm_ideal)
print(f"ADMM on Ideal Circle: Gain={g:.2f}, Null={wn:.2f}")

# 2. ADMM on Measured Discrete
V_admm_meas = solve_admm(prob_meas, rng, max_iter=200, gain_w=0.01)
c_admm_meas = prob_meas.weights(V_admm_meas)
g, wn = score(c_admm_meas)
print(f"ADMM on Measured Locus: Gain={g:.2f}, Null={wn:.2f}")

# 3. MR-LCMV on Measured Discrete
V_ours_meas = solve_ours_variant(prob_meas, rng, mode="adaptive", max_dual_steps=128)
c_ours_meas = prob_meas.weights(V_ours_meas)
g, wn = score(c_ours_meas)
print(f"MR-LCMV on Measured Locus: Gain={g:.2f}, Null={wn:.2f}")

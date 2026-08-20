import sys
sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.baselines import solve_pgd, solve_rcg, Problem
import numpy as np

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
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

prob_ideal = Problem(task, el_ideal, 64, discrete=False)
prob_meas = Problem(task, el_meas, 64, discrete=True)

def score(c):
    u_t = task.target_angles[0]
    g = 20 * np.log10(abs(np.sum(c * np.exp(1j*np.pi*np.arange(64)*u_t)))/64 + 1e-12)
    n1 = 20 * np.log10(abs(np.sum(c * np.exp(1j*np.pi*np.arange(64)*-0.4))) + 1e-12)
    n2 = 20 * np.log10(abs(np.sum(c * np.exp(1j*np.pi*np.arange(64)*0.5))) + 1e-12)
    return g, max(n1, n2)

# PGD
V_pgd = solve_pgd(prob_ideal, rng, max_iter=400)
g, wn = score(prob_ideal.weights(V_pgd))
print(f"PGD on Ideal: Gain={g:.2f}, Null={wn:.2f}")

V_pgd_m = solve_pgd(prob_meas, rng, max_iter=400)
g, wn = score(prob_meas.weights(V_pgd_m))
print(f"PGD on Measured: Gain={g:.2f}, Null={wn:.2f}")

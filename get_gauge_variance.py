import sys
sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.MR_LCMV_certified import _af, reset_counters
from beamformer.coherent_lcmv import _retract
from beamformer.mrlcmv_variants import build_A_matrix
from beamformer.mrlcmv_variants import gain_locked_polish_minimax
import numpy as np

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)
N = 64

nulls = list(task.null_angles)
A, Ginv = build_A_matrix(nulls, N)
c_grid, V_grid = np.asarray(el.c_grid), np.asarray(el.V_grid)
n_idx = np.arange(N)
u_t = task.target_angles[0]
s_t = np.exp(-1j * np.pi * n_idx * u_t)

gains = []
null_depths = []

for psi in np.linspace(0, 2*np.pi, 72, endpoint=False):
    tgt0 = np.exp(1j * psi) * s_t
    c_n, V_n, _ = _retract(tgt0, c_grid, V_grid)

    # Simulate a single step or just the initial projection + GLCP
    V_glcp, c_glcp = gain_locked_polish_minimax(c_n, V_n, task, el, N, nulls, initial_gain=abs(_af(c_n, u_t, N))/N)

    g = 20 * np.log10(abs(_af(c_glcp, u_t, N)) / N + 1e-12)
    wn = 20 * np.log10(max([abs(_af(c_glcp, nu, N)) for nu in nulls]) + 1e-12)

    gains.append(g)
    null_depths.append(wn)

print(f"Gain variation: {np.max(gains) - np.min(gains):.2f} dB")
print(f"Null depth variation: {np.max(null_depths) - np.min(null_depths):.2f} dB")
print(f"Best Gain: {np.max(gains):.2f} dB, Worst Gain: {np.min(gains):.2f} dB")
print(f"Best Null: {np.min(null_depths):.2f} dB, Worst Null: {np.max(null_depths):.2f} dB")

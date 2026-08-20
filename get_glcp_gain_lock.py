import sys
sys.path.append('.')
from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.MR_LCMV_certified import _retract, _af
from beamformer.mrlcmv_variants import gain_locked_polish_minimax
import numpy as np

task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)
N = 64

c_grid, V_grid = np.asarray(el.c_grid), np.asarray(el.V_grid)
tgt0 = np.exp(-1j * np.pi * np.arange(N) * 0.2) * np.exp(1j * 0)
c_n, V_n, _ = _retract(tgt0, c_grid, V_grid)

# Before GLCP
g_before = 20 * np.log10(abs(_af(c_n, 0.2, N))/N)
wn_before = 20 * np.log10(max([abs(_af(c_n, nu, N)) for nu in [-0.4, 0.5]]))

# With 99.5% gain floor
V_glcp, c_glcp = gain_locked_polish_minimax(c_n, V_n, task, el, N, [-0.4, 0.5], initial_gain=abs(_af(c_n, 0.2, N))/N, gain_floor_ratio=0.995)
g_after = 20 * np.log10(abs(_af(c_glcp, 0.2, N))/N)
wn_after = 20 * np.log10(max([abs(_af(c_glcp, nu, N)) for nu in [-0.4, 0.5]]))

# With 0.0% gain floor (unlocked)
V_noguard, c_noguard = gain_locked_polish_minimax(c_n, V_n, task, el, N, [-0.4, 0.5], initial_gain=abs(_af(c_n, 0.2, N))/N, gain_floor_ratio=0.000)
g_noguard = 20 * np.log10(abs(_af(c_noguard, 0.2, N))/N)
wn_noguard = 20 * np.log10(max([abs(_af(c_noguard, nu, N)) for nu in [-0.4, 0.5]]))

print(f"Before GLCP    : Gain={g_before:.2f} dB, Null={wn_before:.2f} dB")
print(f"GLCP (Floor On): Gain={g_after:.2f} dB, Null={wn_after:.2f} dB")
print(f"GLCP (Floor Off): Gain={g_noguard:.2f} dB, Null={wn_noguard:.2f} dB")

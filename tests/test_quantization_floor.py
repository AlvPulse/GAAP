"""Regression tests for the quantization null-floor theory (docs/THEORY.md).

Prop. 2: a grid-native solver's achievable null depth recedes ~20 dB/decade with
manifold resolution. Prop. 1: a continuous solver on the analytic model reports a
resolution-independent "phantom" null far deeper than anything realizable on the
device, while a design-then-project solver does not track the floor.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer import baselines as B

N = 32
TASK = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])


def _ours_null(R):
    el = SyntheticVaractor(beta=0.8, folding=True, n_points=R)
    return B.run_solver("MR-LCMV+GLCP (ours)", TASK, el, N, seed=0, discrete=True)["worst_null"]


def test_floor_recedes_with_resolution():
    """Finer hardware grid -> deeper achievable null (Prop. 2), monotone across decades."""
    coarse = _ours_null(32)      # ~5 bits
    fine = _ours_null(2048)      # ~11 bits
    assert fine <= coarse - 20.0, (
        f"null floor did not recede with resolution: {coarse:.1f} -> {fine:.1f} dB")


def test_floor_slope_near_minus_20_db_per_decade():
    """The achievable-floor slope should be close to the predicted -20 dB/decade."""
    Rs = [32, 128, 512, 2048]
    y = np.array([_ours_null(R) for R in Rs], float)
    slope = np.polyfit(np.log10(Rs), y, 1)[0]
    assert -30.0 <= slope <= -10.0, f"floor slope {slope:.1f} dB/decade off prediction (-20)"


def test_phantom_is_unreachable_and_resolution_independent():
    """Analytic (smooth) Trust-Region reports a deep null that the device cannot
    realize, and that depth barely moves with resolution (Prop. 1)."""
    el_lo = SyntheticVaractor(beta=0.8, folding=True, n_points=32)
    el_hi = SyntheticVaractor(beta=0.8, folding=True, n_points=2048)
    ph_lo = B.run_solver("Trust-Region/LM", TASK, el_lo, N, seed=0, discrete=False,
                         max_nfev=300)["worst_null"]
    ph_hi = B.run_solver("Trust-Region/LM", TASK, el_hi, N, seed=0, discrete=False,
                         max_nfev=300)["worst_null"]
    real = B.run_solver("Trust-Region/LM", TASK, el_lo, N, seed=0, discrete=True,
                        max_nfev=300)["worst_null"]
    assert ph_lo < -90.0, f"analytic phantom not deep ({ph_lo:.1f} dB)"
    assert real > ph_lo + 60.0, "phantom did not collapse on the quantized device"
    assert abs(ph_hi - ph_lo) < 60.0, "phantom should be ~resolution-independent"

"""Regression tests for the MR-LCMV coherent beamformer and GLCP refinement."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, IdealElement
from beamformer.api import beamform
from beamformer.coherent_lcmv import solve_coherent_lcmv, gain_locked_polish
from beamformer.controllers.metrics import get_metrics, get_sll


def _ceiling(element, N, u_t):
    n = np.arange(N)
    s_t = np.exp(-1j * np.pi * n * u_t)
    cg = element.c_grid
    best = -np.inf
    for psi in np.linspace(0, 2 * np.pi, 181):
        c = cg[np.argmax(np.real(np.conj(np.exp(1j * psi) * s_t)[:, None] * cg[None, :]), axis=1)]
        g = 20 * np.log10(abs(np.sum(c * np.exp(1j * np.pi * n * u_t))) / N + 1e-12)
        best = max(best, g)
    return best


@pytest.fixture
def element():
    return SyntheticVaractor(beta=0.8, folding=True)


@pytest.mark.parametrize("N,u_t,nulls", [
    (64, 0.2, [-0.4, 0.5]),
    (32, -0.3, [0.1, 0.5]),
    (64, 0.0, [-0.5, -0.25, 0.4]),
])
def test_glcp_near_ceiling_with_deep_nulls(element, N, u_t, nulls):
    task = Task("nulled", target_angles=[u_t], null_angles=nulls)
    res = beamform(task, element, N=N, solver="mrlcmv", refine="glcp")
    nd, ng, ptnr = get_metrics(res["weights"], task, element)

    ceiling = _ceiling(element, N, u_t)
    # Gain must stay near the physical ceiling. The default 'ptnr' gauge may
    # trade up to ~0.7 dB of gain for much deeper nulls (the offset-tuning
    # combination effect), so allow 1.0 dB headroom.
    assert ng >= ceiling - 1.0, f"gain {ng:.2f} dB collapsed below ceiling {ceiling:.2f} dB"
    # Nulls must be genuinely deep.
    assert nd <= -40.0, f"null only {nd:.2f} dB (expected <= -40 dB)"


def test_glcp_beats_ap_on_ptnr(element):
    """The whole point: GLCP keeps gain AND nulls; AP cannot do both."""
    N = 64
    task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
    ap = beamform(task, element, N=N, solver="ap", projection_method="weighted",
                  enable_local_refinement=True, use_pattern_cost=True)
    new = beamform(task, element, N=N, solver="mrlcmv", refine="glcp")
    _, ng_ap, ptnr_ap = get_metrics(ap["weights"], task, element)
    _, ng_new, ptnr_new = get_metrics(new["weights"], task, element)
    assert ng_new > ng_ap, "MR-LCMV should retain more gain than AP"
    assert ptnr_new > ptnr_ap + 20, "MR-LCMV+GLCP should win PTNR by a wide margin"


def test_ptnr_gauge_beats_gain_gauge(element):
    """Offset tuning: scoring the gauge by post-refinement PTNR should reach a
    deeper-null solution than scoring it by gain (which is ~psi-invariant)."""
    N = 64
    task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
    gain_g = beamform(task, element, N=N, solver="mrlcmv", refine="glcp", gauge="gain")
    ptnr_g = beamform(task, element, N=N, solver="mrlcmv", refine="glcp", gauge="ptnr")
    _, _, ptnr_a = get_metrics(gain_g["weights"], task, element)
    _, _, ptnr_b = get_metrics(ptnr_g["weights"], task, element)
    assert ptnr_b >= ptnr_a - 1e-6, "ptnr gauge should not be worse than gain gauge"


def test_gain_locked_polish_preserves_gain(element):
    """GLCP must not drop the main-beam magnitude below its gain floor."""
    N = 64
    task = Task("nulled", target_angles=[0.1], null_angles=[-0.3, 0.45])
    V0, c0 = solve_coherent_lcmv(task, element, N)
    g0 = abs(np.sum(c0 * np.exp(1j * np.pi * np.arange(N) * 0.1)))
    V1, c1 = gain_locked_polish(c0, V0, task, element, rounds=6, gain_floor=0.99)
    g1 = abs(np.sum(c1 * np.exp(1j * np.pi * np.arange(N) * 0.1)))
    assert g1 >= 0.99 * g0 - 1e-6, f"GLCP violated gain floor: {g1:.3f} < 0.99*{g0:.3f}"


def test_sll_suppression(element):
    """SLL active-set should push sidelobes toward the ceiling without nulls."""
    N = 64
    task = Task("pencil", target_angles=[0.0])
    task.null_angles = []
    task.sll_ceiling = 0.05  # -26 dB relative to N
    res = beamform(task, element, N=N, solver="mrlcmv", refine="glcp", suppress_sll=True)
    base = beamform(task, element, N=N, solver="mrlcmv", refine="none", suppress_sll=False)
    sll_on = get_sll(res["weights"], task)
    sll_off = get_sll(base["weights"], task)
    assert sll_on <= sll_off + 1e-6, "SLL suppression should not raise sidelobes"


def test_ideal_element_perfect_nulls(element):
    """On a lossless unit-modulus element, GLCP should give near-perfect nulls."""
    N = 64
    ideal = IdealElement(n_points=1000)
    task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
    res = beamform(task, ideal, N=N, solver="mrlcmv", refine="glcp")
    nd, ng, ptnr = get_metrics(res["weights"], task, ideal)
    assert nd <= -45.0
    assert ng >= -0.5  # ideal element has ~0 dB ceiling

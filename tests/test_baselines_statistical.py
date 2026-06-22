"""Statistical tests for MR-LCMV(+GLCP) against the family of baselines.

Single-task assertions are brittle: on some geometries the bare MR-LCMV anchor
already nails the null, on others GLCP (or a global searcher) wins. So we assert
DISTRIBUTIONAL properties over many random tasks, on the *realistic discrete*
(measured-grid) manifold -- the model of real hardware, where continuous
gradient/least-squares methods cannot exploit analytic smoothness.

Run:  ./.venv/Scripts/python.exe -m pytest tests/test_baselines_statistical.py -q
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer import baselines as B

N = 24
N_TASKS = 6
# A fast but representative slice of families (heavy DFO trimmed for test speed).
SOLVERS = ["Coordinate Descent", "Perturbation Null", "Basin Hopping",
           "OBH-ZKD (prior)", "MR-LCMV (ours)", "MR-LCMV+GLCP (ours)"]
BUDGET = {
    "Coordinate Descent": dict(sweeps=4),
    "Perturbation Null": dict(rounds=6),
    "Basin Hopping": dict(niter=8),
    "OBH-ZKD (prior)": dict(),
    "MR-LCMV (ours)": dict(),
    "MR-LCMV+GLCP (ours)": dict(),
}


def _make_tasks(n, seed=777):
    rng = np.random.default_rng(seed)
    tasks = []
    while len(tasks) < n:
        ut = round(float(rng.uniform(-0.5, 0.5)), 3)
        k = int(rng.integers(1, 4))
        nulls, tries = [], 0
        while len(nulls) < k and tries < 60:
            un = round(float(rng.uniform(-0.85, 0.85)), 3)
            if abs(un - ut) > 0.15 and all(abs(un - x) > 0.1 for x in nulls):
                nulls.append(un)
            tries += 1
        if nulls:
            tasks.append(Task("nulled", target_angles=[ut], null_angles=sorted(nulls)))
    return tasks


@pytest.fixture(scope="module")
def sweep():
    """Run every solver on every task once (discrete manifold); cache the rows."""
    element = SyntheticVaractor(beta=0.8, folding=True)
    tasks = _make_tasks(N_TASKS)
    rows = []
    for ti, task in enumerate(tasks):
        for name in SOLVERS:
            m = B.run_solver(name, task, element, N, seed=0, discrete=True, **BUDGET[name])
            rows.append(dict(solver=name, task=ti, **m))
    return rows


def _by(rows, name):
    return [r for r in rows if r["solver"] == name]


def test_ours_keeps_gain_near_ceiling(sweep):
    """Gain-first design: MR-LCMV+GLCP stays within ~1 dB of the ceiling on average."""
    g = [r["gain_loss"] for r in _by(sweep, "MR-LCMV+GLCP (ours)")]
    assert np.mean(g) <= 1.0, f"mean gain loss {np.mean(g):.2f} dB too high"


def test_ours_digs_deep_nulls_on_average(sweep):
    """On the realistic discrete manifold, GLCP reaches deep nulls on average."""
    wn = [r["worst_null"] for r in _by(sweep, "MR-LCMV+GLCP (ours)")]
    assert np.mean(wn) <= -40.0, f"mean worst null only {np.mean(wn):.1f} dB"


def test_glcp_improves_over_bare_mrlcmv(sweep):
    """GLCP should beat the bare anchor on PTNR in the MAJORITY of tasks
    (not necessarily all -- on easy single-null geometries the anchor already wins)."""
    bare = {r["task"]: r["ptnr"] for r in _by(sweep, "MR-LCMV (ours)")}
    full = {r["task"]: r["ptnr"] for r in _by(sweep, "MR-LCMV+GLCP (ours)")}
    wins = sum(full[t] > bare[t] + 0.5 for t in full)
    assert wins >= 0.5 * len(full), f"GLCP only improved {wins}/{len(full)} tasks"


def test_ours_best_gain_among_deep_nullers(sweep):
    """The gain-first claim, properly scoped: among methods that actually reach
    deep nulls (worst null <= -30 dB on the real device), ours retains the most gain."""
    deep = {}
    for r in sweep:
        if r["worst_null"] <= -30.0:
            deep.setdefault(r["solver"], []).append(r["gain_loss"])
    assert "MR-LCMV+GLCP (ours)" in deep, "ours failed to reach deep nulls"
    ours_loss = np.mean(deep["MR-LCMV+GLCP (ours)"])
    for name, losses in deep.items():
        if name == "MR-LCMV+GLCP (ours)":
            continue
        assert ours_loss <= np.mean(losses) + 1e-6, (
            f"{name} retained more gain than ours among deep-nullers "
            f"({np.mean(losses):.2f} < {ours_loss:.2f} dB)")


def test_ours_competitive_ptnr_winrate(sweep):
    """Across random tasks, ours (GLCP) is within 0.5 dB of the best PTNR on a
    healthy share of tasks -- it need not win every one (that's the point of
    reporting this statistically)."""
    best = {}
    for r in sweep:
        best[r["task"]] = max(best.get(r["task"], -np.inf), r["ptnr"])
    full = _by(sweep, "MR-LCMV+GLCP (ours)")
    wins = sum(r["ptnr"] >= best[r["task"]] - 0.5 for r in full)
    assert wins >= 0.4 * len(full), f"ours only top-tier on {wins}/{len(full)} tasks"

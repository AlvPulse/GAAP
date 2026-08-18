"""
SOTA comparison: MR-LCMV+GLCP vs the field.

Baselines (from beamformer/sota_solvers.py and the AP backbone):
  - Schelkunoff (analytic null synthesis, projected to manifold)
  - AP (euclidean / weighted / weighted+LR)            [the repo backbone]
  - PGD (finite-difference projected gradient descent)
  - Coordinate descent (exhaustive per-element)
  - ADMM (unimodular splitting + manifold projection)
  - CMA-ES (derivative-free global)
  - Dual annealing (derivative-free global)
  - OBH-ZKD (the project's prior proposed method)
  - MR-LCMV (none) and MR-LCMV+GLCP (ours)

FAIRNESS: sota_solvers.py uses a conjugate AF with psi=pi*sin(theta), which both
flips the u-axis and applies an extra sin() relative to the rest of the repo's
u-space convention. We therefore feed those solvers theta=arcsin(-u) so they
optimize the SAME physical directions, then score every method identically with
controllers.metrics.get_metrics in u-space. Local/global methods get the same
matched-filter manifold warm start.

    ./.venv/Scripts/python.exe experiments/sota_comparison.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.api import beamform
from beamformer.controllers.metrics import get_metrics
from beamformer.synthesis import synthesize_schelkunoff
from beamformer import sota_solvers as S


def theta_task(task_u):
    """Map a u-space task to the sin-convention used by sota_solvers (theta=arcsin(-u))."""
    f = lambda u: float(np.arcsin(np.clip(-u, -1, 1)))
    return Task("nulled",
                target_angles=[f(u) for u in task_u.target_angles],
                null_angles=[f(u) for u in task_u.null_angles])


def warm_start_V(task_u, element, N):
    """Matched-filter manifold projection -- same start MR-LCMV uses."""
    n = np.arange(N)
    s_t = np.exp(-1j * np.pi * n * task_u.target_angles[0])  # exp(-j pi n u_t)
    V, _ = S.proj_manifold(s_t, element, N)
    return V


def run_case(name, N, task_u, element):
    print(f"\n{'='*82}\n{name}  (N={N}, target u={task_u.target_angles[0]}, nulls={list(task_u.null_angles)})\n{'='*82}")
    tt = theta_task(task_u)
    V0 = warm_start_V(task_u, element, N)
    hdr = f"{'method':30s} {'gain dB':>8s} {'null dB':>9s} {'PTNR dB':>8s} {'evals':>8s} {'ms':>7s}"
    print(hdr); print("-" * len(hdr))

    def show(label, c, evals, ms):
        nd, ng, ptnr = get_metrics(c, task_u, element)
        ev = f"{evals:>8}" if evals is not None else f"{'-':>8}"
        print(f"{label:30s} {ng:8.2f} {nd:9.2f} {ptnr:8.2f} {ev} {ms:7.0f}")

    def timed(fn):
        t = time.perf_counter()
        try:
            out = fn()
            return out, (time.perf_counter() - t) * 1e3
        except Exception as e:
            print(f"  (error: {repr(e)[:70]})")
            return None, (time.perf_counter() - t) * 1e3

    # Schelkunoff analytic, projected
    def f_sch():
        w = synthesize_schelkunoff(N, task_u.target_angles[0], task_u.null_angles)
        _, c = S.proj_manifold(w, element, N)
        return c, None
    out, ms = timed(f_sch);  out and show("Schelkunoff (projected)", out[0], out[1], ms)

    # AP backbone
    for label, cfg in [("AP euclidean", dict(projection_method="euclidean")),
                       ("AP weighted", dict(projection_method="weighted")),
                       ("AP weighted + LR", dict(projection_method="weighted", enable_local_refinement=True))]:
        out, ms = timed(lambda cfg=cfg: (beamform(task_u, element, N=N, K_max=80,
                        null_init_method="schelkunoff", use_pattern_cost=True, **cfg)["weights"], None))
        out and show(label, out[0], out[1], ms)

    # sota_solvers (theta-mapped, warm started)
    sota = [
        ("PGD",             lambda: S.solve_pgd_baseline(V0.copy(), tt, element, N, max_iter=150)),
        ("Coordinate descent", lambda: S.solve_coordinate_descent(V0.copy(), tt, element, N, max_iter=6)),
        ("ADMM",            lambda: S.solve_admm(tt, element, N, max_iter=200)),
        ("CMA-ES",          lambda: S.solve_cmaes(V0.copy(), tt, element, N, max_iter=120)),
        ("Dual annealing",  lambda: S.solve_sa_global(V0.copy(), tt, element, N, max_iter=80)),
     #   ("OBH-ZKD (prior)", lambda: S.solve_obh_zkd(tt, element, N)),
    ]
    for label, fn in sota:
        out, ms = timed(fn)
        if out is not None:
            _, c, evals = out
            show(label, c, evals, ms)

    # Ours
    out, ms = timed(lambda: (beamform(task_u, element, N=N, solver="mrlcmv", refine="none")["weights"], None))
    out and show("MR-LCMV (none) [ours]", out[0], out[1], ms)
    out, ms = timed(lambda: (beamform(task_u, element, N=N, solver="mrlcmv", refine="glcp", gauge="ptnr")["weights"], None))
    out and show("MR-LCMV+GLCP [ours]", out[0], out[1], ms)


def main():
    #element = SyntheticVaractor(beta=0.8, folding=True)
    element = MeasuredVaractor()
    cases = [
        ("Canonical", 32, Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])),
        ("Three nulls", 32, Task("nulled", target_angles=[0.0], null_angles=[-0.5, -0.25, 0.4])),
    ]
    for name, N, task in cases:
        run_case(name, N, task, element)
    print("\nNote: 'evals' = objective evaluations (cost-function calls) where the solver reports them.")
    print("MR-LCMV is closed-form + few-shot; it has no iterative cost-eval count.")


if __name__ == "__main__":
    main()

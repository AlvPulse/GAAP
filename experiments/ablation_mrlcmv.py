"""
Ablation study for MR-LCMV+GLCP -- evidence that each component earns its place
(not a heuristic soup).

Components under test:
  R  gain-aware manifold retraction  (vs plain Euclidean projection)
  A  closed-form LCMV anchor          (dual warm start vs mu=0)
  D  Gauss-Newton dual correction     (iterative null repair)
  G  Gain-Locked Coordinate Polish    (vs no final refinement)
  P  PTNR gauge                        (offset chosen by post-refine PTNR vs gain)

We report a CUMULATIVE build-up (each row adds one component) and a LEAVE-ONE-OUT
(remove one component from the full system). A component is justified if turning
it off measurably degrades gain, null depth, PTNR, or convergence speed.

    ./.venv/Scripts/python.exe experiments/ablation_mrlcmv.py
"""
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.controllers.metrics import get_metrics
from beamformer.coherent_lcmv import _retract, _af, _get_manifold, gain_locked_polish


def solve_variant(task, element, N, psi=0.0, gain_retract=True, anchor=True,
                  n_dual=8, glcp=True, glcp_rounds=6):
    """Parametric MR-LCMV with each component switchable, at a fixed gauge psi."""
    V_grid, c_grid = _get_manifold(element)
    n = np.arange(N)
    u_t = task.target_angles[0]
    s_t = np.exp(-1j * np.pi * n * u_t)
    nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []
    tgt0 = np.exp(1j * psi) * s_t

    def retract(target):
        if gain_retract:
            c, V, _ = _retract(target, c_grid, V_grid)
            return c, V
        V = np.zeros(N); c = np.zeros(N, dtype=np.complex128)
        for i in range(N):
            V[i], c[i], _ = element.project(target[i], method="euclidean")
        return c, V

    if not nulls:
        c, V = retract(tgt0)
    else:
        A = np.column_stack([np.exp(1j * np.pi * n * um) for um in nulls])
        G = A.T @ np.conj(A)
        G += 1e-6 * np.trace(G).real / len(nulls) * np.eye(len(nulls))
        Ginv = np.linalg.inv(G)
        mu = Ginv @ (A.T @ tgt0) if anchor else np.zeros(len(nulls), dtype=np.complex128)
        c, V = retract(tgt0 - np.conj(A) @ mu)
        best = (c.copy(), V.copy(), float(max(abs(A.T @ c))))
        for _ in range(n_dual):
            mu = mu + Ginv @ (A.T @ c)
            c, V = retract(tgt0 - np.conj(A) @ mu)
            leak = float(max(abs(A.T @ c)))
            if leak < best[2]:
                best = (c.copy(), V.copy(), leak)
        c, V = best[0], best[1]

    if glcp and nulls:
        V, c = gain_locked_polish(c, V, task, element, null_angles=nulls, rounds=glcp_rounds)
    return V, c


def with_gauge(task, element, N, gauge="ptnr", samples=24, **flags):
    """Wrap solve_variant with a global-gauge search."""
    nulls = list(task.null_angles) if getattr(task, "null_angles", None) else []
    if gauge == "fixed0":
        return solve_variant(task, element, N, psi=0.0, **flags)
    u_t = task.target_angles[0]
    best = None
    for psi in np.linspace(0, 2 * np.pi, samples, endpoint=False):
        V, c = solve_variant(task, element, N, psi=psi, **flags)
        g = 20 * np.log10(abs(_af(c, u_t, N)) + 1e-12)
        if gauge == "ptnr" and nulls:
            nd = np.mean([20 * np.log10(abs(_af(c, um, N)) + 1e-12) for um in nulls])
            score = (g - nd) + 0.01 * g
        else:
            score = g
        if best is None or score > best[0]:
            best = (score, V, c)
    return best[1], best[2]


def row(label, task, element, N, **kw):
    V, c = with_gauge(task, element, N, **kw)
    nd, ng, ptnr = get_metrics(c, task, element)
    print(f"  {label:42s} gain={ng:6.2f}  null={nd:8.2f}  PTNR={ptnr:6.2f}")
    return ng, nd, ptnr


def run_case(name, N, task, element):
    print(f"\n{'='*72}\n{name}  (N={N}, target={task.target_angles[0]}, nulls={list(task.null_angles)})\n{'='*72}")

    print("\n[CUMULATIVE]  start naive, add one component per row")
    row("B0  Euclidean proj, matched filter (psi=0)", task, element, N,
        gauge="fixed0", gain_retract=False, anchor=False, n_dual=0, glcp=False)
    row("B1  + gain-aware retraction (R)", task, element, N,
        gauge="fixed0", gain_retract=True, anchor=False, n_dual=0, glcp=False)
    row("B2  + LCMV anchor + dual correction (A,D)", task, element, N,
        gauge="fixed0", gain_retract=True, anchor=True, n_dual=8, glcp=False)
    row("B3  + GLCP (G)", task, element, N,
        gauge="fixed0", gain_retract=True, anchor=True, n_dual=8, glcp=True)
    full = row("B4  + PTNR gauge (P)  [FULL SYSTEM]", task, element, N,
               gauge="ptnr", gain_retract=True, anchor=True, n_dual=8, glcp=True)

    print("\n[LEAVE-ONE-OUT]  remove one component from the full system")
    loo = [
        ("full - R (use Euclidean retraction)", dict(gauge="ptnr", gain_retract=False, anchor=True, n_dual=8, glcp=True)),
        ("full - A (no LCMV anchor)",           dict(gauge="ptnr", gain_retract=True, anchor=False, n_dual=8, glcp=True)),
        ("full - D (no dual correction)",       dict(gauge="ptnr", gain_retract=True, anchor=True, n_dual=0, glcp=True)),
        ("full - G (no GLCP)",                  dict(gauge="ptnr", gain_retract=True, anchor=True, n_dual=8, glcp=False)),
        ("full - P (gain gauge)",               dict(gauge="gain", gain_retract=True, anchor=True, n_dual=8, glcp=True)),
    ]
    for label, kw in loo:
        ng, nd, ptnr = row(label, task, element, N, **kw)
        print(f"      vs FULL: dGain={ng-full[0]:+5.2f}  dNull={nd-full[1]:+6.2f}  dPTNR={ptnr-full[2]:+6.2f}")

    # Is the LCMV front-end needed, or is GLCP+gauge enough? (quality vs speed)
    print("\n[MINIMAL vs FULL]  is the LCMV front-end worth it?")
    t = time.perf_counter()
    Vm, cm = with_gauge(task, element, N, gauge="ptnr", samples=24,
                        gain_retract=False, anchor=False, n_dual=0, glcp=True, glcp_rounds=8)
    tm = time.perf_counter() - t
    ndm, ngm, pm = get_metrics(cm, task, element)
    t = time.perf_counter()
    Vf, cf = with_gauge(task, element, N, gauge="ptnr", samples=24,
                        gain_retract=True, anchor=True, n_dual=8, glcp=True, glcp_rounds=6)
    tf = time.perf_counter() - t
    ndf, ngf, pf = get_metrics(cf, task, element)
    print(f"  minimal GLCP+gauge (Euclid, 8 rounds)      gain={ngm:6.2f} null={ndm:8.2f} PTNR={pm:6.2f}  {tm*1e3:5.0f} ms")
    print(f"  FULL (R+A+D warm start, 6 rounds)          gain={ngf:6.2f} null={ndf:8.2f} PTNR={pf:6.2f}  {tf*1e3:5.0f} ms")

    # Anchor+dual value as a warm start that accelerates GLCP convergence.
    print("\n[WARM-START VALUE]  null depth (dB) vs GLCP rounds, fixed psi=0")
    print(f"  {'GLCP rounds':14s} " + " ".join(f"{r:>7d}" for r in (0, 1, 2, 4, 8)))
    for tag, flags in [("cold (Euclid MF)", dict(gain_retract=False, anchor=False, n_dual=0)),
                       ("warm (R+A+D)",     dict(gain_retract=True, anchor=True, n_dual=8))]:
        depths = []
        for r in (0, 1, 2, 4, 8):
            V, c = solve_variant(task, element, N, psi=0.0, glcp=(r > 0), glcp_rounds=max(r, 1), **flags)
            if r == 0:
                V, c = solve_variant(task, element, N, psi=0.0, glcp=False, **flags)
            nd, _, _ = get_metrics(c, task, element)
            depths.append(nd)
        print(f"  {tag:14s} " + " ".join(f"{d:7.1f}" for d in depths))


def main():
    #element = SyntheticVaractor(beta=0.8, folding=True)
    element = MeasuredVaractor()
    cases = [
        ("Canonical", 64, Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])),
        ("Steered",   32, Task("nulled", target_angles=[-0.3], null_angles=[0.1, 0.5])),
    ]
    for name, N, task in cases:
        run_case(name, N, task, element)
    print("\nA component is justified if removing it degrades a metric or convergence.")


if __name__ == "__main__":
    main()

"""
Does offset (global-gauge psi) tuning improve MR-LCMV?
=====================================================

The global offset does NOT change the inter-element phase taper (fixed by u_t); it
slides every element to a different absolute phase, i.e. a different point on the
lossy amplitude-phase manifold. MR-LCMV already sweeps this gauge (psi) uniformly.

This study answers three questions:
  Q1. What does the psi -> quality landscape look like (pre/post GLCP)? Is the
      gain-optimal psi the same as the PTNR-optimal psi?
  Q2. Do the repo's one-shot offset selectors (select_optimal_geometry,
      optimize_offset_only) pick a good psi -- as well as / faster than a sweep?
  Q3. Can a cheaper search (coarse-to-fine, golden section) match the 72-pt sweep?

Cost is counted in MR-LCMV gauge-solves (the dominant op). Quality is post-GLCP
gain / null / PTNR. Run:

    ./.venv/Scripts/python.exe experiments/offset_tuning_study.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer.coherent_lcmv import solve_coherent_lcmv, gain_locked_polish, _af
from beamformer.controllers.metrics import get_metrics
from beamformer.geometry import compute_aperture_potential, compute_array_controllability
from beamformer.geometry_selection import select_optimal_geometry
from beamformer.baseline_optimization import optimize_offset_only


def eval_psi(task, element, N, psi, do_glcp=True):
    """One gauge-solve at fixed psi (+ optional GLCP). Returns metrics dict."""
    V, c, info = solve_coherent_lcmv(task, element, N, psi_grid=[psi], return_history=True)
    pre = get_metrics(c, task, element)  # (null_db, gain_db, ptnr)
    E_eff = compute_aperture_potential(c)
    C = compute_array_controllability(V, element)
    v_mid = (element.v_max - element.v_min) / 2.0 + element.v_min
    B = float(np.mean(V > v_mid))
    out = dict(psi=psi, pre_gain=pre[1], pre_null=pre[0], E_eff=E_eff, C=C, B=B)
    if do_glcp:
        Vg, cg = gain_locked_polish(c, V, task, element, null_angles=info["null_list"])
        nd, ng, ptnr = get_metrics(cg, task, element)
        out.update(gain=ng, null=nd, ptnr=ptnr)
    return out


def landscape(task, element, N, n=240):
    rows = [eval_psi(task, element, N, p) for p in np.linspace(0, 2 * np.pi, n, endpoint=False)]
    return rows


def argbest(rows, key, maximize=True):
    return (max if maximize else min)(rows, key=lambda r: r[key])


def corr(rows, a, b):
    x = np.array([r[a] for r in rows]); y = np.array([r[b] for r in rows])
    if x.std() < 1e-9 or y.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def run_case(name, N, task, element):
    print(f"\n{'='*78}\n{name}  (N={N}, target u={task.target_angles[0]}, nulls={list(task.null_angles)})\n{'='*78}")
    u_t = task.target_angles[0]
    n = np.arange(N)
    s_t = np.exp(-1j * np.pi * n * u_t)

    # ---- Q1: landscape (reference) ----
    rows = landscape(task, element, N, n=240)
    ref_ptnr = argbest(rows, "ptnr", True)
    ref_gain = argbest(rows, "gain", True)
    ref_pregain = argbest(rows, "pre_gain", True)
    span_gain = max(r["gain"] for r in rows) - min(r["gain"] for r in rows)
    span_ptnr = max(r["ptnr"] for r in rows) - min(r["ptnr"] for r in rows)
    print("\n[Q1] psi -> quality landscape (240-pt reference)")
    print(f"  post-GLCP gain spans {span_gain:5.2f} dB ;  PTNR spans {span_ptnr:5.2f} dB across psi")
    print(f"  best PTNR     : psi={ref_ptnr['psi']:.2f}  gain={ref_ptnr['gain']:.2f}  null={ref_ptnr['null']:.1f}  PTNR={ref_ptnr['ptnr']:.1f}")
    print(f"  best gain     : psi={ref_gain['psi']:.2f}  gain={ref_gain['gain']:.2f}  null={ref_gain['null']:.1f}  PTNR={ref_gain['ptnr']:.1f}")
    print(f"  best PRE-gain : psi={ref_pregain['psi']:.2f}  gain={ref_pregain['gain']:.2f}  null={ref_pregain['null']:.1f}  PTNR={ref_pregain['ptnr']:.1f}")
    print(f"  |psi(best gain) - psi(best PTNR)| = {abs(ref_gain['psi']-ref_ptnr['psi']):.2f} rad")
    print(f"  corr(E_eff, post-gain)={corr(rows,'E_eff','gain'):+.2f}  "
          f"corr(C, post-null)={corr(rows,'C','null'):+.2f}  "
          f"corr(pre_gain, post-gain)={corr(rows,'pre_gain','gain'):+.2f}")

    best = ref_ptnr  # reference optimum

    # ---- Q2/Q3: strategy comparison ----
    def report(label, psi, n_solves, extra=""):
        r = eval_psi(task, element, N, psi)
        dptnr = r["ptnr"] - best["ptnr"]
        print(f"  {label:30s} psi={psi:4.2f}  gain={r['gain']:6.2f}  null={r['null']:7.1f}  "
              f"PTNR={r['ptnr']:6.1f}  dPTNR={dptnr:+5.1f}  solves={n_solves:>3}{extra}")

    print("\n[Q2/Q3] strategy comparison  (cost = # MR-LCMV gauge-solves; dPTNR vs 240-ref)")
    report("reference best (240 sweep)", best["psi"], 240)

    # uniform sweeps of varying density
    for k in (72, 24, 12):
        psis = np.linspace(0, 2 * np.pi, k, endpoint=False)
        rk = [eval_psi(task, element, N, p, do_glcp=False) for p in psis]
        # pick by pre-GLCP score (gain - 0.5*null-violation), as the solver does
        def score(r):
            pen = max(0.0, r["pre_null"] - (20*np.log10(N) - 45))
            return r["pre_gain"] - 0.5 * pen
        psi_k = max(rk, key=score)["psi"]
        report(f"uniform-{k} (default uses 72)", psi_k, k)

    # coarse-to-fine: 9 coarse + 9 local refine
    coarse = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    rc = [eval_psi(task, element, N, p, do_glcp=False) for p in coarse]
    p0 = max(rc, key=lambda r: r["pre_gain"])["psi"]
    win = 2 * np.pi / 9
    fine = np.linspace(p0 - win, p0 + win, 9)
    rf = [eval_psi(task, element, N, p, do_glcp=False) for p in fine]
    psi_ctf = max(rf, key=lambda r: r["pre_gain"])["psi"] % (2 * np.pi)
    report("coarse-to-fine (9+9)", psi_ctf, 18)

    # golden-section inside best coarse bracket (assumes locally unimodal)
    def gss(f, a, b, iters=10):
        gr = (np.sqrt(5) - 1) / 2
        c1 = b - gr * (b - a); c2 = a + gr * (b - a)
        f1, f2 = f(c1), f(c2); ev = 2
        for _ in range(iters):
            if f1 > f2:
                b, c2, f2 = c2, c1, f1
                c1 = b - gr * (b - a); f1 = f(c1)
            else:
                a, c1, f1 = c1, c2, f2
                c2 = a + gr * (b - a); f2 = f(c2)
            ev += 1
        return ((a + b) / 2, ev)
    fpre = lambda p: eval_psi(task, element, N, p % (2*np.pi), do_glcp=False)["pre_gain"]
    psi_g, ev_g = gss(fpre, p0 - win, p0 + win, iters=8)
    report("golden-section (6 coarse+gss)", psi_g % (2*np.pi), 6 + ev_g)

    # repo one-shot offset selectors (cheap internal projection scan, no MR-LCMV solve)
    psi_sog = select_optimal_geometry(s_t, element, N, projection_method="weighted")
    report("select_optimal_geometry (repo)", psi_sog, 1, "  +180 cheap proj")
    psi_off, _, _ = optimize_offset_only(s_t, element, projection_method="weighted")
    report("optimize_offset_only (repo)", psi_off, 1, "  +200 cheap proj")


def main():
    element = SyntheticVaractor(beta=0.8, folding=True)
    cases = [
        ("Canonical", 64, Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])),
        ("Steered",   32, Task("nulled", target_angles=[-0.3], null_angles=[0.1, 0.5])),
    ]
    t0 = time.perf_counter()
    for name, N, task in cases:
        run_case(name, N, task, element)
    print(f"\n(total {time.perf_counter()-t0:.1f}s)")


if __name__ == "__main__":
    main()

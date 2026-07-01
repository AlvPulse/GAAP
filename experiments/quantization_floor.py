"""
Quantization null-floor experiment (Stream 1: the discreteness theory).
=======================================================================

This experiment validates the two claims formalized in ``docs/THEORY.md``:

  Proposition 1 (phantom-null lemma).  A finite-difference / gradient optimizer
  whose probe step falls below the manifold grid step observes a zero gradient on
  the real (quantized) device, yet on the analytic model it drives the null
  residual toward machine zero. The deep null it "reports" on the analytic model
  is therefore unrealizable -- a smoothness artifact.

  Proposition 2 (quantization null floor).  When each element snaps to a grid with
  nearest-point error <= eps, the realized null residual is
  AF(u_m) = sum_n e_n a(u_m)_n with |e_n| <= eps. For generic geometries the
  achievable floor scales like eps * sqrt(N) (incoherent residual), i.e. the best
  achievable WORST-NULL DEPTH recedes by ~6 dB per octave (~20 dB per decade) of
  grid resolution. Grid-native solvers (ours) attain this floor; design-continuous
  -then-project solvers lag it; gradient-on-analytic is resolution-independent
  fantasy.

We sweep the manifold resolution (grid points -> effective phase bits), fix the
array/task, and measure the achievable worst-null for:
  * MR-LCMV+GLCP (ours)            -- grid-native, discrete (the real device)
  * Riemannian CG                  -- continuous design, then projected (discrete)
  * Trust-Region/LM on analytic    -- the phantom (smooth, resolution-independent)
and overlay the eps*sqrt(N) covering-radius prediction.

Output: experiments/figures/fig10_quantization_floor.(pdf|png) + pub_quantization.csv
Usage:  ./.venv/Scripts/python.exe experiments/quantization_floor.py
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer import baselines as B
from experiments.publication_figures import init_style, save, FIGDIR, OURS_COLOR, FAMILY_STYLE
from experiments.sota_families_benchmark import make_tasks


def grid_eps(c_grid):
    """Nearest-point quantization error proxy: half the median spacing between
    adjacent achievable manifold points (the covering radius of the 1-D curve)."""
    d = np.abs(np.diff(np.asarray(c_grid)))
    d = d[d > 0]
    return float(np.median(d) / 2.0) if d.size else np.nan


def _mean_std(xs):
    xs = np.asarray([x for x in xs if np.isfinite(x)], float)
    return (np.nan, np.nan) if xs.size == 0 else (float(xs.mean()), float(xs.std()))


def run(N=32, tasks=None, resolutions=None):
    # average over several geometries so the floor trend is not masked by the
    # per-task ripple of a single greedy GLCP solve (ours is deterministic/seed).
    tasks = tasks or make_tasks(4)
    resolutions = resolutions or [16, 24, 32, 48, 64, 96, 128, 192, 256, 384,
                                  512, 768, 1024, 1536, 2048, 3072, 4096]
    rows = []
    for npnts in resolutions:
        el = SyntheticVaractor(beta=0.8, folding=True, n_points=npnts)
        eps = grid_eps(el.c_grid)
        o, r, p = [], [], []
        for task in tasks:
            o.append(B.run_solver("MR-LCMV+GLCP (ours)", task, el, N, seed=0,
                                  discrete=True)["worst_null"])
            try:
                r.append(B.run_solver("Riemannian CG", task, el, N, seed=0,
                                      discrete=True, max_iter=250)["worst_null"])
            except Exception:
                pass
            try:
                p.append(B.run_solver("Trust-Region/LM", task, el, N, seed=0,
                                      discrete=False, max_nfev=300)["worst_null"])
            except Exception:
                pass
        bits = np.log2(max(npnts, 2))
        pred = 20 * np.log10(eps * np.sqrt(N) + 1e-18)   # eps*sqrt(N) covering model (abs dB)
        om, os_ = _mean_std(o); rm, _ = _mean_std(r); pm, _ = _mean_std(p)
        rows.append(dict(n=npnts, bits=bits, eps=eps, pred=pred,
                         ours=om, ours_s=os_, rcg=rm, phantom=pm))
        print(f"  n={npnts:5d} (~{bits:4.1f} bits)  eps={eps:.4f}  "
              f"ours={om:7.1f}±{os_:4.1f}  rcg={rm:7.1f}  phantom={pm:7.1f}  pred={pred:7.1f}")
    return rows


def fit_slope(x_log10, y):
    x = np.asarray(x_log10, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan, np.nan
    a, b = np.polyfit(x[m], y[m], 1)
    return a, b   # dB per decade, intercept


def fig(plt, rows, N):
    ns = np.array([r["n"] for r in rows], float)
    ours = np.array([r["ours"] for r in rows], float)
    ours_s = np.array([r.get("ours_s", 0.0) for r in rows], float)
    rcg = np.array([r["rcg"] for r in rows], float)
    phantom = np.array([r["phantom"] for r in rows], float)
    pred = np.array([r["pred"] for r in rows], float)
    # offset the covering-radius prediction to ours (we predict SLOPE, not intercept)
    off = np.nanmedian(ours - pred)
    pred_fit = pred + off
    slope_ours, _ = fit_slope(np.log10(ns), ours)

    figu, ax = plt.subplots(figsize=(8.4, 6.2))
    ax.fill_between(ns, ours - ours_s, ours + ours_s, color=OURS_COLOR, alpha=0.15, zorder=2)
    ax.plot(ns, ours, "-*", color=OURS_COLOR, ms=12, lw=2.6, mec="black", mew=0.8,
            zorder=5, label="MR-LCMV+GLCP (ours): grid-native, real device")
    ax.plot(ns, rcg, "-v", color=FAMILY_STYLE["Manifold"][0], ms=6, lw=1.5,
            label="Riemannian CG: continuous design $\\to$ projected")
    ax.plot(ns, phantom, ":X", color="#9467bd", ms=6, lw=1.6,
            label="Trust-Region on analytic $c(V)$: PHANTOM (unrealizable)")
    ax.plot(ns, pred_fit, "--", color="#444444", lw=1.4,
            label=r"$\varepsilon\sqrt{N}$ covering-radius floor (slope $\approx -20$ dB/decade)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Manifold grid resolution (achievable points per element)")
    ax.set_ylabel("Achievable worst-null depth (dB)  -- deeper is better")
    ax.set_title("Quantization null floor: deep nulls require fine hardware resolution\n"
                 f"($N={N}$; ours tracks the $\\varepsilon\\sqrt{{N}}$ floor at "
                 f"{slope_ours:.1f} dB/decade; the analytic phantom is flat & unreachable)")
    # secondary axis: effective phase bits (force clean integer-bit ticks)
    import matplotlib.ticker as mticker
    secax = ax.secondary_xaxis("top", functions=(np.log2, lambda b: 2 ** b))
    secax.set_xlabel("effective resolution (bits, $\\log_2$ points)")
    secax.xaxis.set_major_locator(mticker.FixedLocator([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    secax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    secax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.legend(loc="lower left", fontsize=8.0)
    ax.grid(alpha=0.3, which="both")
    save(plt, figu, "fig10_quantization_floor")
    return slope_ours


def main():
    plt = init_style()
    os.makedirs(FIGDIR, exist_ok=True)
    N = 32
    print(f"[quantization-floor]  N={N}, sweeping manifold resolution ...")
    rows = run(N=N)
    slope = fig(plt, rows, N)
    p = os.path.join(FIGDIR, "pub_quantization.csv")
    fields = ["n", "bits", "eps", "pred", "ours", "ours_s", "rcg", "phantom"]
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  wrote {p}")
    print(f"\nMeasured floor slope for ours: {slope:.2f} dB/decade "
          f"(theory predicts -20 dB/decade from eps ~ 1/resolution and eps*sqrt(N) residual).")


if __name__ == "__main__":
    main()

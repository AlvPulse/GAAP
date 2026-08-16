"""
Robustness & statistical-significance study (Stream 2).
=======================================================

Two things a top-tier hardware paper must show:

(A) ROBUSTNESS.  A deep null is only useful if it survives real-world
    imperfections. After each solver produces device weights, we apply Monte-Carlo
    perturbations and measure how the worst-null degrades:
      * calibration error -- per-element amplitude (sigma_a) and phase (sigma_phi)
        jitter, c_n -> c_n (1+da) e^{j dphi};
      * dead/stuck elements -- K random elements forced to 0.
    We report mean +/- std over (tasks x MC draws). The takeaway is the
    *calibration spec* needed to preserve a method's advantage -- ours starts far
    deeper, so it keeps more null margin at any given error level.

(B) SIGNIFICANCE.  Beyond mean +/- std we run paired non-parametric hypothesis
    tests on per-task PTNR: Wilcoxon signed-rank (one-sided, ours > baseline) with
    Holm-Bonferroni correction across the family of comparisons, plus Cliff's delta
    dominance effect size. This turns "95% win-rate" into p-values and effect sizes.

Outputs (experiments/figures/):
  fig11_robustness_calibration.(pdf|png)
  fig12_robustness_dead_elements.(pdf|png)
  fig13_significance.(pdf|png)
  pub_robustness.csv, pub_significance.csv

Usage: ./.venv/Scripts/python.exe experiments/robustness_study.py [--tasks 16 --draws 200]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer import baselines as B
from experiments.sota_families_benchmark import make_tasks, budgets
from experiments.publication_figures import (init_style, save, FIGDIR, solver_color,
                                             solver_marker, is_ours, _solve_weights)

# Methods compared for robustness (one grid-native baseline + one DFO + ours).
ROBUST_SOLVERS = ["MR-LCMV+GLCP (ours)", "Perturbation Null", "Coordinate Descent"]
# Methods compared for significance (every method that ever digs a real null).
SIG_SOLVERS = ["Perturbation Null", "Cross-Entropy", "Coordinate Descent",
               "CMA-ES", "OBH-ZKD (prior)", "MR-LCMV (ours)"]
OURS = "MR-LCMV+GLCP (ours)"


def _af_metrics(c, task, N):
    n = np.arange(N)
    g = 20 * np.log10(abs(np.sum(c * np.exp(1j * np.pi * n * task.target_angles[0]))) + 1e-12)
    wn = max(20 * np.log10(abs(np.sum(c * np.exp(1j * np.pi * n * um))) + 1e-12)
             for um in task.null_angles)
    return g, wn, g - wn   # gain_abs, worst_null_abs, ptnr


# --------------------------------------------------------------------------- #
# (A) robustness Monte-Carlo
# --------------------------------------------------------------------------- #
def perturb_calibration(c, rng, sigma_a, sigma_phi_deg):
    da = rng.normal(0.0, sigma_a, c.shape)
    dphi = rng.normal(0.0, np.deg2rad(sigma_phi_deg), c.shape)
    return c * (1.0 + da) * np.exp(1j * dphi)


def perturb_dead(c, rng, k):
    cc = c.copy()
    if k > 0:
        idx = rng.choice(len(c), size=min(k, len(c)), replace=False)
        cc[idx] = 0.0
    return cc


def run_robustness(element, tasks, N, draws, seed=0):
    rng = np.random.default_rng(seed)
    phi_levels = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]      # deg RMS phase jitter
    dead_levels = [0, 1, 2, 3, 4, 6, 8]                    # number of dead elements
    bud = budgets(False)
    # pre-solve device weights once per (solver, task)
    weights = {}
    for name in ROBUST_SOLVERS:
        for ti, task in enumerate(tasks):
            try:
                weights[(name, ti)] = _solve_weights(name, task, element, N, seed=0,
                                                      discrete=True, **bud.get(name, {}))
            except Exception as e:
                sys.stderr.write(f"  ! solve {name} task{ti}: {repr(e)[:50]}\n")
    cal, dead = {}, {}
    for name in ROBUST_SOLVERS:
        cal[name] = {"phi": phi_levels, "mean": [], "std": []}
        dead[name] = {"k": dead_levels, "mean": [], "std": []}
        for sphi in phi_levels:
            vals = []
            for ti, task in enumerate(tasks):
                c = weights.get((name, ti))
                if c is None:
                    continue
                for _ in range(draws):
                    cp = perturb_calibration(c, rng, sigma_a=0.5 * sphi / 100.0,
                                             sigma_phi_deg=sphi)
                    vals.append(_af_metrics(cp, task, N)[1])
            cal[name]["mean"].append(float(np.mean(vals)) if vals else np.nan)
            cal[name]["std"].append(float(np.std(vals)) if vals else np.nan)
        for k in dead_levels:
            vals = []
            for ti, task in enumerate(tasks):
                c = weights.get((name, ti))
                if c is None:
                    continue
                reps = 1 if k == 0 else draws
                for _ in range(reps):
                    cp = perturb_dead(c, rng, k)
                    vals.append(_af_metrics(cp, task, N)[1])
            dead[name]["mean"].append(float(np.mean(vals)) if vals else np.nan)
            dead[name]["std"].append(float(np.std(vals)) if vals else np.nan)
        print(f"  robustness done: {name}")
    return cal, dead


def fig_robustness(plt, data, xs_key, xlabel, title, fname):
    figu, ax = plt.subplots(figsize=(8.0, 5.8))
    for name, d in data.items():
        x = d[xs_key]; m = np.array(d["mean"], float); s = np.array(d["std"], float)
        fam = B.SOLVERS[name][1]; col = solver_color(name, fam); ours = is_ours(name)
        ax.plot(x, m, marker=solver_marker(name, fam), color=col,
                lw=(2.6 if ours else 1.5), ms=(11 if ours else 6),
                mec="black" if ours else col, mew=0.8 if ours else 0.4,
                zorder=5 if ours else 3, label=name)
        ax.fill_between(x, m - s, m + s, color=col, alpha=0.12, zorder=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Worst-null depth after perturbation (dB)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    save(plt, figu, fname)


# --------------------------------------------------------------------------- #
# (B) significance: paired Wilcoxon + Holm + Cliff's delta
# --------------------------------------------------------------------------- #
def cliffs_delta(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))


def run_significance(element, tasks, N):
    from scipy.stats import wilcoxon
    bud = budgets(False)
    # per-task PTNR (baselines averaged over a few seeds for a stable per-task value)
    ours_ptnr, base_ptnr = {}, {n: {} for n in SIG_SOLVERS}
    for ti, task in enumerate(tasks):
        ours_ptnr[ti] = B.run_solver(OURS, task, element, N, seed=0, discrete=True)["ptnr"]
        for name in SIG_SOLVERS:
            stoch = B.SOLVERS[name][2]
            seeds = [0, 1, 2] if stoch else [0]
            vals = [B.run_solver(name, task, element, N, seed=s, discrete=True,
                                 **bud.get(name, {}))["ptnr"] for s in seeds]
            base_ptnr[name][ti] = float(np.nanmean(vals))
        sys.stderr.write(f"  significance: task {ti + 1}/{len(tasks)}\n")
    tids = sorted(ours_ptnr)
    o = np.array([ours_ptnr[t] for t in tids], float)
    rows = []
    raw_p = []
    for name in SIG_SOLVERS:
        b = np.array([base_ptnr[name][t] for t in tids], float)
        d = o - b
        try:
            stat, p = wilcoxon(d, alternative="greater")
        except ValueError:        # all-zero/identical diffs
            p = 1.0
        raw_p.append(p)
        rows.append(dict(baseline=name, n=len(d), median_adv=float(np.median(d)),
                         win=float(np.mean(d > 0)), cliffs=cliffs_delta(o, b), p=p))
    # Holm-Bonferroni across the family
    order = np.argsort(raw_p)
    m = len(raw_p)
    holm = [None] * m
    prev = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * raw_p[i])
        prev = max(prev, adj)
        holm[i] = prev
    for r, h in zip(rows, holm):
        r["p_holm"] = h
    rows.sort(key=lambda r: -r["median_adv"])
    return rows


def fig_significance(plt, rows):
    rows = sorted(rows, key=lambda r: r["median_adv"])
    names = [r["baseline"] for r in rows]
    adv = [r["median_adv"] for r in rows]
    figu, ax = plt.subplots(figsize=(8.0, 5.0))
    cols = ["#2ca02c" if r["p_holm"] < 0.05 else "#999999" for r in rows]
    ax.barh(range(len(names)), adv, color=cols, edgecolor="#333333", alpha=0.9)
    for i, r in enumerate(rows):
        star = ("***" if r["p_holm"] < 1e-3 else "**" if r["p_holm"] < 1e-2
                else "*" if r["p_holm"] < 5e-2 else "ns")
        ax.text(r["median_adv"] + 0.6, i,
                f"{r['median_adv']:.1f} dB  ({star}, $\\delta$={r['cliffs']:.2f})",
                va="center", fontsize=8.0)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Median PTNR advantage of MR-LCMV+GLCP (ours) over baseline (dB)")
    ax.set_title("Paired significance: ours vs each baseline\n"
                 "(Wilcoxon signed-rank, Holm-corrected; $\\delta$ = Cliff's dominance)")
    ax.axvline(0, color="black", lw=0.7)
    ax.grid(axis="x", alpha=0.3)
    save(plt, figu, "fig13_significance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--tasks", type=int, default=16)
    ap.add_argument("--draws", type=int, default=200)
    args = ap.parse_args()

    plt = init_style()
    os.makedirs(FIGDIR, exist_ok=True)
    # element = SyntheticVaractor(beta=0.8, folding=True)
    element = MeasuredVaractor()
    tasks = make_tasks(args.tasks)

    print(f"[A] Robustness Monte-Carlo ({len(tasks)} tasks x {args.draws} draws, N={args.N})")
    cal, dead = run_robustness(element, tasks, args.N, args.draws)
    fig_robustness(plt, cal, "phi",
                   "RMS per-element phase calibration error (deg)",
                   "Robustness to calibration error\n(amplitude error scaled with phase; mean $\\pm$ std)",
                   "fig11_robustness_calibration")
    fig_robustness(plt, dead, "k",
                   "Number of dead / stuck elements",
                   "Robustness to element failures (mean $\\pm$ std)",
                   "fig12_robustness_dead_elements")

    print(f"[B] Statistical significance ({len(tasks)} paired tasks)")
    sig = run_significance(element, tasks, args.N)
    fig_significance(plt, sig)

    # console + CSV
    print(f"\n{'baseline':22s}{'n':>4s}{'med adv dB':>12s}{'win':>6s}"
          f"{'Cliff d':>9s}{'p':>10s}{'p_holm':>10s}")
    for r in sig:
        print(f"{r['baseline']:22s}{r['n']:4d}{r['median_adv']:12.1f}"
              f"{r['win']*100:5.0f}%{r['cliffs']:9.2f}{r['p']:10.2e}{r['p_holm']:10.2e}")
    with open(os.path.join(FIGDIR, "pub_significance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["baseline", "n", "median_adv", "win",
                                          "cliffs", "p", "p_holm"])
        w.writeheader()
        for r in sig:
            w.writerow(r)
    robust_rows = []
    for name in ROBUST_SOLVERS:
        for i, phi in enumerate(cal[name]["phi"]):
            robust_rows.append(dict(solver=name, kind="calibration_deg", level=phi,
                                    mean=cal[name]["mean"][i], std=cal[name]["std"][i]))
        for i, k in enumerate(dead[name]["k"]):
            robust_rows.append(dict(solver=name, kind="dead_elements", level=k,
                                    mean=dead[name]["mean"][i], std=dead[name]["std"][i]))
    with open(os.path.join(FIGDIR, "pub_robustness.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["solver", "kind", "level", "mean", "std"])
        w.writeheader()
        for r in robust_rows:
            w.writerow(r)
    print(f"\nDone. Wrote fig11/fig12/fig13 + pub_robustness.csv + pub_significance.csv to {FIGDIR}")


if __name__ == "__main__":
    main()

"""
Publication-quality figures for the MR-LCMV(+GLCP) phase-only nullforming study.
================================================================================

This script regenerates every figure intended for a journal submission. It is a
THIN, reproducible layer over the validated, fair benchmark harness in
``beamformer.baselines`` -- every method minimizes the SAME penalized objective
over the SAME decision variable (control voltages on the real, measured hardware
manifold) and is scored by the SAME u-space metrics. See ``docs/ALGORITHMS.md``.

WHY NOT WALL-CLOCK TIME.  Wall-clock seconds are not a publishable cost metric:
they depend on the machine, the BLAS build, vectorization luck, and Python
overhead, and they are not reproducible across reviewers. We therefore report a
machine-independent cost: the number of **forward-model evaluations** ("oracle
calls"), i.e. how many times a method evaluates the array factor of a complete
candidate weight vector at the constraint directions. This is the standard
currency of derivative-free / numerical-optimization benchmarking and is exactly
what a Dolan-More performance profile is built on. For our closed-form method we
also report the (cheap, O(M)-incremental) Gain-Locked Coordinate Polish moves as
a SEPARATE quantity so the comparison is neither inflated nor flattering by
omission.

FIGURES (each saved as vector .pdf AND 300-dpi .png under experiments/figures/):
  fig1_distributions      box plots of PTNR / gain-loss / worst-null over instances
  fig2_success_heatmap     P(worst null <= threshold) heatmap, all solvers
  fig3_pareto              gain retention vs nulling Pareto (mean +/- std)
  fig4_winrate             per-solver PTNR win-rate
  fig5_perf_profile        Dolan-More performance profile keyed on ORACLE CALLS
  fig6_cost_quality        oracle calls vs PTNR efficiency frontier
  fig7_scaling             PTNR and oracle calls vs array size N (log-log)
  fig8_manifold_realism    smooth-analytic vs discrete-measured null collapse
  fig9_beam_patterns       representative array-factor patterns

CSV tables (for the paper) under experiments/figures/:
  pub_runs.csv, pub_summary.csv, pub_profile.csv, pub_scaling.csv,
  pub_realism.csv

Usage:
  ./.venv/Scripts/python.exe experiments/publication_figures.py            # full
  ./.venv/Scripts/python.exe experiments/publication_figures.py --quick    # smoke
  ./.venv/Scripts/python.exe experiments/publication_figures.py --tasks 12 --seeds 5
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer import baselines as B
from experiments.sota_families_benchmark import make_tasks, budgets

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
THRESHOLDS = [-20.0, -30.0, -40.0, -50.0]   # worst-null success levels (dB)
PROFILE_DB = -40.0                           # "solved" = worst null <= this (dB)
WIN_TOL = 0.5                                # dB tolerance for a PTNR win
OURS_HI = "MR-LCMV-certified"
OURS_LO = "MR-LCMV (ours)"

# Solvers omitted from the N-scaling study: their decision dimension IS N, so
# global/DFO cost grows super-linearly and they are not competitive past N~32.
SCALE_SUBSET = ["Coordinate Descent", "Riemannian CG", "Scaled ADMM",
                "Perturbation Null", "OBH-ZKD (prior)", OURS_LO, OURS_HI]


def is_ours(name):
    return "ours" in name.lower()

import json
from pathlib import Path


def save_task_suite(tasks, path):
    """
    Save the exact spatial task suite used in an experiment.
    """
    payload = []

    for i, task in enumerate(tasks):
        payload.append({
            "task_id": i,
            "type": getattr(task, "type", "nulled"),
            "target_angles": [
                float(x) for x in task.target_angles
            ],
            "null_angles": [
                float(x) for x in task.null_angles
            ],
        })

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return path

# --------------------------------------------------------------------------- #
# Matplotlib style (journal-grade, colorblind-aware)
# --------------------------------------------------------------------------- #
def init_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.titleweight": "bold",
        "legend.fontsize": 8.0,
        "legend.framealpha": 0.92,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.30,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "axes.edgecolor": "#333333",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "lines.linewidth": 1.6,
        "pdf.fonttype": 42,    # editable text in vector PDF (journal requirement)
        "ps.fonttype": 42,
    })
    return plt


# Family -> (color, marker). Colorblind-safe (Wong/Okabe-Ito derived).
FAMILY_STYLE = {
    "Local":          ("#0072B2", "o"),
    "Manifold":       ("#56B4E9", "v"),
    "Splitting":      ("#009E73", "s"),
    "Global":         ("#E69F00", "D"),
    "Quantum-insp.":  ("#CC79A7", "P"),
    "Relaxation":     ("#117733", "h"),
    "Hardware-aware": ("#7F7F7F", "X"),
    "Prior proposed": ("#8C564B", "p"),
    "Ours":           ("#C81E2E", "*"),
}
OURS_COLOR = "#C81E2E"
OURS_LO_COLOR = "#F08C00"   # bare anchor: amber, to distinguish from +GLCP



def solver_color(name, family):
    if name == OURS_HI:
        return OURS_COLOR
    if name == OURS_LO:
        return OURS_LO_COLOR
    return FAMILY_STYLE.get(family, ("#444444", "o"))[0]


def solver_marker(name, family):
    if is_ours(name):
        return "*"
    return FAMILY_STYLE.get(family, ("#444444", "o"))[1]


# --------------------------------------------------------------------------- #
# Sweep + aggregation (oracle-call cost, not wall-clock)
# --------------------------------------------------------------------------- #
def run_sweep(element, N, tasks, seeds, quick=False, discrete=True):
    bud = budgets(quick)
    names = list(B.SOLVERS.keys())
    rows = []
    n_total = sum((len(seeds) if B.SOLVERS[n][2] else 1) for n in names) * len(tasks)
    done = 0
    for ti, task in enumerate(tasks):
        for name in names:
            fn_seeds = seeds if B.SOLVERS[name][2] else seeds[:1]   # det. solver -> 1 run
            for seed in fn_seeds:
                try:
                    m = B.run_solver(name, task, element=element, N=N, seed=seed,
                                     discrete=discrete, **bud[name])
                    ok = True
                except Exception as e:
                    m = dict(family=B.SOLVERS[name][1], gain=np.nan, worst_null=np.nan,
                             avg_null=np.nan, ptnr=np.nan, gain_loss=np.nan,
                             evals=-1, ms=np.nan, glcp_updates=0)
                    ok = False
                    sys.stderr.write(f"  ! {name} task{ti} seed{seed}: {repr(e)[:70]}\n")
                done += 1
                rows.append(dict(solver=name, family=m["family"], task=ti, seed=seed,
                                 n_nulls=len(task.null_angles), ok=ok,
                                 **{k: m[k] for k in ("gain", "worst_null", "avg_null",
                                    "ptnr", "gain_loss", "evals", "ms", "glcp_updates")}))
        sys.stderr.write(f"  sweep: task {ti + 1}/{len(tasks)} ({done}/{n_total})\n")
    return rows


def _ms(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (np.nan, np.nan) if x.size == 0 else (float(np.mean(x)), float(np.std(x)))


def _fill_seeds(rows, names, seeds):
    """Replicate deterministic solvers (run once at seed 0) across all seeds so
    per-instance win-rate / profiles are not structurally capped. A deterministic
    solver gives the identical result every trial, so this is exact."""
    tasks = sorted(set(r["task"] for r in rows))
    have = {(r["solver"], r["task"], r["seed"]): r for r in rows if r["ok"]}
    filled = []
    for n in names:
        det = not B.SOLVERS[n][2]
        for t in tasks:
            for s in seeds:
                if (n, t, s) in have:
                    filled.append(have[(n, t, s)])
                elif det and (n, t, seeds[0]) in have:
                    filled.append({**have[(n, t, seeds[0])], "seed": s})
    return filled


def aggregate(rows, names, seeds):
    by_solver = {n: [r for r in rows if r["solver"] == n and r["ok"]] for n in names}
    filled = _fill_seeds(rows, names, seeds)
    instances = {}
    for r in filled:
        if not np.isnan(r["ptnr"]):
            instances.setdefault((r["task"], r["seed"]), []).append(r)
    wins = {n: 0 for n in names}
    n_inst = 0
    for inst in instances.values():
        best = max(x["ptnr"] for x in inst)
        n_inst += 1
        for x in inst:
            if x["ptnr"] >= best - WIN_TOL:
                wins[x["solver"]] += 1

    summary = []
    for n in names:
        rs = by_solver[n]
        if not rs:
            continue
        gm, gs = _ms([r["gain"] for r in rs])
        wm, ws = _ms([r["worst_null"] for r in rs])
        pm, ps = _ms([r["ptnr"] for r in rs])
        lm, ls = _ms([r["gain_loss"] for r in rs])
        succ = {th: float(np.mean([r["worst_null"] <= th for r in rs])) for th in THRESHOLDS}
        ev = np.array([r["evals"] for r in rs], float); ev = ev[ev >= 0]
        gl = np.array([r["glcp_updates"] for r in rs], float)
        summary.append(dict(
            solver=n, family=rs[0]["family"], runs=len(rs),
            gain_m=gm, gain_s=gs, wn_m=wm, wn_s=ws, ptnr_m=pm, ptnr_s=ps,
            loss_m=lm, loss_s=ls,
            evals=(float(np.mean(ev)) if ev.size else np.nan),
            evals_s=(float(np.std(ev)) if ev.size else np.nan),
            glcp=(float(np.mean(gl)) if gl.size else 0.0),
            **{f"s{int(-th)}": succ[th] for th in THRESHOLDS},
            winrate=(wins[n] / n_inst if n_inst else 0.0)))
    summary.sort(key=lambda d: -d["ptnr_m"])
    return summary, n_inst


def performance_profile(rows, names, seeds, thresh_db=PROFILE_DB):
    """Dolan-More profile with cost = ORACLE CALLS (forward-model evaluations).
    rho_s(tau) = fraction of instances solver s solves using <= tau x the fewest
    oracle calls any solver needed to solve that instance."""
    instances = {}
    for r in _fill_seeds(rows, names, seeds):
        instances.setdefault((r["task"], r["seed"]), []).append(r)
    P = len(instances)
    best_cost = {}
    for key, inst in instances.items():
        solved = [x["evals"] for x in inst
                  if x["worst_null"] <= thresh_db and x["evals"] is not None and x["evals"] >= 0]
        best_cost[key] = min(solved) if solved else None
    taus = [1, 2, 4, 8, 16, 32, 64, 128, 1e9]
    prof = {}
    for n in names:
        ratios = []
        for key, inst in instances.items():
            r = next((x for x in inst if x["solver"] == n), None)
            if (r and r["worst_null"] <= thresh_db and best_cost[key]
                    and r["evals"] is not None and r["evals"] >= 0):
                ratios.append(max(1.0, r["evals"] / best_cost[key]))
        prof[n] = [(sum(1 for rt in ratios if rt <= t) / P) if P else 0.0 for t in taus]
    return taus, prof, P


# --------------------------------------------------------------------------- #
# Figure 1 -- distributions (box plots) over random instances
# --------------------------------------------------------------------------- #
def fig_distributions(plt, rows, order):
    metrics = [("ptnr", "Peak-to-null ratio (dB)", False),
               ("loss", "Gain loss vs ceiling (dB)", True),
               ("worst_null", "Worst null depth (dB)", True)]
    # 'loss' is gain_loss; map key
    key_map = {"ptnr": "ptnr", "loss": "gain_loss", "worst_null": "worst_null"}
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 6.2))
    ylabels = list(reversed(order))   # best (top) -> worst (bottom)
    for ax, (mk, label, lower_better) in zip(axes, metrics):
        data, colors = [], []
        for name in ylabels:
            vals = [r[key_map[mk]] for r in rows
                    if r["solver"] == name and r["ok"] and not np.isnan(r[key_map[mk]])]
            data.append(vals if vals else [np.nan])
            fam = next(r["family"] for r in rows if r["solver"] == name)
            colors.append(solver_color(name, fam))
        bp = ax.boxplot(data, orientation="horizontal", patch_artist=True, widths=0.62,
                        showfliers=False, medianprops=dict(color="black", lw=1.3),
                        whiskerprops=dict(color="#555555"), capprops=dict(color="#555555"))
        for patch, name, col in zip(bp["boxes"], ylabels, colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.55 if is_ours(name) else 0.78)
            patch.set_edgecolor("black" if is_ours(name) else "#555555")
            patch.set_linewidth(1.6 if is_ours(name) else 0.8)
        # overlay jittered points
        rng = np.random.default_rng(0)
        for i, (vals, col) in enumerate(zip(data, colors), start=1):
            v = np.asarray(vals, float); v = v[~np.isnan(v)]
            if v.size:
                y = i + rng.uniform(-0.16, 0.16, v.size)
                ax.scatter(v, y, s=8, color=col, edgecolor="white", linewidth=0.2, zorder=3, alpha=0.8)
        ax.set_yticks(range(1, len(ylabels) + 1))
        ax.set_yticklabels([f"$\\bf{{{n.replace(' (ours)','')}}}$" if is_ours(n) else n
                            for n in ylabels])
        ax.set_xlabel(label + (" $\\downarrow$" if lower_better else " $\\uparrow$"))
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_title("(a) Nulling quality")
    axes[1].set_title("(b) Gain retention")
    axes[2].set_title("(c) Null depth")
    fig.suptitle("Distribution of outcomes over random nulling tasks "
                 "(box = IQR, line = median, dots = instances)", y=1.005, fontsize=12.5)
    save(plt, fig, "fig1_distributions")


# --------------------------------------------------------------------------- #
# Figure 2 -- success-probability heatmap
# --------------------------------------------------------------------------- #
def fig_success_heatmap(plt, summary):
    order = [d["solver"] for d in summary]
    M = np.array([[d[f"s{int(-th)}"] for th in THRESHOLDS] for d in summary]) * 100.0
    fig, ax = plt.subplots(figsize=(7.0, 7.4))
    im = ax.imshow(M, aspect="auto", cmap="YlGnBu", vmin=0, vmax=100)
    ax.set_xticks(range(len(THRESHOLDS)))
    ax.set_xticklabels([f"$\\leq${int(t)} dB" for t in THRESHOLDS])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"$\\bf{{{n.replace(' (ours)','')}}}$" if is_ours(n) else n
                        for n in order])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    color="white" if v > 55 else "#222222", fontsize=8.5,
                    fontweight="bold" if is_ours(order[i]) else "normal")
    ax.set_xlabel("Worst-null success threshold")
    ax.set_title("Reliability: P(worst null $\\leq$ threshold) [%]\nover random tasks (discrete measured manifold)")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("success probability (%)")
    save(plt, fig, "fig2_success_heatmap")


# --------------------------------------------------------------------------- #
# Figure 3 -- Pareto: gain retention vs nulling
# --------------------------------------------------------------------------- #
def fig_pareto(plt, summary):
    fig, ax = plt.subplots(figsize=(8.2, 6.4))
    seen_fam = set()
    for d in summary:
        col = solver_color(d["solver"], d["family"])
        mk = solver_marker(d["solver"], d["family"])
        ours = is_ours(d["solver"])
        lab = d["family"] if (d["family"] not in seen_fam) else None
        seen_fam.add(d["family"])
        ax.errorbar(d["loss_m"], d["ptnr_m"], xerr=d["loss_s"], yerr=d["ptnr_s"],
                    fmt=mk, ms=(20 if ours else 9), color=col,
                    mec="black", mew=(1.4 if ours else 0.6),
                    ecolor=col, elinewidth=1.0, capsize=2.5, alpha=0.95,
                    zorder=(5 if ours else 3), label=lab)
        if ours:
            ax.annotate(d["solver"].replace(" (ours)", "\n(ours)"),
                        (d["loss_m"], d["ptnr_m"]), fontsize=8.5, fontweight="bold",
                        color=col, xytext=(10, -4), textcoords="offset points")
    ax.set_xlabel("Gain loss vs hardware ceiling (dB)  $\\leftarrow$ better")
    ax.set_ylabel("Peak-to-null ratio (dB)  better $\\uparrow$")
    ax.set_title("Pareto front: gain retention vs nulling\n(mean $\\pm$ std over random tasks; top-left is best)")
    ax.legend(title="family", loc="lower right", ncol=2)
    ax.grid(alpha=0.3)
    save(plt, fig, "fig3_pareto")


# --------------------------------------------------------------------------- #
# Figure 4 -- PTNR win-rate
# --------------------------------------------------------------------------- #
def fig_winrate(plt, summary):
    order = sorted(summary, key=lambda d: d["winrate"])
    names = [d["solver"] for d in order]
    vals = [d["winrate"] * 100 for d in order]
    cols = [solver_color(d["solver"], d["family"]) for d in order]
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    bars = ax.barh(range(len(names)), vals, color=cols,
                   edgecolor=["black" if is_ours(n) else "#555555" for n in names],
                   linewidth=[1.6 if is_ours(n) else 0.6 for n in names], alpha=0.9)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(v + 1.0, i, f"{v:.0f}%", va="center", fontsize=8.5,
                fontweight="bold" if is_ours(names[i]) else "normal")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f"$\\bf{{{n.replace(' (ours)','')}}}$" if is_ours(n) else n
                        for n in names])
    ax.set_xlabel(f"PTNR win-rate (%)  -- within {WIN_TOL} dB of best per instance")
    ax.set_xlim(0, 105)
    ax.set_title("How often each method ties or wins the best PTNR")
    ax.grid(axis="x", alpha=0.3)
    save(plt, fig, "fig4_winrate")


# --------------------------------------------------------------------------- #
# Figure 5 -- Dolan-More performance profile keyed on ORACLE CALLS
# --------------------------------------------------------------------------- #
def fig_perf_profile(plt, taus, prof, P):
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    xt = [t if t < 1e8 else taus[-2] * 2 for t in taus]
    ranked = sorted(prof.keys(), key=lambda n: -prof[n][-1])
    for n in ranked:
        if max(prof[n]) == 0:
            continue
        fam = B.SOLVERS[n][1]
        col = solver_color(n, fam)
        ours = is_ours(n)
        ax.plot(xt, [v * 100 for v in prof[n]], marker=solver_marker(n, fam),
                ms=(9 if ours else 5), lw=(3.0 if ours else 1.4), color=col,
                mec="black" if ours else col, mew=0.8 if ours else 0.4,
                zorder=(5 if ours else 3), label=n)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(r"$\tau$  ($\times$ fewest oracle calls needed on that instance)")
    ax.set_ylabel(f"% instances solved (worst null $\\leq$ {PROFILE_DB:.0f} dB)")
    ax.set_title(f"Performance profile keyed on forward-model evaluations\n"
                 f"(P = {P} instances; left = also cheapest, up = more reliable)")
    ax.set_ylim(-3, 103)
    ax.legend(fontsize=7.0, loc="lower right", ncol=2)
    ax.grid(alpha=0.3, which="both")
    save(plt, fig, "fig5_perf_profile")


# --------------------------------------------------------------------------- #
# Figure 6 -- cost (oracle calls) vs quality (PTNR) efficiency frontier
# --------------------------------------------------------------------------- #
def fig_cost_quality(plt, summary):
    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    seen_fam = set()
    for d in summary:
        if np.isnan(d["evals"]) or d["evals"] <= 0:
            continue
        col = solver_color(d["solver"], d["family"])
        mk = solver_marker(d["solver"], d["family"])
        ours = is_ours(d["solver"])
        lab = d["family"] if d["family"] not in seen_fam else None
        seen_fam.add(d["family"])
        ax.errorbar(d["evals"], d["ptnr_m"], yerr=d["ptnr_s"],
                    fmt=mk, ms=(20 if ours else 9), color=col,
                    mec="black", mew=(1.4 if ours else 0.6),
                    ecolor=col, elinewidth=1.0, capsize=2.5,
                    zorder=(5 if ours else 3), label=lab)
        if ours:
            extra = (f"\n(+{d['glcp']:.0f} incr. moves)" if d["glcp"] > 0 else "")
            ax.annotate(d["solver"].replace(" (ours)", " (ours)") + extra,
                        (d["evals"], d["ptnr_m"]), fontsize=8.0, fontweight="bold",
                        color=col, xytext=(10, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Forward-model evaluations (oracle calls)  $\\leftarrow$ cheaper")
    ax.set_ylabel("Peak-to-null ratio (dB)  better $\\uparrow$")
    ax.set_title("Efficiency frontier: nulling quality per oracle call\n"
                 "(top-left = high quality at low cost)")
    ax.legend(title="family", loc="upper right", ncol=2)
    ax.grid(alpha=0.3, which="both")
    save(plt, fig, "fig6_cost_quality")


# --------------------------------------------------------------------------- #
# Figure 7 -- scaling vs N
# --------------------------------------------------------------------------- #
def run_scaling_prev(element, Ns, quick=False):
    task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
    data = {n: dict(N=[], ptnr=[], wn=[], evals=[], glcp=[]) for n in SCALE_SUBSET}
    for N in Ns:
        bud = {"Coordinate Descent": dict(sweeps=4, grid=48),
               "Riemannian CG": dict(max_iter=200), "Scaled ADMM": dict(max_iter=200),
               "Perturbation Null": dict(rounds=6), "OBH-ZKD (prior)": dict(),
               OURS_LO: dict(), OURS_HI: dict()}
        for n in SCALE_SUBSET:
            try:
                m = B.run_solver(n, task, element, N, seed=0, discrete=True, **bud[n])
                data[n]["N"].append(N); data[n]["ptnr"].append(m["ptnr"])
                data[n]["wn"].append(m["worst_null"]); data[n]["evals"].append(m["evals"])
                data[n]["glcp"].append(m["glcp_updates"])
            except Exception as e:
                sys.stderr.write(f"  ! scaling {n} N={N}: {repr(e)[:60]}\n")
        sys.stderr.write(f"  scaling: N={N} done\n")
    return data

def run_scaling(
    element,
    Ns,
    tasks,
    seeds,
    quick=False,
    discrete=True,
):
    """
    Statistical scaling experiment.

    Experimental design
    -------------------
    Deterministic solvers:
        1 run per task

    Stochastic solvers:
        len(seeds) independent runs per task

    Every solver sees exactly the same task suite at every N.

    Returns
    -------
    list[dict]
        One raw record per (solver, N, task, seed_used).
    """

    solver_budgets = budgets(quick=quick)

    records = []

    # Solver registry stores:
    # (solver_fn, family, stochastic?)
    for solver in SCALE_SUBSET:

        if solver not in B.SOLVERS:
            raise KeyError(
                f"Scaling solver {solver!r} is not present in B.SOLVERS"
            )

    total_jobs = 0

    for solver in SCALE_SUBSET:
        is_stochastic = bool(B.SOLVERS[solver][2])

        n_runs_per_task = len(seeds) if is_stochastic else 1

        total_jobs += (
            len(Ns)
            * len(tasks)
            * n_runs_per_task
        )

    completed = 0

    for N in Ns:

        print(
            f"[scaling] N={N}: "
            f"{len(tasks)} tasks, "
            f"{len(SCALE_SUBSET)} solvers"
        )

        for task_id, task in enumerate(tasks):

            for solver in SCALE_SUBSET:

                is_stochastic = bool(
                    B.SOLVERS[solver][2]
                )

                # Deterministic solvers:
                # task is the source of variability, not seed.
                #
                # Stochastic solvers:
                # use all requested seeds.
                run_seeds = (
                    seeds if is_stochastic
                    else [seeds[0]]
                )

                cfg = dict(
                    solver_budgets.get(solver, {})
                )

                for seed in run_seeds:

                    try:
                        m = B.run_solver(
                            solver,
                            task,
                            element,
                            N,
                            seed=seed,
                            discrete=discrete,
                            **cfg,
                        )

                        record = {
                            "solver": solver,
                            "N": int(N),
                            "task_id": int(task_id),
                            "seed": int(seed),
                            "stochastic": is_stochastic,

                            # Problem definition
                            "target_angle": float(
                                task.target_angles[0]
                            ),
                            "null_angles": [
                                float(x)
                                for x in task.null_angles
                            ],
                            "num_nulls": int(
                                len(task.null_angles)
                            ),

                            # Performance
                            "ptnr": float(m["ptnr"]),
                            "worst_null": float(
                                m["worst_null"]
                            ),
                            "avg_null": float(
                                m["avg_null"]
                            ),
                            "gain": float(m["gain"]),
                            "gain_loss": float(
                                m["gain_loss"]
                            ),

                            # Computational bookkeeping
                            "evals": (
                                float(m["evals"])
                                if m.get("evals") is not None
                                else np.nan
                            ),
                            "glcp_updates": float(
                                m.get("glcp_updates", 0)
                            ),

                            # Optional solver metadata
                            "family": m.get(
                                "family",
                                B.SOLVERS[solver][1],
                            ),
                        }

                        records.append(record)

                    except Exception as exc:

                        records.append({
                            "solver": solver,
                            "N": int(N),
                            "task_id": int(task_id),
                            "seed": int(seed),
                            "stochastic": is_stochastic,

                            "target_angle": float(
                                task.target_angles[0]
                            ),
                            "null_angles": [
                                float(x)
                                for x in task.null_angles
                            ],
                            "num_nulls": int(
                                len(task.null_angles)
                            ),

                            "ptnr": np.nan,
                            "worst_null": np.nan,
                            "avg_null": np.nan,
                            "gain": np.nan,
                            "gain_loss": np.nan,
                            "evals": np.nan,
                            "glcp_updates": np.nan,

                            "family": B.SOLVERS[solver][1],
                            "error": repr(exc),
                        })

                    completed += 1

                    if completed % 50 == 0:
                        print(
                            f"[scaling] "
                            f"{completed}/{total_jobs} runs complete"
                        )

    return records

def aggregate_scaling_by_task(
    records,
    metric,
):
    """
    Reduce raw scaling records to one observation per task
    for each (solver, N).

    Deterministic solvers already have one run/task.

    Stochastic solvers:
        median over seeds within each task.

    Returns
    -------
    dict:
        summary[solver][N] -> 1D array over tasks
    """

    grouped = {}

    for r in records:

        value = r.get(metric, np.nan)

        if not np.isfinite(value):
            continue

        key = (
            r["solver"],
            int(r["N"]),
            int(r["task_id"]),
        )

        grouped.setdefault(key, []).append(
            float(value)
        )

    summary = {}

    for (solver, N, task_id), values in grouped.items():

        task_value = float(
            np.median(values)
        )

        summary.setdefault(solver, {})
        summary[solver].setdefault(N, {})
        summary[solver][N][task_id] = task_value

    # Convert each N/task dictionary into an array
    # sorted by task ID.
    for solver in summary:

        for N in summary[solver]:

            task_dict = summary[solver][N]

            summary[solver][N] = np.asarray(
                [
                    task_dict[k]
                    for k in sorted(task_dict)
                ],
                dtype=float,
            )

    return summary


def bootstrap_ci(
    values,
    statistic=np.median,
    n_boot=5000,
    confidence=0.95,
    seed=12345,
):
    """
    Nonparametric bootstrap confidence interval.

    The input values must already represent one independent
    observation per task.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan, np.nan, np.nan

    center = float(statistic(x))

    if x.size == 1:
        return center, center, center

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        x.size,
        size=(n_boot, x.size),
    )

    boot = statistic(
        x[idx],
        axis=1,
    )

    alpha = 1.0 - confidence

    lo = float(
        np.quantile(boot, alpha / 2.0)
    )
    hi = float(
        np.quantile(boot, 1.0 - alpha / 2.0)
    )

    return center, lo, hi


def bootstrap_ci(
    values,
    statistic=np.median,
    n_boot=5000,
    confidence=0.95,
    seed=12345,
):
    """
    Nonparametric bootstrap confidence interval.

    The input values must already represent one independent
    observation per task.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if x.size == 0:
        return np.nan, np.nan, np.nan

    center = float(statistic(x))

    if x.size == 1:
        return center, center, center

    rng = np.random.default_rng(seed)

    idx = rng.integers(
        0,
        x.size,
        size=(n_boot, x.size),
    )

    boot = statistic(
        x[idx],
        axis=1,
    )

    alpha = 1.0 - confidence

    lo = float(
        np.quantile(boot, alpha / 2.0)
    )
    hi = float(
        np.quantile(boot, 1.0 - alpha / 2.0)
    )

    return center, lo, hi

def paired_task_differences(
    records,
    reference_solver,
    comparison_solver,
    metric="ptnr",
):
    """
    Compute paired task-level differences:
        reference - comparison.

    For stochastic comparison solvers, first take the median
    over seeds within each task.
    """

    grouped = {}

    for r in records:

        value = r.get(metric, np.nan)

        if not np.isfinite(value):
            continue

        key = (
            r["solver"],
            int(r["N"]),
            int(r["task_id"]),
        )

        grouped.setdefault(key, []).append(
            float(value)
        )

    ref = {}
    cmp = {}

    for (solver, N, task_id), values in grouped.items():

        task_value = float(
            np.median(values)
        )

        if solver == reference_solver:
            ref[(N, task_id)] = task_value

        elif solver == comparison_solver:
            cmp[(N, task_id)] = task_value

    result = {}

    for key in sorted(
        set(ref).intersection(cmp)
    ):

        N, task_id = key

        result.setdefault(N, []).append(
            ref[key] - cmp[key]
        )

    for N in result:
        result[N] = np.asarray(
            result[N],
            dtype=float,
        )

    return result

def bootstrap_scaling_exponent(
    task_summary,
    n_boot=5000,
    seed=12345,
):
    """
    Estimate empirical scaling exponent alpha in

        y(N) = a * N^alpha

    using the task-level median metric at each N.

    Returns
    -------
    dict with:
        alpha
        alpha_lo
        alpha_hi
        intercept
        r2
    """

    Ns = sorted(task_summary.keys())

    if len(Ns) < 3:
        raise ValueError(
            "At least three N values are required "
            "for empirical scaling analysis."
        )

    # Aggregate each N first.
    medians = []

    for N in Ns:

        x = np.asarray(
            task_summary[N],
            dtype=float,
        )

        x = x[np.isfinite(x)]

        medians.append(
            np.median(x)
        )

    Ns = np.asarray(Ns, dtype=float)
    medians = np.asarray(medians, dtype=float)

    positive = (
        np.isfinite(medians)
        & (medians > 0)
        & np.isfinite(Ns)
    )

    Ns = Ns[positive]
    medians = medians[positive]

    if len(Ns) < 3:
        raise ValueError(
            "Insufficient positive finite points "
            "for log-log scaling fit."
        )

    x = np.log(Ns)
    y = np.log(medians)

    slope, intercept = np.polyfit(
        x,
        y,
        1,
    )

    yhat = intercept + slope * x

    ss_res = np.sum(
        (y - yhat) ** 2
    )
    ss_tot = np.sum(
        (y - np.mean(y)) ** 2
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    # Bootstrap over Ns using the task-level distributions.
    rng = np.random.default_rng(seed)

    boot_alpha = []

    per_N_values = [
        np.asarray(
            task_summary[int(N)],
            dtype=float,
        )
        for N in Ns
    ]

    for _ in range(n_boot):

        boot_medians = []

        for values in per_N_values:

            values = values[
                np.isfinite(values)
            ]

            if len(values) == 0:
                boot_medians.append(np.nan)
            else:
                sample = rng.choice(
                    values,
                    size=len(values),
                    replace=True,
                )
                boot_medians.append(
                    np.median(sample)
                )

        boot_medians = np.asarray(
            boot_medians
        )

        if np.any(
            ~np.isfinite(boot_medians)
        ) or np.any(
            boot_medians <= 0
        ):
            continue

        alpha_b, _ = np.polyfit(
            np.log(Ns),
            np.log(boot_medians),
            1,
        )

        boot_alpha.append(
            alpha_b
        )

    boot_alpha = np.asarray(
        boot_alpha,
        dtype=float,
    )

    if len(boot_alpha):

        alpha_lo = float(
            np.quantile(
                boot_alpha,
                0.025,
            )
        )

        alpha_hi = float(
            np.quantile(
                boot_alpha,
                0.975,
            )
        )

    else:

        alpha_lo = np.nan
        alpha_hi = np.nan

    return {
        "alpha": float(slope),
        "alpha_lo": alpha_lo,
        "alpha_hi": alpha_hi,
        "intercept": float(intercept),
        "r2": float(r2),
    }

def fig_scaling_prev(plt, data):
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    for n, d in data.items():
        if not d["N"]:
            continue
        fam = B.SOLVERS[n][1]
        col = solver_color(n, fam); ours = is_ours(n)
        kw = dict(marker=solver_marker(n, fam), color=col, ms=(10 if ours else 5),
                  lw=(3.0 if ours else 1.4), mec="black" if ours else col,
                  mew=0.8 if ours else 0.4, zorder=5 if ours else 3, label=n)
        axes[0].plot(d["N"], d["ptnr"], **kw)
        ev = [e if (e is not None and e > 0) else np.nan for e in d["evals"]]
        axes[1].plot(d["N"], ev, **kw)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.set_xlabel("Array size $N$")
        ax.grid(alpha=0.3, which="both")
        ax.set_xticks(data[OURS_HI]["N"]); ax.set_xticklabels(data[OURS_HI]["N"])
    axes[0].set_ylabel("Peak-to-null ratio (dB)"); axes[0].set_title("(a) Nulling quality vs $N$")
    axes[1].set_yscale("log"); axes[1].set_ylabel("Forward-model evaluations")
    axes[1].set_title("(b) Oracle-call cost vs $N$")
    axes[1].legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Scaling with array size (cost in oracle calls, not wall-clock)",
                 y=1.02, fontsize=12.5)
    save(plt, fig, "fig7_scaling")

def fig_scaling(
    plt,
    records,
    target_ptnr_db=60.0,
):
    """
    Publication-oriented statistical scaling figure.

    (a) Worst-null PTNR vs N
    (b) Main-beam gain loss vs N
    (c) Solver-reported evaluations vs N
    (d) Success probability vs N

    Error bands are 95% task-bootstrap confidence intervals.
    """

    ptnr_summary = aggregate_scaling_by_task(
        records,
        metric="worst_null",
    )

    gain_summary = aggregate_scaling_by_task(
        records,
        metric="gain_loss",
    )

    eval_summary = aggregate_scaling_by_task(
        records,
        metric="evals",
    )

    success_summary = aggregate_success_by_task(
        records,
        target_ptnr_db=target_ptnr_db,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.2, 5.8),
        constrained_layout=True,
    )

    ax_ptnr = axes[0, 0]
    ax_gain = axes[0, 1]
    ax_eval = axes[1, 0]
    ax_success = axes[1, 1]

    all_N = sorted(
        {
            int(r["N"])
            for r in records
            if r.get("N") is not None
        }
    )

    for solver in SCALE_SUBSET:

        if solver not in ptnr_summary:
            continue

        Ns = sorted(
            set(ptnr_summary[solver])
        )

        x = np.asarray(
            Ns,
            dtype=float,
        )

        style = solver_plot_style(
            solver
        )

        # ---------------------------------------------------------
        # (a) Worst-null PTNR
        # ---------------------------------------------------------
        med = []
        lo = []
        hi = []

        for N in Ns:

            c, l, h = bootstrap_ci(
                ptnr_summary[solver][N]
            )

            med.append(c)
            lo.append(l)
            hi.append(h)

        med = np.asarray(med)
        lo = np.asarray(lo)
        hi = np.asarray(hi)

        ax_ptnr.plot(
            x,
            med,
            **style,
        )

        ax_ptnr.fill_between(
            x,
            lo,
            hi,
            alpha=0.15,
            linewidth=0,
        )

        # ---------------------------------------------------------
        # (b) Gain loss
        # ---------------------------------------------------------
        if solver in gain_summary:

            med = []
            lo = []
            hi = []

            gain_Ns = sorted(
                gain_summary[solver]
            )

            for N in gain_Ns:

                c, l, h = bootstrap_ci(
                    gain_summary[solver][N]
                )

                med.append(c)
                lo.append(l)
                hi.append(h)

            ax_gain.plot(
                gain_Ns,
                med,
                **style,
            )

            ax_gain.fill_between(
                gain_Ns,
                lo,
                hi,
                alpha=0.15,
                linewidth=0,
            )

        # ---------------------------------------------------------
        # (c) Evaluation count
        # ---------------------------------------------------------
        if solver in eval_summary:

            med = []
            lo = []
            hi = []

            eval_Ns = sorted(
                eval_summary[solver]
            )

            for N in eval_Ns:

                c, l, h = bootstrap_ci(
                    eval_summary[solver][N]
                )

                med.append(c)
                lo.append(l)
                hi.append(h)

            ax_eval.plot(
                eval_Ns,
                med,
                **style,
            )

            ax_eval.fill_between(
                eval_Ns,
                lo,
                hi,
                alpha=0.15,
                linewidth=0,
            )

        # ---------------------------------------------------------
        # (d) Success probability
        # ---------------------------------------------------------
        if solver in success_summary:

            success_Ns = sorted(
                success_summary[solver]
            )

            med = []
            lo = []
            hi = []

            for N in success_Ns:

                c, l, h = bootstrap_ci(
                    success_summary[solver][N],
                    statistic=np.mean,
                )

                med.append(c)
                lo.append(l)
                hi.append(h)

            ax_success.plot(
                success_Ns,
                med,
                **style,
            )

            ax_success.fill_between(
                success_Ns,
                lo,
                hi,
                alpha=0.15,
                linewidth=0,
            )

    # -------------------------------------------------------------
    # Axis configuration
    # -------------------------------------------------------------
    for ax in axes.flat:

        ax.set_xscale(
            "log",
            base=2,
        )

        ax.set_xticks(
            all_N
        )

        ax.set_xticklabels(
            [str(N) for N in all_N]
        )

        ax.set_xlabel(
            "Array size $N$"
        )

        ax.grid(
            alpha=0.25,
            which="both",
        )

    ax_ptnr.set_ylabel(
        "Worst-null PTNR (dB)"
    )
    ax_ptnr.set_title(
        "(a) Nulling quality"
    )

    ax_gain.set_ylabel(
        "Main-beam gain loss (dB)"
    )
    ax_gain.set_title(
        "(b) Gain preservation"
    )

    ax_eval.set_ylabel(
        "Reported solver evaluations"
    )
    ax_eval.set_yscale(
        "log"
    )
    ax_eval.set_title(
        "(c) Computational scaling"
    )

    ax_success.set_ylabel(
        f"Success probability\n"
        f"(PTNR ≥ {target_ptnr_db:.0f} dB)"
    )
    ax_success.set_ylim(
        0.0,
        1.05,
    )
    ax_success.set_title(
        "(d) Reliability"
    )

    handles, labels = (
        ax_ptnr.get_legend_handles_labels()
    )

    ax_ptnr.legend(
        handles,
        labels,
        fontsize=6.5,
        loc="best",
        frameon=True,
    )

    fig.suptitle(
        "Statistical scaling over randomized null-forming tasks",
        fontsize=10.5,
    )

    save(
        plt,
        fig,
        "fig7_scaling",
    )

    return fig

def summarize_scaling(
    records,
    target_ptnr_db=60.0,
):
    """
    Print paper-ready scaling statistics.
    """

    ptnr = aggregate_scaling_by_task(
        records,
        "worst_null",
    )

    costs = aggregate_scaling_by_task(
        records,
        "evals",
    )

    success = aggregate_scaling_by_task(
        records,
        target_ptnr_db,
    )

    print("\n=== Scaling summary ===")

    for solver in SCALE_SUBSET:

        if solver not in ptnr:
            continue

        print(f"\n{solver}")

        for N in sorted(ptnr[solver]):

            c, lo, hi = bootstrap_ci(
                ptnr[solver][N]
            )

            if solver in costs and N in costs[solver]:

                ec, elo, ehi = bootstrap_ci(
                    costs[solver][N]
                )

            else:

                ec = elo = ehi = np.nan

            if (
                solver in success
                and N in success[solver]
            ):

                sc, slo, shi = bootstrap_ci(
                    success[solver][N],
                    statistic=np.mean,
                )

            else:

                sc = slo = shi = np.nan

            print(
                f"  N={N:4d} | "
                f"PTNR={c:6.2f} "
                f"[{lo:6.2f}, {hi:6.2f}] | "
                f"evals={ec:8.1f} "
                f"[{elo:8.1f}, {ehi:8.1f}] | "
                f"success={sc:.3f} "
                f"[{slo:.3f}, {shi:.3f}]"
            )
# --------------------------------------------------------------------------- #
# Figure 8 -- smooth-analytic vs discrete-measured manifold (the decisive plot)
# --------------------------------------------------------------------------- #
def run_realism(element, tasks, N=64, quick=False):
    bud = budgets(quick)
    names = list(B.SOLVERS.keys())
    sub = tasks[:min(2, len(tasks))]
    out = []
    for name in names:
        res = {}
        for disc in (False, True):
            wn = []
            for task in sub:
                try:
                    m = B.run_solver(name, task, element, N,
                                     seed=0, discrete=disc, **bud[name])
                    wn.append(m["worst_null"])
                except Exception:
                    pass
            res[disc] = np.nanmean(wn) if wn else np.nan
        out.append(dict(solver=name, family=B.SOLVERS[name][1],
                        smooth=res[False], discrete=res[True],
                        collapse=res[True] - res[False]))
    out.sort(key=lambda r: -(r["collapse"] if not np.isnan(r["collapse"]) else -1e9))
    return out


def fig_realism(plt, realism):
    names = [r["solver"] for r in realism]
    y = np.arange(len(names))
    h = 0.4
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    sm = [r["smooth"] for r in realism]
    di = [r["discrete"] for r in realism]
    ax.barh(y + h/2, sm, height=h, color="#9ecae1", edgecolor="#3182bd",
            label="smooth analytic $c(V)$ (optimistic)")
    ax.barh(y - h/2, di, height=h, color="#fb6a4a", edgecolor="#cb181d",
            label="discrete measured grid (real device)")
    finite = [v for r in realism for v in (r["smooth"], r["discrete"]) if not np.isnan(v)]
    xlo = (min(finite) * 1.05) if finite else -160.0
    ax.set_xlim(xlo, 18)
    # collapse magnitude printed to the RIGHT of the zero axis (clear of all bars)
    for i, r in enumerate(realism):
        if not np.isnan(r["collapse"]) and r["collapse"] > 20:
            ax.text(2.0, i, f"collapse +{r['collapse']:.0f} dB", va="center", ha="left",
                    fontsize=7.0, color="#cb181d", fontweight="bold")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels([f"$\\bf{{{n.replace(' (ours)','')}}}$" if is_ours(n) else n
                        for n in names])
    ax.set_xlabel("Mean worst-null depth (dB)  -- more negative is deeper/better")
    ax.set_title("Why analytic nulls are a mirage on real hardware\n"
                 "deep nulls on the smooth model COLLAPSE on the quantized device")
    ax.legend(loc="upper left")
    ax.grid(axis="x", alpha=0.3)
    save(plt, fig, "fig8_manifold_realism")


# --------------------------------------------------------------------------- #
# Figure 9 -- representative array-factor patterns
# --------------------------------------------------------------------------- #
def _solve_weights(name, task, element, N, seed=0, discrete=True, **kw):
    fn = B.SOLVERS[name][0]
    prob = B.Problem(task, element, N, discrete=discrete)
    rng = np.random.default_rng(seed)
    V = fn(prob, rng, **kw)
    return prob.weights(V)


def fig_beam_patterns(plt, element, N=64):
    task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
    bud = budgets(False)
    show = [(OURS_HI, OURS_COLOR, "-", 2.4),
            ("Perturbation Null", FAMILY_STYLE["Hardware-aware"][0], "-", 1.3),
            ("Coordinate Descent", FAMILY_STYLE["Local"][0], "--", 1.3),
            ("Trust-Region/LM", "#9467bd", ":", 1.3)]
    u = np.linspace(-1, 1, 2001)
    n = np.arange(N)
    steer = np.exp(1j * np.pi * np.outer(u, n))     # (U, N)
    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    for name, col, ls, lw in show:
        try:
            c = _solve_weights(name, task, element, N, seed=0, discrete=True,
                               **bud.get(name, {}))
        except Exception as e:
            sys.stderr.write(f"  ! pattern {name}: {repr(e)[:50]}\n"); continue
        AF = steer @ c
        mag = 20 * np.log10(np.abs(AF) / N + 1e-12)
        ax.plot(u, mag, ls, color=col, lw=lw, label=name, zorder=4 if is_ours(name) else 2)
    for un in task.null_angles:
        ax.axvline(un, color="#cb181d", ls=":", lw=1.0, alpha=0.7)
    ax.axvline(task.target_angles[0], color="#2ca02c", ls="-", lw=1.0, alpha=0.7)
    ax.annotate("target", (task.target_angles[0], 2), color="#2ca02c", fontsize=8.5, ha="center")
    ax.annotate("nulls", (task.null_angles[0], 2), color="#cb181d", fontsize=8.5, ha="center")
    ax.set_xlabel("$u = \\sin\\theta$")
    ax.set_ylabel("Normalized array factor (dB)")
    ax.set_ylim(-80, 6)
    ax.set_title(f"Representative beam patterns ($N={N}$, measured manifold)\n"
                 "target $u=0.2$, nulls at $u=-0.4,\\,0.5$")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    save(plt, fig, "fig9_beam_patterns")


# --------------------------------------------------------------------------- #
# IO helpers
# --------------------------------------------------------------------------- #
def save(plt, fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        p = os.path.join(FIGDIR, f"{name}.{ext}")
        fig.savefig(p)
    plt.close(fig)
    print(f"  wrote {os.path.join(FIGDIR, name)}.(pdf|png)")


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  wrote {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--tasks", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="tiny fast smoke run")
    ap.add_argument("--scale-N", type=int, nargs="+", default=[16, 32, 64, 128, 256,512, 1024])
    ap.add_argument("--no-scaling", action="store_true")
    ap.add_argument("--no-realism", action="store_true")
    ap.add_argument("--no-patterns", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.tasks = min(args.tasks, 3); args.seeds = min(args.seeds, 1)
        args.scale_N = [16, 32, 64]

    plt = init_style()
    os.makedirs(FIGDIR, exist_ok=True)
    #element = SyntheticVaractor(beta=0.8, folding=True)
    element= MeasuredVaractor()
    tasks = make_tasks(
        args.tasks
    )

    save_task_suite(
        tasks,
        "experiments/scaling_task_suite.json",
    )

    seeds = list(range(args.seeds))
    names = list(B.SOLVERS.keys())
    ceiling = B.Problem(tasks[0], element, args.N).g_ceiling_db()

    print(f"[1/4] Statistical sweep: {len(names)} solvers x {len(tasks)} tasks x "
          f"up to {len(seeds)} seeds at N={args.N} (discrete measured manifold)")
    rows = run_sweep(element, args.N, tasks, seeds, quick=args.quick, discrete=True)
    summary, n_inst = aggregate(rows, names, seeds)
    order = [d["solver"] for d in summary]
    taus, prof, P = performance_profile(rows, names, seeds)

    # ---- console summary table ----
    print(f"\nGain ceiling = {ceiling:.2f} dB (0 dB = ideal coherent N).  "
          f"{n_inst} instances.")
    print(f"{'solver':22s}{'PTNR dB':>13s}{'gain dB':>13s}{'worst null':>13s}"
          f"{'oracle':>9s}{'win%':>6s}")
    for d in summary:
        print(f"{d['solver']:22s}{d['ptnr_m']:7.1f}±{d['ptnr_s']:4.1f}"
              f"{d['gain_m']:7.2f}±{d['gain_s']:4.2f}{d['wn_m']:8.1f}±{d['wn_s']:4.1f}"
              f"{d['evals']:9.0f}{d['winrate']*100:6.0f}")

    print("\n[2/4] Core figures")
    fig_distributions(plt, rows, order)
    fig_success_heatmap(plt, summary)
    fig_pareto(plt, summary)
    fig_winrate(plt, summary)
    fig_perf_profile(plt, taus, prof, P)
    fig_cost_quality(plt, summary)

    scaling = {}
    if not args.no_scaling:
        print("[3/4] Scaling sweep over N")
        
        scaling = run_scaling(
            element=element,
            Ns=args.scale_N,
            tasks=tasks,
            seeds=seeds,
            quick=args.quick,
            discrete=True,
        )

        summarize_scaling(
            scaling,
            target_ptnr_db=60.0,
        )

        fig_scaling(
            plt,
            scaling,
            target_ptnr_db=60.0,
        )

    realism = []
    if not args.no_realism:
        print("[4/4] Manifold realism contrast (smooth vs discrete)")
        realism = run_realism(element, tasks, N=args.N, quick=args.quick)
        fig_realism(plt, realism)

    if not args.no_patterns:
        print("      Beam-pattern figure")
        fig_beam_patterns(plt, element, N=max(64, args.N))

    # ---- CSV exports for the paper ----
    print("\nCSV tables")
    write_csv(os.path.join(FIGDIR, "pub_runs.csv"), rows,
              ["solver", "family", "task", "seed", "n_nulls", "ok",
               "gain", "worst_null", "avg_null", "ptnr", "gain_loss",
               "evals", "glcp_updates", "ms"])
    write_csv(os.path.join(FIGDIR, "pub_summary.csv"), summary,
              ["solver", "family", "runs", "ptnr_m", "ptnr_s", "gain_m", "gain_s",
               "wn_m", "wn_s", "loss_m", "loss_s", "evals", "evals_s", "glcp",
               "s20", "s30", "s40", "s50", "winrate"])
    write_csv(os.path.join(FIGDIR, "pub_profile.csv"),
              [dict(solver=n, **{f"tau_{int(t) if t < 1e8 else 'inf'}": prof[n][i]
               for i, t in enumerate(taus)}) for n in names],
              ["solver"] + [f"tau_{int(t) if t < 1e8 else 'inf'}" for t in taus])
    if scaling:
        srows = []
        for n, d in scaling.items():
            for i, N in enumerate(d["N"]):
                srows.append(dict(solver=n, N=N, ptnr=d["ptnr"][i],
                                  worst_null=d["wn"][i], evals=d["evals"][i],
                                  glcp_updates=d["glcp"][i]))
        write_csv(os.path.join(FIGDIR, "pub_scaling.csv"), srows,
                  ["solver", "N", "ptnr", "worst_null", "evals", "glcp_updates"])
    if realism:
        write_csv(os.path.join(FIGDIR, "pub_realism.csv"), realism,
                  ["solver", "family", "smooth", "discrete", "collapse"])

    print(f"\nDone. Figures + tables in {FIGDIR}")
    print("PDF = vector (journal); PNG = preview. (PNG/CSV are .gitignored; PDF is tracked.)")


if __name__ == "__main__":
    main()

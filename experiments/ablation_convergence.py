"""
Ablation + anytime-convergence study (Stream 4).
================================================

(A) ABLATION.  Isolate the marginal value of each component of our method on a
    statistical task set. We vary the two design axes independently:
      * gauge   : how the global phase offset psi is chosen -- by pre-refinement
                  gain ('gain', legacy) vs by post-refinement peak-to-null
                  ratio ('ptnr', ours);
      * refine  : closed-form anchor only ('none') vs anchor + Gain-Locked
                  Coordinate Polish ('glcp').
    The 2x2 grid shows that the PTNR gauge and GLCP each contribute, and that the
    full method (PTNR + GLCP) dominates.

(B) ANYTIME CONVERGENCE.  Null depth vs compute budget, measured in oracle calls
    (forward-model evaluations), not wall-clock. Each iterative baseline is run at
    a ladder of budgets to trace its (oracle-calls, worst-null) trajectory; ours is
    a single fixed-cost point. This is the "anytime" complement to the Dolan-More
    profile: it shows ours reaches a deeper null at a smaller budget than any
    baseline reaches at its largest budget.

Outputs (experiments/figures/):
  fig14_ablation.(pdf|png), fig15_convergence.(pdf|png)
  pub_ablation.csv, pub_convergence.csv

Usage: ./.venv/Scripts/python.exe experiments/ablation_convergence.py [--tasks 12]
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.api import beamform
from beamformer.controllers.metrics import get_metrics
from beamformer import baselines as B
from experiments.sota_families_benchmark import make_tasks
from experiments.publication_figures import (init_style, save, FIGDIR, OURS_COLOR,
                                             solver_color, solver_marker, is_ours)

# (label, refine, gauge, is_full)
ABLATION = [
    ("Anchor\n(gain gauge)",      "none", "gain", False),
    ("Anchor +\nPTNR gauge",      "none", "ptnr", False),
    ("Anchor + GLCP\n(gain gauge)", "glcp", "gain", False),
    ("Full:\nPTNR + GLCP",        "glcp", "ptnr", True),
]

# budget ladders for the convergence trace: (solver, kwarg-name, [levels])
CONV = [
    ("Coordinate Descent", "sweeps",  [1, 2, 3, 4, 6, 8]),
    ("Perturbation Null",  "rounds",  [1, 2, 4, 6, 8, 12]),
    ("CMA-ES",             "max_iter",[20, 50, 100, 200, 400]),
    ("Cross-Entropy",      "iters",   [10, 20, 40, 80, 120]),
    ("Differential Eq.",   "maxiter", [10, 20, 40, 80]),
]
OURS = "MR-LCMV+GLCP (ours)"
GCLP_ROUNDS= [1,2,3,4,5,6,7,8,9]


# --------------------------------------------------------------------------- #
# (A) ablation
# --------------------------------------------------------------------------- #
def run_ablation(element, tasks, N):
    data = {lbl: [] for (lbl, *_ ) in ABLATION}
    for task in tasks:
        for lbl, refine, gauge, _ in ABLATION:
            try:
                res = beamform(task, element, N=N, solver="mrlcmv", refine=refine, gauge=gauge)
                _, _, ptnr = get_metrics(res["weights"], task, element)
                data[lbl].append(ptnr)
            except Exception as e:
                sys.stderr.write(f"  ! ablation {lbl}: {repr(e)[:50]}\n")
    return data


def fig_ablation(plt, data):
    labels = [lbl for (lbl, *_ ) in ABLATION]
    fulls = [f for (*_, f) in ABLATION]
    vals = [data[l] for l in labels]
    figu, ax = plt.subplots(figsize=(8.2, 5.6))
    bp = ax.boxplot(vals, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", lw=1.3))
    rng = np.random.default_rng(0)
    for i, (v, full) in enumerate(zip(vals, fulls), start=1):
        col = OURS_COLOR if full else "#4C72B0"
        bp["boxes"][i - 1].set_facecolor(col)
        bp["boxes"][i - 1].set_alpha(0.55 if full else 0.75)
        bp["boxes"][i - 1].set_edgecolor("black"); bp["boxes"][i - 1].set_linewidth(1.4 if full else 0.8)
        v = np.asarray(v, float); v = v[np.isfinite(v)]
        ax.scatter(i + rng.uniform(-0.16, 0.16, v.size), v, s=12, color=col,
                   edgecolor="white", linewidth=0.3, zorder=3)
        ax.text(i, np.median(v) , f" {np.median(v):.0f}", fontsize=8, va="bottom")
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("Peak-to-null ratio (dB)")
    ax.set_title("Ablation: marginal value of the PTNR gauge and GLCP\n"
                 "(box over random tasks; red = full method)")
    ax.grid(axis="y", alpha=0.3)
    save(plt, figu, "fig14_ablation")


# --------------------------------------------------------------------------- #
# (B) convergence
# --------------------------------------------------------------------------- #
def run_convergence(element, tasks, N):
    traj = {}
    oet, owt=[],[]
    for round_num in GCLP_ROUNDS:
        oe, ow,ou = [], [],[]
        for task in tasks:
            m = B.run_solver(OURS, task, element, N, seed=0, discrete=True,n_dual_steps= round_num)
            oe.append(m["evals"]); ow.append(m["worst_null"]);ou.append(m["glcp_updates"])
        oet.append(np.mean(oe));owt.append(np.mean(ow))
    traj[OURS] = (oet, owt)
    print(traj)
    for name, kw, levels in CONV:
        ev, wn = [], []
        for lv in levels:
            es, ws = [], []
            for task in tasks:
                try:
                    m = B.run_solver(name, task, element, N, seed=0, discrete=True, **{kw: lv})
                    es.append(m["evals"]); ws.append(m["worst_null"])
                except Exception:
                    pass
            if es:
                ev.append(np.mean(es)); wn.append(np.mean(ws))
        traj[name] = (ev, wn)
        print(f"  convergence trace: {name}")
    # ours: single fixed-cost point (mean over tasks)
    
    return traj


def fig_convergence(plt, traj):
    figu, ax = plt.subplots(figsize=(8.4, 6.0))
    for name, (ev, wn) in traj.items():
        if not ev:
            continue
        fam = B.SOLVERS[name][1]; col = solver_color(name, fam); ours = is_ours(name)
        if ours:
            ax.scatter(ev, wn, s=420, marker="*", color=col, edgecolor="black",
                       linewidth=1.4, zorder=6, label=name)
            ax.axhline(wn[0], color=col, ls="--", lw=1.2, alpha=0.6, zorder=1)
        else:
            ax.plot(ev, wn, marker=solver_marker(name, fam), color=col, lw=1.6,
                    ms=6, label=name, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Compute budget = oracle calls (forward-model evaluations)  $\\leftarrow$ cheaper")
    ax.set_ylabel("Worst-null depth (dB)  -- deeper is better")
    ax.set_title("Anytime convergence: null depth vs oracle-call budget\n"
                 "(ours = one fixed-cost point; dashed line = its null depth)")
    ax.legend(loc="upper right", fontsize=8.0)
    ax.grid(alpha=0.3, which="both")
    save(plt, figu, "fig15_convergence")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--tasks", type=int, default=12)
    args = ap.parse_args()

    plt = init_style()
    os.makedirs(FIGDIR, exist_ok=True)
    #element = SyntheticVaractor(beta=0.8, folding=True)
    element = MeasuredVaractor()
    tasks = make_tasks(args.tasks)

    print(f"[A] Ablation ({len(tasks)} tasks, N={args.N})")
    abl = run_ablation(element, tasks, args.N)
    fig_ablation(plt, abl)
    for lbl, *_ in ABLATION:
        v = np.array(abl[lbl], float); v = v[np.isfinite(v)]
        print(f"  {lbl.replace(chr(10),' '):28s} median PTNR = {np.median(v):6.1f} dB")

    print(f"[B] Convergence trace ({len(tasks)} tasks, N={args.N})")
    traj = run_convergence(element, tasks, args.N)
    

    with open(os.path.join(FIGDIR, "pub_ablation.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["variant", "task_idx", "ptnr"])
        for lbl, *_ in ABLATION:
            for i, v in enumerate(abl[lbl]):
                w.writerow([lbl.replace("\n", " "), i, v])
    with open(os.path.join(FIGDIR, "pub_convergence.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["solver", "oracle_calls", "worst_null"])
        for name, (ev, wn) in traj.items():
            for e, n_ in zip(ev, wn):
                w.writerow([name, e, n_])
    fig_convergence(plt, traj)
    print(f"\nDone. fig14/fig15 + pub_ablation.csv + pub_convergence.csv in {FIGDIR}")


if __name__ == "__main__":
    main()

"""
Clean anytime-convergence figure for IEEE double-column.
- Ours: dual-step residual trajectory (certified path) + final star
- Baselines: budget ladder, median + IQR
- Statistical aggregation over many random tasks
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import MeasuredVaractor
from beamformer.pattern_projection import Task
from beamformer import baselines as B
from beamformer.MR_LCMV_certified import (
    beamform_mrlcmv, reset_counters, get_counters, _af,
    _dual_correction_certified, _get_manifold, solve_coherent_lcmv,
    gain_locked_polish
)
from experiments.publication_figures import init_style, save, FIGDIR, OURS_COLOR

OURS = "MR-LCMV+GLCP (certified)"

CONV_LADDERS = [
    ("Coordinate Descent", "sweeps",   [1, 2, 4, 8, 12]),
    ("Perturbation Null",  "rounds",   [1, 2, 4, 8, 12]),
    ("CMA-ES",             "max_iter", [50, 100, 200, 400]),
    ("Cross-Entropy",      "iters",    [20, 40, 80, 120]),
    ("Differential Eq.",   "maxiter",  [20, 40, 80]),
]

# ------------------------------------------------------------------
# Distinct visual language
# ------------------------------------------------------------------
FAMILY_STYLE = {
    # family            : (color, base marker)
    "Local"             : ("#1f77b4", "o"),   # blue
    "Hardware-aware"    : ("#2ca02c", "s"),   # green
    "Global"            : ("#ff7f0e", "D"),   # orange
    "Manifold"          : ("#9467bd", "^"),   # purple
    "Splitting"         : ("#8c564b", "v"),   # brown
    "Relaxation"        : ("#e377c2", "P"),   # pink
    "Quantum-insp."     : ("#7f7f7f", "X"),   # grey
    "Prior proposed"    : ("#bcbd22", "h"),   # olive
    "Ours"              : ("#d62728", "*"),   # red
}

TASK_SEED=1203

# Fine-grained marker overrides so methods inside the same family stay distinguishable
MARKER_OVERRIDE = {
    "PGD"                : "o",
    "Trust-Region/LM"    : "^",
    "Coordinate Descent" : "s",
    "Perturbation Null"  : "D",
    "CMA-ES"             : "P",
    "Cross-Entropy"      : "X",
    "Differential Eq."   : "h",
}

def style_for(name):
    """Return (color, marker) for a solver name."""
    fam = B.SOLVERS.get(name, (None, "Other", None))[1]
    color, default_m = FAMILY_STYLE.get(fam, ("#333333", "o"))
    marker = MARKER_OVERRIDE.get(name, default_m)
    return color, marker


def worst_null_db(c, task, N):
    nulls = list(getattr(task, "null_angles", []) or [])
    if not nulls:
        return -np.inf
    return max(20 * np.log10(abs(_af(c, um, N)) + 1e-12) for um in nulls)

def make_tasks(n, u_max=0.6, max_nulls=3, seed=TASK_SEED):
    """
    Generate a reproducible benchmark task suite.

    The task suite is the statistical population for deterministic solvers.
    Each task contains:
        - one target direction
        - 1..max_nulls null directions
        - minimum angular separation constraints

    The same returned task suite must be reused for every N and every solver
    in the scaling experiment.
    """
    rng = np.random.default_rng(seed)

    tasks = []

    while len(tasks) < n:
        ut = round(float(rng.uniform(-u_max, u_max)), 3)

        k = int(rng.integers(1, max_nulls + 1))

        nulls = []
        tries = 0

        while len(nulls) < k and tries < 60:
            un = round(float(rng.uniform(-0.9, 0.9)), 3)

            if (
                abs(un - ut) > 0.15
                and all(abs(un - x) > 0.1 for x in nulls)
            ):
                nulls.append(un)

            tries += 1

        if nulls:
            tasks.append(
                Task(
                    "nulled",
                    target_angles=[ut],
                    null_angles=sorted(nulls),
                )
            )

    return tasks


# ---------------------------------------------------------------------------
# Ours: true step-wise dual trajectory (one gauge, residual-driven)
# ---------------------------------------------------------------------------
def ours_dual_trajectory(task, element, N, psi=0.0, max_dual=32):
    """
    Run the certified dual at a fixed gauge and return
    (oracle_calls_after_each_step, best_null_depth_so_far).
    """
    from beamformer.MR_LCMV_certified import _dual_correction_certified, _get_manifold
    V_grid, c_grid = _get_manifold(element)
    n = np.arange(N)
    u_t = task.target_angles[0]
    s_t = np.exp(-1j * np.pi * n * u_t)
    nulls = list(task.null_angles or [])
    A = np.column_stack([np.exp(1j * np.pi * n * um) for um in nulls]) if nulls else None
    if A is not None:
        G = A.T @ np.conj(A)
        G += 1e-6 * np.trace(G).real / max(len(nulls), 1) * np.eye(len(nulls))
        Ginv = np.linalg.inv(G)
    else:
        Ginv = None

    tgt0 = np.exp(1j * psi) * s_t
    reset_counters()

    # We need the internal history – use the instrumented dual if you have it,
    # otherwise fall back to a simple loop that mirrors the certified logic.
    history = []
    mu = Ginv @ (A.T @ tgt0) if A is not None else None
    best_null = np.inf
    best_c = None

    for k in range(max_dual):
        if A is None:
            from beamformer.coherent_lcmv import _retract
            c, V, _ = _retract(tgt0, c_grid, V_grid)
            leak = 0.0
        else:
            target = tgt0 - np.conj(A) @ mu
            from beamformer.coherent_lcmv import _retract
            c, V, _ = _retract(target, c_grid, V_grid)
            leak_vec = A.T @ c
            from beamformer.coherent_lcmv import _bump_eval
            _bump_eval()
            leak = float(np.max(np.abs(leak_vec)))

        nd = 20 * np.log10(leak + 1e-15)
        if nd < best_null:
            best_null = nd
            best_c = c.copy()

        oracle, _ = get_counters()
        history.append((oracle, best_null))

        if leak <= 10 ** (-45 / 20):
            break
        if A is not None:
            mu = mu + Ginv @ leak_vec

    # final GLCP (optional, adds a few more points)
    if best_c is not None:
        V_final, c_final = gain_locked_polish(
            best_c, V, task, element, null_angles=nulls, rounds=6
        )
        nd_final = worst_null_db(c_final, task, N)
        oracle, _ = get_counters()
        if nd_final < best_null:
            history.append((oracle, nd_final))

    o = np.array([h[0] for h in history])
    n = np.array([h[1] for h in history])
    return o, n


# ---------------------------------------------------------------------------
# Statistical aggregation
# ---------------------------------------------------------------------------
def run_comparison(element, tasks, N):
    traj = {}

    # ---- Ours ----
    print("Ours (certified dual trajectory)")
    all_o, all_n = [], []
    for i, task in enumerate(tasks):
        o, n = ours_dual_trajectory(task, element, N)
        all_o.append(o)
        all_n.append(n)
        print(f"  task {i:02d}: steps={len(o)}  final null={n[-1]:.1f} dB  oracle={o[-1]}")

    # interpolate onto a common oracle grid
    max_o = max(o[-1] for o in all_o)
    grid = np.unique(np.concatenate([np.linspace(1, max_o, 30)] + all_o))
    mat = []
    for o, n in zip(all_o, all_n):
        mat.append(np.interp(grid, o, n, left=n[0], right=n[-1]))
    mat = np.asarray(mat)
    traj[OURS] = {
        "oracle": grid,
        "med": np.median(mat, axis=0),
        "q25": np.percentile(mat, 25, axis=0),
        "q75": np.percentile(mat, 75, axis=0),
    }

    # ---- Baselines ----
    for name, kw, levels in CONV_LADDERS:
        print(name)
        os_, ns_ = [], []
        for lv in levels:
            evs, nds = [], []
            for i, task in enumerate(tasks):
                try:
                    m = B.run_solver(name, task, element, N, seed=i, discrete=True, **{kw: lv})
                    evs.append(m["evals"])
                    nds.append(m["worst_null"])
                except Exception:
                    pass
            if evs:
                os_.append(np.mean(evs))
                ns_.append(nds)
        if not os_:
            continue
        ns_ = np.asarray(ns_)
        traj[name] = {
            "oracle": np.asarray(os_),
            "med": np.median(ns_, axis=1),
            "q25": np.percentile(ns_, 25, axis=1),
            "q75": np.percentile(ns_, 75, axis=1),
        }
    return traj


# ---------------------------------------------------------------------------
# Publication figure (double-column)
# ---------------------------------------------------------------------------
def plot_anytime(traj, save_prefix="fig_anytime"):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.labelsize": 8, "legend.fontsize": 6,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.35))

    # baselines first (behind)
    for name, d in traj.items():
        if name == OURS:
            continue
        fam = B.SOLVERS.get(name, (None, "Other"))[1]
        col = solver_color(name, fam) if "solver_color" in dir() else "0.5"
        ax.plot(d["oracle"], d["med"], "-o", ms=3, lw=1.0, color=col, label=name, zorder=2)
        ax.fill_between(d["oracle"], d["q25"], d["q75"], color=col, alpha=0.15, zorder=1)

    # ours on top
    d = traj[OURS]
    ax.plot(d["oracle"], d["med"], color=OURS_COLOR, lw=1.8, zorder=5, label=OURS)
    ax.fill_between(d["oracle"], d["q25"], d["q75"], color=OURS_COLOR, alpha=0.25, zorder=4)
    ax.plot(d["oracle"][-1], d["med"][-1], marker="*", ms=11,
            color=OURS_COLOR, markeredgecolor="k", markeredgewidth=0.5, zorder=6)

    ax.set_xscale("log")
    ax.set_xlim(left=5)
    ax.set_ylim(-70, 5)          # depth ≤ 0, small headroom
    ax.axhline(0, color="0.7", ls=":", lw=0.5)
    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Worst-null depth (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True, which="major", ls=":", lw=0.4)
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="0.6", framealpha=0.92, borderpad=0.3,
              handlelength=1.3, labelspacing=0.25)

    fig.tight_layout(pad=0.25)
    fig.savefig(f"{save_prefix}.pdf")
    fig.savefig(f"{save_prefix}.png")
    plt.close(fig)
    print(f"Saved {save_prefix}.pdf/.png")


def plot_publishable(traj_baselines, ours_oracle, ours_null, save_prefix="fig_anytime"):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8,
        "axes.labelsize": 8, "legend.fontsize": 5.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    })

    fig, ax = plt.subplots(figsize=(3.45, 2.55))

    # ----- baselines -----
    for name, d in traj_baselines.items():
        if name == OURS:
            continue
        col, mkr = style_for(name)
        ax.plot(d["oracle"], d["med"],
                color=col, marker=mkr, ms=4, lw=1.15,
                markeredgecolor="k", markeredgewidth=0.3,
                label=name, zorder=3)
        ax.fill_between(d["oracle"], d["q25"], d["q75"],
                        color=col, alpha=0.12, zorder=1)

    # ----- ours -----
    med_o = np.median(ours_oracle)
    med_n = np.median(ours_null)
    q25, q75 = np.percentile(ours_null, [25, 75])

    ax.axhspan(q25, q75, color=OURS_COLOR, alpha=0.15, zorder=0)
    ax.axhline(med_n, color=OURS_COLOR, ls="--", lw=0.9, alpha=0.7)

    # individual task terminations (small dots)
    ax.scatter(ours_oracle, ours_null, s=9, color=OURS_COLOR,
               alpha=0.35, edgecolor="none", zorder=4)

    # median star
    ax.plot(med_o, med_n, marker="*", ms=13, color=OURS_COLOR,
            markeredgecolor="k", markeredgewidth=0.5, zorder=5,
            label=f"{OURS} (med. {med_o:.0f})")

    ax.set_xscale("log")
    ax.set_xlim(8, 4e4)
    ax.set_ylim(-75, 15)
    ax.axhline(0, color="0.7", ls=":", lw=0.5)
    ax.set_xlabel("Oracle calls")
    ax.set_ylabel("Worst-null depth (dB)")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.grid(True, which="major", ls=":", lw=0.4)

    # ----- legend outside the axes so it never covers data -----
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=True, fancybox=False, edgecolor="0.6",
              framealpha=0.95, borderpad=0.3,
              handlelength=1.3, labelspacing=0.25,
              fontsize=5.5)

    fig.tight_layout(pad=0.3)
    # make room for the external legend
    fig.subplots_adjust(right=0.72)

    fig.savefig(f"{save_prefix}.pdf")
    fig.savefig(f"{save_prefix}.png")
    plt.close(fig)
    print(f"Saved {save_prefix}.pdf / .png")

def collect_ours_final(tasks, element, N):
    """Return arrays of (final_oracle, final_null) for every task."""
    oracles, nulls = [], []
    for i, task in enumerate(tasks):
        reset_counters()
        out = beamform_mrlcmv(
            task, element, N,
            refine="glcp", gauge="ptnr", gauge_samples=24,
            null_target_db=-45, glcp_rounds=6,
        )
        o, _ = get_counters()
        nd = worst_null_db(out["weights"], task, N)
        oracles.append(o)
        nulls.append(nd)
        print(f"  task {i:02d}: oracle={o:4d}  null={nd:6.1f} dB")
    return np.asarray(oracles), np.asarray(nulls)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    element = MeasuredVaractor()
    tasks = make_tasks(25, seed=42)
    traj = run_comparison(element, tasks, N=32)
    # 2. ours – just the final (oracle, null) of every task
    print("Ours (certified final points)")
    o_ours, n_ours = collect_ours_final(tasks, element, N=32)
    plot_anytime(traj, save_prefix=os.path.join(FIGDIR, "fig_anytime_clean"))
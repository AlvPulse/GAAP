"""
Statistical family-of-methods benchmark for phase-only nullforming on a lossy,
amplitude-phase-coupled hardware manifold.

WHY STATISTICAL: no single task is decisive -- on some geometries the bare
MR-LCMV anchor already nails the null, on others the GLCP polish (or even a
global searcher) wins. We therefore run MANY random tasks x seeds and report
distributions (mean +/- std), success probabilities, win-rates, and a
Dolan-More performance profile -- not a single cherry-picked table.

Every method optimizes the SAME objective over the SAME variable (control
voltages on the real manifold) and is scored by the SAME u-space metrics
(beamformer.controllers.metrics). One modern representative per family:

  Local            PGD, Coordinate Descent, Trust-Region / Levenberg-Marquardt
  Manifold         Riemannian Conjugate Gradient  (unit-modulus, then project)
  Splitting        Scaled ADMM (unimodular)       (then project)
  Global / DFO     Simulated Annealing, Basin Hopping, Differential Evolution,
                   CMA-ES, Cross-Entropy Method
  Quantum-insp.    Simulated Bifurcation (ballistic bSB)
  Hardware-aware   Perturbation-based nulling (measured manifold)
  Prior proposed   OBH-ZKD
  Ours             MR-LCMV  and  MR-LCMV+GLCP

Outputs (under experiments/results/):
  families_runs.csv        every (solver, task, seed) row
  families_summary.csv     per-solver aggregate
  families_profile.csv     performance-profile data
  families_*.png           plots (with --plot)

Usage:
  ./.venv/Scripts/python.exe experiments/sota_families_benchmark.py
  ./.venv/Scripts/python.exe experiments/sota_families_benchmark.py --quick
  ./.venv/Scripts/python.exe experiments/sota_families_benchmark.py --scale --plot
"""
import argparse
import csv
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer import baselines as B

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
THRESHOLDS = [-20.0, -30.0, -40.0, -50.0]   # worst-null success levels (dB)
PROFILE_DB = -30.0                            # "solved" = worst null <= this
WIN_TOL = 0.5                                 # dB tolerance for a PTNR win
TASK_SEED = 20240617                          # fixed -> reproducible task set


# --------------------------------------------------------------------------- #
# Random task generation
# --------------------------------------------------------------------------- #
def make_tasks(n, u_max=0.6, max_nulls=3):
    rng = np.random.default_rng(TASK_SEED)
    tasks = []
    while len(tasks) < n:
        ut = round(float(rng.uniform(-u_max, u_max)), 3)
        k = int(rng.integers(1, max_nulls + 1))
        nulls, tries = [], 0
        while len(nulls) < k and tries < 60:
            un = round(float(rng.uniform(-0.9, 0.9)), 3)
            if abs(un - ut) > 0.15 and all(abs(un - x) > 0.1 for x in nulls):
                nulls.append(un)
            tries += 1
        if nulls:
            tasks.append(Task("nulled", target_angles=[ut], null_angles=sorted(nulls)))
    return tasks


# --------------------------------------------------------------------------- #
# Per-solver compute budgets (kept roughly comparable; global/DFO get 1000s of
# evals so they are not handicapped vs the few-shot methods)
# --------------------------------------------------------------------------- #
def budgets(quick=False):
    b = {
        "PGD":                dict(max_iter=200),
        "Coordinate Descent": dict(sweeps=6),
        "Trust-Region/LM":    dict(max_nfev=200),
        "Riemannian CG":      dict(max_iter=250),
        "Scaled ADMM":        dict(max_iter=250),
        "Simulated Annealing":dict(max_iter=60),
        "Basin Hopping":      dict(niter=15),
        "Differential Eq.":   dict(maxiter=40),
        "CMA-ES":             dict(max_iter=150),
        "Cross-Entropy":      dict(iters=50),
        "Simulated Bifurc.":  dict(steps=300),
        "SDR (randomized)":   dict(n_rand=64),
        "Perturbation Null":  dict(rounds=8),
        "OBH-ZKD (prior)":    dict(),
        "MR-LCMV (ours)":     dict(),
        "MR-LCMV+GLCP (ours)":dict(),
    }
    if quick:
        b.update({"PGD": dict(max_iter=60), "Differential Eq.": dict(maxiter=15),
                  "CMA-ES": dict(max_iter=60), "Basin Hopping": dict(niter=5),
                  "Simulated Annealing": dict(max_iter=25), "Cross-Entropy": dict(iters=20),
                  "Riemannian CG": dict(max_iter=120), "Scaled ADMM": dict(max_iter=120),
                  "Coordinate Descent": dict(sweeps=3)})
    return b


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def run_sweep(N, tasks, seeds, quick=False, solver_subset=None, discrete=True):
    bud = budgets(quick)
    names = solver_subset or list(B.SOLVERS.keys())
    rows = []
    n_total = len(names) * len(tasks) * len(seeds)
    done = 0
    for ti, task in enumerate(tasks):
        for name in names:
            fn_seeds = seeds if B.SOLVERS[name][2] else seeds[:1]  # det. solver -> 1 run
            for seed in fn_seeds:
                t0 = time.perf_counter()
                try:
                    m = B.run_solver(name, task, element=ELEMENT, N=N, seed=seed,
                                     discrete=discrete, **bud[name])
                    ok = True
                except Exception as e:
                    m = dict(family=B.SOLVERS[name][1], gain=np.nan, worst_null=np.nan,
                             avg_null=np.nan, ptnr=np.nan, gain_loss=np.nan, evals=-1,
                             ms=(time.perf_counter() - t0) * 1e3)
                    ok = False
                    sys.stderr.write(f"  ! {name} task{ti} seed{seed}: {repr(e)[:70]}\n")
                done += 1
                rows.append(dict(solver=name, family=m["family"], task=ti, seed=seed,
                                 n_nulls=len(task.null_angles), ok=ok, **{k: m[k] for k in
                                 ("gain", "worst_null", "avg_null", "ptnr", "gain_loss", "evals", "ms")}))
        sys.stderr.write(f"  task {ti + 1}/{len(tasks)} done ({done}/{n_total} runs)\n")
    return rows


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _ms(x):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    return (np.nan, np.nan) if x.size == 0 else (float(np.mean(x)), float(np.std(x)))


def _fill_seeds(rows, names):
    """Expand deterministic solvers (run once at seed 0) across every seed so that
    per-trial win-rate / performance profiles are not structurally capped at
    (#seeds-run / #seeds-total). A deterministic solver gives the same result every
    trial, so replicating its single run across the seeds of its task is exact."""
    tasks = sorted(set(r["task"] for r in rows))
    seeds = sorted(set(r["seed"] for r in rows))
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


def aggregate(rows, names):
    # group instance results by (solver) and by (task,seed) for win-rate
    by_solver = {n: [r for r in rows if r["solver"] == n and r["ok"]] for n in names}

    # win-rate on PTNR: per (task,seed) instance, who is within WIN_TOL of best
    filled = _fill_seeds(rows, names)
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
        ev = np.array([r["evals"] for r in rs], float)
        ev = ev[ev >= 0]
        summary.append(dict(
            solver=n, family=rs[0]["family"], runs=len(rs),
            gain_m=gm, gain_s=gs, wn_m=wm, wn_s=ws, ptnr_m=pm, ptnr_s=ps,
            loss_m=lm, loss_s=ls, evals=(float(np.mean(ev)) if ev.size else 0.0),
            ms=_ms([r["ms"] for r in rs])[0],
            **{f"s{int(-th)}": succ[th] for th in THRESHOLDS},
            winrate=(wins[n] / n_inst if n_inst else 0.0)))
    summary.sort(key=lambda d: (-d["ptnr_m"]))
    return summary, n_inst


def print_summary(summary, N, n_inst, ceiling_db):
    print(f"\n{'='*120}")
    print(f"STATISTICAL FAMILIES BENCHMARK   N={N}   instances={n_inst}   "
          f"gain ceiling={ceiling_db:.2f} dB (0 dB = ideal coherent)")
    print(f"{'='*120}")
    h = (f"{'solver':22s} {'family':15s} {'gain dB':>12s} {'worst null':>13s} "
         f"{'PTNR dB':>13s} {'loss':>7s} {'s30':>5s} {'s40':>5s} {'s50':>5s} {'win%':>6s} {'ms':>7s}")
    print(h); print("-" * len(h))
    for d in summary:
        print(f"{d['solver']:22s} {d['family']:15s} "
              f"{d['gain_m']:6.2f}±{d['gain_s']:4.2f} "
              f"{d['wn_m']:7.1f}±{d['wn_s']:4.1f} "
              f"{d['ptnr_m']:7.1f}±{d['ptnr_s']:4.1f} "
              f"{d['loss_m']:6.2f} "
              f"{d['s30']*100:4.0f} {d['s40']*100:4.0f} {d['s50']*100:4.0f} "
              f"{d['winrate']*100:5.0f} {d['ms']:7.0f}")
    print("\ngain = normalized main-beam gain (mean±std), loss = ceiling-gain (lower=better)")
    print("sXX = P(worst null <= -XX dB);  win% = share of instances within "
          f"{WIN_TOL} dB of best PTNR")


def realism_contrast(N, tasks, quick=False):
    """Smooth (analytic, differentiable) vs discrete (measured grid) manifold.

    Shows which methods' nulls are a smoothness ARTIFACT: deep on the analytic
    proxy, gone on a quantized lookup table (the real device). Grid-native methods
    are invariant. One seed, a few tasks -- this is a mechanism demo, not statistics.
    """
    bud = budgets(quick)
    names = list(B.SOLVERS.keys())
    sub = tasks[:min(4, len(tasks))]
    print(f"\n{'='*96}")
    print("MANIFOLD REALISM CONTRAST  -- smooth analytic c(V)  vs  discrete measured grid")
    print(f"(mean worst-null over {len(sub)} tasks, seed 0; the discrete grid is the real device)")
    print(f"{'='*96}")
    h = (f"{'solver':22s} {'family':15s} {'smooth null':>12s} {'discr null':>12s} "
         f"{'collapse':>9s} {'smooth PTNR':>12s} {'discr PTNR':>11s}")
    print(h); print("-" * len(h))
    out = []
    for name in names:
        res = {}
        for disc in (False, True):
            wn, pt = [], []
            for task in sub:
                try:
                    m = B.run_solver(name, task, ELEMENT, N, seed=0, discrete=disc, **bud[name])
                    wn.append(m["worst_null"]); pt.append(m["ptnr"])
                except Exception:
                    pass
            res[disc] = (np.nanmean(wn) if wn else np.nan, np.nanmean(pt) if pt else np.nan)
        sm_n, sm_p = res[False]; di_n, di_p = res[True]
        out.append((name, B.SOLVERS[name][1], sm_n, di_n, di_n - sm_n, sm_p, di_p))
    # sort by collapse magnitude (largest smooth->discrete degradation first)
    out.sort(key=lambda r: -(r[4] if not np.isnan(r[4]) else -1e9))
    for name, fam, sm_n, di_n, coll, sm_p, di_p in out:
        flag = "  <-- artifact" if coll > 20 else ""
        print(f"{name:22s} {fam:15s} {sm_n:12.1f} {di_n:12.1f} {coll:+9.1f} "
              f"{sm_p:12.1f} {di_p:11.1f}{flag}")
    print("\ncollapse = discrete_null - smooth_null (large + => method relied on smoothness;")
    print("its deep null vanishes on a real quantized device). Grid-native methods ~0.")


def head_to_head(rows, a, b):
    """Per-instance comparison of two solvers (a vs b) on PTNR and gain."""
    ra = {(r["task"], r["seed"]): r for r in rows if r["solver"] == a and r["ok"]}
    rb = {(r["task"], r["seed"]): r for r in rows if r["solver"] == b and r["ok"]}
    keys = sorted(set(ra) & set(rb))
    if not keys:
        return
    aw = bw = tie = 0
    ag = bg = 0
    for k in keys:
        pa, pb = ra[k]["ptnr"], rb[k]["ptnr"]
        if pa > pb + WIN_TOL: aw += 1
        elif pb > pa + WIN_TOL: bw += 1
        else: tie += 1
        if ra[k]["gain"] > rb[k]["gain"] + 0.1: ag += 1
        elif rb[k]["gain"] > ra[k]["gain"] + 0.1: bg += 1
    print(f"\n[HEAD-TO-HEAD]  {a}  vs  {b}   ({len(keys)} instances)")
    print(f"  PTNR: {a} wins {aw}, {b} wins {bw}, ties {tie}")
    print(f"  gain: {a} higher {ag}, {b} higher {bg}")


# --------------------------------------------------------------------------- #
# Dolan-More performance profile (time-to-success)
# --------------------------------------------------------------------------- #
def performance_profile(rows, names, thresh_db=PROFILE_DB):
    """A solver 'solves' an instance if worst_null <= thresh_db; cost = wall ms.
    rho_s(tau) = fraction of instances solved by s within tau x the best solver's cost."""
    instances = {}
    for r in _fill_seeds(rows, names):
        instances.setdefault((r["task"], r["seed"]), []).append(r)
    P = len(instances)
    # best cost per instance among solvers that solved it
    best_cost = {}
    for key, inst in instances.items():
        solved = [x["ms"] for x in inst if x["worst_null"] <= thresh_db]
        best_cost[key] = min(solved) if solved else None
    taus = [1, 2, 4, 8, 16, 32, 64, 1e9]
    prof = {}
    for n in names:
        ratios = []
        for key, inst in instances.items():
            r = next((x for x in inst if x["solver"] == n), None)
            if r and r["worst_null"] <= thresh_db and best_cost[key]:
                ratios.append(r["ms"] / best_cost[key])
        prof[n] = [(float(sum(1 for rt in ratios if rt <= t)) / P) if (P and ratios) else 0.0
                   for t in taus]
    return taus, prof, P


def print_profile(taus, prof, names, P):
    print(f"\n[PERFORMANCE PROFILE]  solved = worst null <= {PROFILE_DB:.0f} dB, "
          f"cost = wall-clock ms   (P={P} instances)")
    print(f"  rho_s(tau) = P(solver solves instance within tau x fastest solver's time)")
    hdr = f"  {'solver':22s} " + " ".join(f"t<={int(t) if t<1e8 else 'inf':>5}" for t in taus)
    print(hdr)
    ranked = sorted(names, key=lambda n: -prof[n][-1])
    for n in ranked:
        if max(prof[n]) == 0:
            continue
        print(f"  {n:22s} " + " ".join(f"{v*100:5.0f}" for v in prof[n]))
    print("  (last column = overall success rate at this threshold; left = also fastest)")


# --------------------------------------------------------------------------- #
# Scaling sweep over N (cheap solvers + ours; DFO/global omitted -- dim=N blows up)
# --------------------------------------------------------------------------- #
SCALE_SUBSET = ["Coordinate Descent", "Riemannian CG", "Scaled ADMM",
                "Perturbation Null", "OBH-ZKD (prior)",
                "MR-LCMV (ours)", "MR-LCMV+GLCP (ours)"]


def run_scaling(Ns, plot=False):
    print(f"\n{'='*100}\nSCALING over N  (target u=0.2, nulls=[-0.4, 0.5])")
    print("Global/DFO families (DE/CMA/SA/BH/CEM) omitted: their dimension is N, "
          "cost grows ~ exponentially -- not competitive past N~32.")
    print(f"{'='*100}")
    task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
    data = {n: dict(N=[], gain=[], wn=[], ptnr=[], ms=[]) for n in SCALE_SUBSET}
    for N in Ns:
        # keep per-N cost bounded for the heavier local solvers
        bud = {"Coordinate Descent": dict(sweeps=4, grid=48),
               "Riemannian CG": dict(max_iter=250), "Scaled ADMM": dict(max_iter=250),
               "Perturbation Null": dict(rounds=6), "OBH-ZKD (prior)": dict(),
               "MR-LCMV (ours)": dict(), "MR-LCMV+GLCP (ours)": dict()}
        print(f"\n  N={N}")
        print(f"  {'solver':22s} {'gain':>7s} {'worst null':>11s} {'PTNR':>8s} {'ms':>8s}")
        for n in SCALE_SUBSET:
            t0 = time.perf_counter()
            try:
                m = B.run_solver(n, task, ELEMENT, N, seed=0, discrete=True, **bud[n])
                print(f"  {n:22s} {m['gain']:7.2f} {m['worst_null']:11.1f} "
                      f"{m['ptnr']:8.1f} {m['ms']:8.0f}")
                data[n]["N"].append(N); data[n]["gain"].append(m["gain"])
                data[n]["wn"].append(m["worst_null"]); data[n]["ptnr"].append(m["ptnr"])
                data[n]["ms"].append(m["ms"])
            except Exception as e:
                print(f"  {n:22s} ERROR {repr(e)[:50]}")
    if plot:
        _plot_scaling(data)
    return data


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_summary(summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    for d in summary:
        marker = "*" if "ours" in d["solver"] else "o"
        sz = 220 if "ours" in d["solver"] else 70
        ax.scatter(d["loss_m"], d["ptnr_m"], s=sz, marker=marker, zorder=3)
        ax.annotate(d["solver"], (d["loss_m"], d["ptnr_m"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("gain loss vs ceiling (dB, lower = better)")
    ax.set_ylabel("mean PTNR (dB, higher = better)")
    ax.set_title("Pareto: gain retention vs nulling (mean over instances)")
    ax.grid(alpha=0.3)
    p = os.path.join(RESULTS, "families_pareto.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


def _plot_profile(taus, prof, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    xt = [t if t < 1e8 else taus[-2] * 2 for t in taus]
    for n in names:
        if max(prof[n]) == 0:
            continue
        lw = 3 if "ours" in n else 1.3
        ax.plot(xt, [v * 100 for v in prof[n]], marker="o", lw=lw, label=n)
    ax.set_xscale("log", base=2)
    ax.set_xlabel(f"tau (x fastest solver's time)")
    ax.set_ylabel(f"% instances solved (worst null <= {PROFILE_DB:.0f} dB)")
    ax.set_title("Dolan-More performance profile (time-to-success)")
    ax.legend(fontsize=6, loc="lower right"); ax.grid(alpha=0.3)
    p = os.path.join(RESULTS, "families_profile.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


def _plot_scaling(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for n, d in data.items():
        if not d["N"]:
            continue
        lw = 3 if "ours" in n else 1.3
        axes[0].plot(d["N"], d["ptnr"], marker="o", lw=lw, label=n)
        axes[1].plot(d["N"], d["ms"], marker="o", lw=lw, label=n)
    for ax in axes:
        ax.set_xscale("log", base=2); ax.grid(alpha=0.3); ax.set_xlabel("N elements")
    axes[0].set_ylabel("PTNR dB"); axes[0].set_title("Nulling quality vs N")
    axes[1].set_yscale("log"); axes[1].set_ylabel("wall ms"); axes[1].set_title("Runtime vs N")
    axes[1].legend(fontsize=7)
    p = os.path.join(RESULTS, "families_scaling.png")
    fig.tight_layout(); fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {p}")


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #
def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    print(f"  wrote {path}")


ELEMENT = None


def main():
    global ELEMENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--tasks", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--scale", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--manifold", choices=["discrete", "smooth"], default="discrete",
                    help="discrete = measured grid (realistic, default); smooth = analytic c(V)")
    ap.add_argument("--no-contrast", action="store_true", help="skip the realism contrast")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    ELEMENT = SyntheticVaractor(beta=0.8, folding=True)
    if args.quick:
        args.tasks = min(args.tasks, 4); args.seeds = min(args.seeds, 2)

    tasks = make_tasks(args.tasks)
    seeds = list(range(args.seeds))
    names = list(B.SOLVERS.keys())
    ceiling_db = B.Problem(tasks[0], ELEMENT, args.N).g_ceiling_db()

    discrete = (args.manifold == "discrete")
    print(f"Running {len(names)} solvers x {len(tasks)} tasks x up to {len(seeds)} seeds "
          f"at N={args.N}  [manifold={args.manifold}] ...")
    t0 = time.perf_counter()
    rows = run_sweep(args.N, tasks, seeds, quick=args.quick, discrete=discrete)
    print(f"sweep done in {time.perf_counter() - t0:.1f}s")

    summary, n_inst = aggregate(rows, names)
    mode_note = ("MEASURED GRID (realistic hardware lookup table)" if discrete
                 else "SMOOTH analytic c(V) (optimistic, differentiable)")
    print(f"\nManifold model: {mode_note}")
    print_summary(summary, args.N, n_inst, ceiling_db)
    head_to_head(rows, "MR-LCMV (ours)", "MR-LCMV+GLCP (ours)")
    # also vs the strongest non-ours baseline by mean PTNR
    non_ours = [d["solver"] for d in summary if "ours" not in d["solver"]]
    if non_ours:
        head_to_head(rows, "MR-LCMV+GLCP (ours)", non_ours[0])

    taus, prof, P = performance_profile(rows, names)
    print_profile(taus, prof, names, P)

    # CSVs
    write_csv(os.path.join(RESULTS, "families_runs.csv"), rows,
              ["solver", "family", "task", "seed", "n_nulls", "ok",
               "gain", "worst_null", "avg_null", "ptnr", "gain_loss", "evals", "ms"])
    write_csv(os.path.join(RESULTS, "families_summary.csv"), summary,
              ["solver", "family", "runs", "gain_m", "gain_s", "wn_m", "wn_s",
               "ptnr_m", "ptnr_s", "loss_m", "loss_s", "evals", "ms",
               "s20", "s30", "s40", "s50", "winrate"])
    prof_rows = [dict(solver=n, **{f"tau_{int(t) if t<1e8 else 'inf'}": prof[n][i]
                 for i, t in enumerate(taus)}) for n in names]
    write_csv(os.path.join(RESULTS, "families_profile.csv"), prof_rows,
              ["solver"] + [f"tau_{int(t) if t<1e8 else 'inf'}" for t in taus])

    if args.plot:
        _plot_summary(summary)
        _plot_profile(taus, prof, names)

    if not args.no_contrast:
        realism_contrast(args.N, tasks, quick=args.quick)

    if args.scale:
        run_scaling([16, 32, 64, 128, 256], plot=args.plot)

    print("\nTakeaway: report the DISTRIBUTION, not one task. Win-rate + success "
          "probabilities show where each family actually wins.")


if __name__ == "__main__":
    main()

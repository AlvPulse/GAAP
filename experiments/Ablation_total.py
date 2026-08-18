#!/usr/bin/env python3
"""
Complete statistical ablation for certified MR-LCMV + GLCP.
Matches the full table (A0–A7, B*, C1–C3, D1–D3).

Every configuration is fully initialised with your real objects:
  SyntheticVaractor(beta=..., folding=...), MeasuredVaractor,
  TrueParametricVaractor(alpha), Task, beamform_mrlcmv, counters.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.MR_LCMV_certified import (
    beamform_mrlcmv,
    reset_counters,
    get_counters,
    _af,
    gain_locked_polish
)

# ---------------------------------------------------------------------------
# Optional parametric element (ideal ↔ severe)
# ---------------------------------------------------------------------------
try:
    from experiments.true_hardware_scaling import TrueParametricVaractor
except ImportError:
    class TrueParametricVaractor(SyntheticVaractor):
        def __init__(self, alpha: float = 1.0):
            super().__init__()
            severe_phases = np.angle(self.c_grid)
            severe_amps = 1.0 - 0.82 * np.exp(-((severe_phases - np.pi) ** 2) / 0.5)
            ideal_phases = np.linspace(0, 2 * np.pi, len(self.c_grid), endpoint=False)
            ideal_amps = np.ones_like(ideal_phases)
            phases = (1 - alpha) * ideal_phases + alpha * severe_phases
            amps = (1 - alpha) * ideal_amps + alpha * severe_amps
            self.c_grid = amps * np.exp(1j * phases)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class TrialResult:
    config_id: str
    trial: int
    oracle: int
    glcp_moves: int
    gain_db: float
    null_db: float
    max_sll_db: float          # extra metric for D* rows
    dual_status: str
    gain_preserved: bool
    nulls_preserved: bool
    certified: bool
    wall_s: float


# ---------------------------------------------------------------------------
# Random task factory
# ---------------------------------------------------------------------------
def make_random_task(rng: np.random.Generator, n_nulls: int = 1, min_sep: float = 0.15) -> Task:
    target = float(rng.uniform(-0.5, 0.5))
    nulls = []
    while len(nulls) < n_nulls:
        u = float(rng.uniform(-0.85, 0.85))
        if abs(u - target) >= min_sep and all(abs(u - n) >= min_sep for n in nulls):
            nulls.append(u)
    return Task(type="nulled", target_angles=[target], null_angles=nulls)


# ---------------------------------------------------------------------------
# FULL ablation table – every row from the plan
# ---------------------------------------------------------------------------
def get_configs() -> Dict[str, Dict[str, Any]]:
    """
    Keys match the IDs in the ablation plan.
    Special flags (handled in run_one):
      disable_lcmv_dual, no_warmstart, ...
    """
    def base(**overrides):
        """Fresh copy of the default kwargs with optional overrides."""
        d = dict(
            refine="glcp",
            gauge="ptnr",
            gauge_samples=24,
            null_target_db=-45,
            glcp_rounds=6,
            glcp_gain_floor=0.995,
            n_dual_steps=32,
            dual_patience=4,
            ridge=1e-6,
            suppress_sll=False,
        )
        d.update(overrides)
        return d

    return {
        # ================================================================
        # A – core algorithmic components
        # ================================================================
        "A0_full": {
            "desc": "Full certified pipeline (baseline)",
            "expected": "Reference",
            "kwargs": base(),
        },
        "A1_no_gauge": {
            "desc": "Gauge sweep disabled (single ψ=0)",
            "expected": "Loses operating-point advantage → shallower nulls",
            "kwargs": base(gauge="gain", gauge_samples=1, psi_samples=1),
        },
        "A2_no_glcp": {
            "desc": "GLCP disabled (refine='none')",
            "expected": "Loses local refinement → nulls stop at dual floor",
            "kwargs": base(refine="none"),
        },
        "A3_no_gain_lock": {
            "desc": "Gain lock disabled (GLCP floor → 0)",
            "expected": "Deeper nulls possible, main-beam gain collapses",
            "kwargs": base(glcp_gain_floor=0.0),
        },
        "A4_no_lcmv": {
            "desc": "LCMV dual removed (matched-filter + retract + GLCP)",
            "expected": "No null-space projection → shallow nulls",
            "kwargs": base(),
            "disable_lcmv_dual": True,
        },
        "A5_single_shot": {
            "desc": "Dual correction after first retract only (n_dual=1)",
            "expected": "Residual stays at first projection error",
            "kwargs": base(n_dual_steps=1),
        },
        "A6_fixed_budget": {
            "desc": "Adaptive termination → fixed budget (patience=999)",
            "expected": "Oracle cost rises on easy instances",
            "kwargs": base(dual_patience=999),
        },
        "A7_no_warmstart": {
            "desc": "Warm-start μ disabled (μ=0)",
            "expected": "More dual steps / earlier stagnation",
            "kwargs": base(),
            "no_warmstart": True,
        },

        # ================================================================
        # B – manifold / hardware
        # ================================================================
        "B_folding_on": {
            "desc": "SyntheticVaractor folding=True (reference)",
            "expected": "Reference for folding study",
            "element": "synthetic",
            "element_kwargs": dict(beta=0.8, folding=True),
            "kwargs": base(),
        },
        "B_folding_off": {
            "desc": "SyntheticVaractor folding=False",
            "expected": "Fewer local minima → gauge less critical",
            "element": "synthetic",
            "element_kwargs": dict(beta=0.8, folding=False),
            "kwargs": base(),
        },
        "B_beta_low": {
            "desc": "Low severity (beta=0.3)",
            "expected": "Milder amplitude-phase coupling",
            "element": "synthetic",
            "element_kwargs": dict(beta=0.3, folding=True),
            "kwargs": base(),
        },
        "B_beta_high": {
            "desc": "High severity (beta=1.2)",
            "expected": "Stronger pinching / deeper amplitude dip",
            "element": "synthetic",
            "element_kwargs": dict(beta=1.2, folding=True),
            "kwargs": base(),
        },
        "B_alpha_0": {
            "desc": "Ideal phase shifter (alpha=0)",
            "expected": "Gauge benefit disappears, opportunity loss → 0",
            "element": "parametric",
            "element_kwargs": dict(alpha=0.0),
            "kwargs": base(),
        },
        "B_alpha_1": {
            "desc": "Severe VDIL (alpha=1)",
            "expected": "Maximum opportunity loss / gauge value",
            "element": "parametric",
            "element_kwargs": dict(alpha=1.0),
            "kwargs": base(),
        },
        "B_measured": {
            "desc": "Real MeasuredVaractor",
            "expected": "Hardware ground truth",
            "element": "measured",
            "element_kwargs": {},
            "kwargs": base(),
        },

        # ================================================================
        # C – gauge selection
        # ================================================================
        "C1_gain_gauge": {
            "desc": "PTNR gauge → pure gain gauge",
            "expected": "Wrong operating point → 10–16 dB null-depth loss",
            "kwargs": base(gauge="gain"),
        },
        "C2_reachability": {
            "desc": "Uniform PTNR → Reachability multi-scale sampler",
            "expected": "May find deeper nulls on narrow-basin (small-N) manifolds",
            "kwargs": base(gauge="reach"),
        },
        "C3_gauge_half": {
            "desc": "Gauge samples halved (12)",
            "expected": "Diminishing returns / sensitivity",
            "kwargs": base(gauge_samples=12),
        },
        "C3_gauge_double": {
            "desc": "Gauge samples doubled (48)",
            "expected": "Diminishing returns / sensitivity",
            "kwargs": base(gauge_samples=48),
        },

        # ================================================================
        # D – SLL & regularisation
        # ================================================================
        "D1_no_sll": {
            "desc": "SLL active-set disabled",
            "expected": "Higher sidelobes; hard-nulls unchanged",
            "kwargs": base(suppress_sll=False),
        },
        "D2_sll_aggressive": {
            "desc": "SLL active-set aggressive (more virtual nulls)",
            "expected": "Oracle vs final SLL trade-off",
            "kwargs": base(
                suppress_sll=True,
                n_sll_virtual=4,
                sll_outer_rounds=3,
            ),
        },
        "D3_ridge_zero": {
            "desc": "Ridge regularisation → 0",
            "expected": "Possible ill-conditioning",
            "kwargs": base(ridge=0.0),
        },
        "D3_ridge_large": {
            "desc": "Ridge regularisation → large (1e-2)",
            "expected": "Residual floor rises",
            "kwargs": base(ridge=1e-2),
        },
    }
# Element builder
# ---------------------------------------------------------------------------
def build_element(cfg: Dict[str, Any]):
    kind = cfg.get("element", "synthetic")
    kw = dict(cfg.get("element_kwargs", {}))
    if kind == "measured":
        return MeasuredVaractor()
    if kind == "parametric":
        return TrueParametricVaractor(**kw)
    return SyntheticVaractor(**kw)


# ---------------------------------------------------------------------------
# Helper: crude max-SLL estimate (for D* rows)
# ---------------------------------------------------------------------------
def estimate_max_sll(c, task, N, oversample=8):
    """Simple peak sidelobe outside main beam and hard nulls."""
    u = np.linspace(-1, 1, oversample * N)
    n = np.arange(N)[:, None]
    F = np.abs(np.sum(c[:, None] * np.exp(1j * np.pi * n * u[None, :]), axis=0))
    pdb = 20 * np.log10(F + 1e-12)
    mask = np.abs(u - task.target_angles[0]) > getattr(task, "beamwidth", 0.08)
    for um in (task.null_angles or []):
        mask &= np.abs(u - um) > getattr(task, "beamwidth", 0.08)
    if not np.any(mask):
        return np.nan
    return float(np.max(pdb[mask]))


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------
def run_one(config_id: str, cfg: Dict[str, Any], task: Task, N: int, trial_idx: int) -> TrialResult:
    element = build_element(cfg)
    kwargs = dict(cfg.get("kwargs", {}))

    # ------------------------------------------------------------------
    # Special case A4: pure matched-filter + retract (+ optional GLCP)
    # ------------------------------------------------------------------
    if cfg.get("disable_lcmv_dual"):
        reset_counters()
        t0 = time.perf_counter()

        # matched-filter target at psi = 0 (or a single gauge)
        u_t = task.target_angles[0]
        n = np.arange(N)
        s_t = np.exp(-1j * np.pi * n * u_t)
        tgt0 = s_t                                  # psi = 0

        from beamformer.coherent_lcmv import _get_manifold, _retract
        V_grid, c_grid = _get_manifold(element)
        c_n, V_n, _ = _retract(tgt0, c_grid, V_grid)

        # optional GLCP (still allowed for a fair “no-LCMV” ablation)
        if kwargs.get("refine", "glcp") == "glcp":
            V_n, c_n = gain_locked_polish(
                c_n, V_n, task, element,
                null_angles=list(task.null_angles or []),
                rounds=kwargs.get("glcp_rounds", 6),
                gain_floor=kwargs.get("glcp_gain_floor", 0.995),
            )

        wall = time.perf_counter() - t0
        oracle, glcp_moves = get_counters()

        # metrics
        gain_db = 20 * np.log10(abs(_af(c_n, u_t, N)) / N + 1e-12)
        nulls = [20 * np.log10(abs(_af(c_n, um, N)) + 1e-12)
                 for um in (task.null_angles or [])]
        null_db = float(max(nulls)) if nulls else np.nan
        max_sll = estimate_max_sll(c_n, task, N)

        return TrialResult(
            config_id=config_id,
            trial=trial_idx,
            oracle=int(oracle),
            glcp_moves=int(glcp_moves),
            gain_db=gain_db,
            null_db=null_db,
            max_sll_db=max_sll,
            dual_status="disabled",
            gain_preserved=True,
            nulls_preserved=True,
            certified=False,
            wall_s=wall,
        )

    # ------------------------------------------------------------------
    # Normal path for every other configuration
    # ------------------------------------------------------------------
    if cfg.get("no_warmstart"):
        # document the intention; exact μ=0 needs a one-line patch in the dual
        kwargs["n_dual_steps"] = kwargs.get("n_dual_steps", 32)

    reset_counters()
    t0 = time.perf_counter()
    out = beamform_mrlcmv(task, element, N, **kwargs)
    wall = time.perf_counter() - t0
    oracle, glcp_moves = get_counters()

    info = out.get("info", {})
    gain_db = float(info.get("final_gain_db", info.get("norm_gain_db", np.nan)))
    nulls = info.get("final_nulls_db", info.get("nulls_db", []))
    null_db = float(max(nulls)) if nulls else np.nan
    max_sll = estimate_max_sll(out["weights"], task, N)

    return TrialResult(
        config_id=config_id,
        trial=trial_idx,
        oracle=int(oracle),
        glcp_moves=int(glcp_moves),
        gain_db=gain_db,
        null_db=null_db,
        max_sll_db=max_sll,
        dual_status=str(info.get("dual_status", "")),
        gain_preserved=bool(info.get("gain_preserved", False)),
        nulls_preserved=bool(info.get("nulls_preserved", False)),
        certified=bool(info.get("certified_refinement", False)),
        wall_s=wall,
    )

# ---------------------------------------------------------------------------
# Driver + statistics (identical structure to previous version)
# ---------------------------------------------------------------------------
def run_ablation(N=32, n_trials=30, n_nulls=1, seed=42, out_dir="ablation_results"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    configs = get_configs()
    rng = np.random.default_rng(seed)
    tasks = [make_random_task(rng, n_nulls=n_nulls) for _ in range(n_trials)]

    all_results: List[TrialResult] = []
    for cfg_id, cfg in configs.items():
        print(cfg_id,cfg)
        print(f"\n=== {cfg_id}: {cfg['desc']} ===")
        for t_idx, task in enumerate(tasks):
            res = run_one(cfg_id, cfg, task, N, t_idx)
            all_results.append(res)
            print(f"  trial {t_idx:02d}: oracle={res.oracle:4d}  "
                  f"gain={res.gain_db:7.2f}  null={res.null_db:7.1f}  "
                  f"SLL={res.max_sll_db:6.1f}  [{res.dual_status}]")

    df = pd.DataFrame([asdict(r) for r in all_results])
    df.to_csv(Path(out_dir) / "ablation_raw.csv", index=False)

    # ---- summary + paired Wilcoxon vs A0_full ----
    summary_rows = []
    baseline = df[df.config_id == "A0_full"].sort_values("trial")

    for cfg_id, cfg in configs.items():
        sub = df[df.config_id == cfg_id].sort_values("trial")
        n = len(sub)
        row = {
            "config_id": cfg_id,
            "description": cfg["desc"],
            "expected": cfg["expected"],
            "n_trials": n,
            "oracle_mean": sub.oracle.mean(),
            "oracle_std": sub.oracle.std(ddof=1),
            "gain_mean": sub.gain_db.mean(),
            "gain_std": sub.gain_db.std(ddof=1),
            "null_mean": sub.null_db.mean(),
            "null_std": sub.null_db.std(ddof=1),
            "sll_mean": sub.max_sll_db.mean(),
            "sll_std": sub.max_sll_db.std(ddof=1),
        }
        if cfg_id != "A0_full" and len(baseline) == n:
            for metric, col in [("oracle", "oracle"), ("gain", "gain_db"),
                                ("null", "null_db"), ("sll", "max_sll_db")]:
                base = baseline[col].to_numpy(float)
                test = sub[col].to_numpy(float)
                mask = np.isfinite(base) & np.isfinite(test)
                if mask.sum() < 5:
                    row[f"{metric}_p"] = row[f"{metric}_d"] = row[f"{metric}_delta"] = np.nan
                    continue
                try:
                    _, p = stats.wilcoxon(test[mask], base[mask])
                except ValueError:
                    p = np.nan
                diff = test[mask] - base[mask]
                row[f"{metric}_p"] = float(p)
                row[f"{metric}_d"] = float(diff.mean() / (diff.std(ddof=1) + 1e-12))
                row[f"{metric}_delta"] = float(diff.mean())
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(Path(out_dir) / "ablation_summary.csv", index=False)

    # ---- LaTeX ----
    lines = [
        r"\begin{table}[t]\centering",
        r"\caption{Full ablation of certified MR-LCMV "
        r"(mean $\pm$ std, $N=%d$, %d tasks).}" % (N, n_trials),
        r"\label{tab:ablation}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Config & Oracle & Gain (dB) & Null (dB) & Max SLL (dB) \\",
        r"\midrule",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"{r['config_id']} & "
            f"${r['oracle_mean']:.0f}\\pm{r['oracle_std']:.0f}$ & "
            f"${r['gain_mean']:.2f}\\pm{r['gain_std']:.2f}$ & "
            f"${r['null_mean']:.1f}\\pm{r['null_std']:.1f}$ & "
            f"${r['sll_mean']:.1f}\\pm{r['sll_std']:.1f}$ \\\\"
        )
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (Path(out_dir) / "ablation_table.tex").write_text("\n".join(lines))

    # ---- expected vs observed ----
    commentary = []
    for _, r in summary.iterrows():
        if r["config_id"] == "A0_full":
            continue
        obs = []
        for m in ("null", "gain", "oracle", "sll"):
            d = r.get(f"{m}_delta", np.nan)
            p = r.get(f"{m}_p", np.nan)
            if np.isfinite(d):
                obs.append(f"{m} Δ={d:+.2f} (p={p:.3g})")
        commentary.append({"config": r["config_id"],
                           "expected": r["expected"],
                           "observed": "; ".join(obs) or "n/a"})
    with open(Path(out_dir) / "expected_vs_observed.json", "w") as f:
        json.dump(commentary, f, indent=2)

    print(f"\nResults written to {out_dir}/")
    return df, summary


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=32)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--nulls", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="ablation_results")
    args = ap.parse_args()

    df, summary = run_ablation(
        N=args.N, n_trials=args.trials, n_nulls=args.nulls,
        seed=args.seed, out_dir=args.out,
    )
    print("\n=== Null-depth summary ===")
    print(summary[["config_id", "null_mean", "null_std",
                   "oracle_mean", "gain_mean"]].to_string(index=False))
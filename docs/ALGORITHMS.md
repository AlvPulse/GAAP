# Algorithms: Gain-First Phase-Only Nullforming on a Lossy, Coupled Hardware Manifold

This document specifies, with enough rigor for a journal submission, (i) the
problem we solve, (ii) the hardware model that makes it hard, (iii) the shared,
fair benchmark protocol against which every method is judged, (iv) the metrics
and the machine-independent **cost model**, (v) each baseline family, and
(vi) **our method — MR-LCMV with Gain-Locked Coordinate Polish (GLCP)** — in
full. It closes with the *discrete-manifold realism* argument that decides which
methods are credible on real hardware, and a map from each figure to the claim
it supports.

All figures referenced below are produced by
[`experiments/publication_figures.py`](../experiments/publication_figures.py)
into `experiments/figures/` (vector `.pdf` for the manuscript, `.png` for
preview). The benchmark harness is
[`beamformer/baselines.py`](../beamformer/baselines.py); our solver is
[`beamformer/coherent_lcmv.py`](../beamformer/coherent_lcmv.py).

---

## 1. Problem statement

We have a uniform linear array of `N` phase-only elements. Each element `n`
applies a complex weight `c_n` drawn from a **per-element hardware manifold**
`M_n = { c_n(V) : V ∈ [V_min, V_max] }`, where `V` is a control voltage. The
array factor in direction `u = sin θ` is

```
        AF(u) = Σ_{n=0}^{N-1}  c_n · exp(+jπ n u).
```

Given one **target** direction `u_t` and a set of **null** directions
`{u_1, …, u_M}`, we want the weights that maximize coherent main-beam gain while
forcing the array factor to (near) zero at every null:

```
        maximize    |AF(u_t)|^2                      (coherent gain)
        subject to  AF(u_m) = 0,   m = 1 … M         (hard nulls)
                    c_n ∈ M_n,     n = 0 … N-1        (hardware manifold)
```

The difficulty is the last constraint. `M_n` is **not** the unit circle: it is a
measured, lossy, amplitude–phase-**coupled**, non-monotonic ("branch-folding")
curve, and on real hardware it is a **finite lookup table**, not a smooth
analytic function. You cannot independently set amplitude and phase, and you
cannot finite-difference below the voltage quantization step.

---

## 2. The hardware model

Elements are modeled by `SyntheticVaractor` (a development proxy for measured
`.mat` varactor data loaded by `MeasuredVaractor`), see
[`beamformer/element_model.py`](../beamformer/element_model.py). For a normalized
voltage `v ∈ [0,1]`:

```
        φ(v)  = 2π · 1.05 · v  +  0.8 · sin(2π · 1.2 · v)        # non-monotonic phase (folding)
        A(v)  = 1 − 0.8β · exp(−((φ−π)/π)^2) − 0.6β · cos(φ)     # amplitude coupled to phase
        A(v)  = clip(A(v), 0.05, 1.0)
        c(v)  = A(v) · exp(jφ(v))
```

Two consequences are central to the paper:

1. **Insertion-loss gain ceiling.** Because `|c| ≤ 1` and the high-amplitude
   region is narrow, the best achievable normalized main-beam gain is bounded
   well below the lossless ideal. For the default element
   (`β = 0.8`, folding on) the ceiling is **≈ −3.8 dB** relative to the ideal
   coherent sum `N` (0 dB). This is *physics*, not an algorithmic artifact; all
   gain numbers are reported relative to this ceiling
   (`g_ceiling_db()` in `baselines.Problem`). A method that reports near-0 dB
   gain on this device is buggy or cheating.

2. **Discreteness.** The real device exposes a finite grid `V_grid` (here 500
   samples). The realistic evaluation snaps any requested `V` to the grid before
   computing weights (`Problem(discrete=True)`, the default). Methods that need a
   differentiable `c(V)` to dig their nulls are operating on a model they will
   never have from measurements. This is the crux of §8.

---

## 3. Fair-comparison protocol

The contribution is a *solver*, so the comparison must isolate the optimizer’s
ability to exploit the hardware manifold — not differences in objective,
variable, or scoring. Every method in
[`baselines.py`](../beamformer/baselines.py) therefore:

- optimizes the **same decision variable** — control voltages `V ∈ R^N` on the
  real manifold (manifold-design methods that work in ideal-phase space are
  explicitly **projected** onto `M_n` afterward, see §6);
- minimizes the **same penalized scalar objective**

  ```
        J(V) = (1/M) Σ_m |AF(u_m)|^2 / N^2  +  ρ · max(0, g_floor − |AF(u_t)|/N)^2,
  ```

  i.e. *minimize null power subject to a soft floor on coherent gain*
  (`g_floor = 0.85 · g_ceiling`, `ρ = 50`); and

- is scored by the **same u-space metrics** computed from exact steering vectors
  (`beamformer/controllers/metrics.py`), never from an FFT grid that can snap to
  fake zeros.

Tasks are random (`make_tasks`): a random target `u_t ∈ [−0.6, 0.6]` and 1–3
random nulls, with a fixed task seed for reproducibility. Stochastic solvers are
run over multiple RNG seeds; deterministic solvers (ours) are run once and
replicated across seeds only for per-instance win-rate accounting
(`_fill_seeds`, so their win-rate is not structurally undercounted).

---

## 4. Metrics and the cost model

**Quality metrics** (per task, from exact steering vectors):

| Symbol | Name | Definition | Better |
|---|---|---|---|
| `g` | normalized gain (dB) | `20·log10(|AF(u_t)|) − 20·log10(N)` | higher (ceiling ≈ −3.8 dB) |
| `loss` | gain loss (dB) | `g_ceiling_db − g` | lower (→ 0) |
| `wn` | worst null (dB) | `max_m 20·log10(|AF(u_m)|)` | lower (deeper) |
| `PTNR` | peak-to-null ratio (dB) | `g − wn` | higher |

PTNR is the headline scalar: it rewards a method **only** if it keeps the beam
*and* digs the null — exactly the trade-off the legacy AP backbone cannot make.

**Cost metric — why not wall-clock.** Wall-clock seconds are not publishable:
they depend on the machine, the BLAS/threading build, vectorization luck, and
Python overhead, and they are not reproducible across reviewers. We instead use
the standard machine-independent currency of derivative-free / numerical
optimization:

> **Oracle calls** = number of **forward-model evaluations**, i.e. how many times
> a method evaluates the array factor of a *complete* candidate weight vector at
> the constraint directions. One oracle call ≈ one evaluation of `J(V)`.

This is exactly the cost axis of a Dolan–Moré performance profile. Every solver
counts it via `Problem.eval_count`. Finite-difference gradients are charged
honestly: one PGD step that probes `N` coordinates costs `N+1` oracle calls.

For our closed-form method the oracle calls are the gauge-sweep evaluations
(dual-correction leakage checks + gauge scoring). The Gain-Locked Coordinate
Polish (GLCP, §7.4) updates the manifold grid by **O(M) rank-1 incremental
updates** and performs **no full array-factor evaluation**, so its committed
coordinate moves are reported as a **separate** quantity (`glcp_updates`) rather
than folded into the oracle count — neither inflating nor flattering our cost by
omission. Counters live in `coherent_lcmv.py` (`reset_counters` / `get_counters`).

---

## 5. What "iterations" means here

A reviewer asking for "iterations rather than wall-clock" wants a cost that is
intrinsic to the algorithm. *Outer iterations* are not comparable across families
(one CMA-ES generation = `popsize` evaluations; one coordinate-descent sweep =
`N·G` evaluations; one PGD step = `N+1`). The **forward-model evaluation count**
is the common denominator that makes them comparable, and it is what every plot
in this study uses on its cost axis. Representative counts at `N = 32` (one task):

| Method | oracle calls | worst null (dB) |
|---|---:|---:|
| **MR-LCMV+GLCP (ours)** | **240** (+635 incremental moves) | **−59** |
| Perturbation nulling | 1 024 | −42 |
| Coordinate Descent | 6 240 | −38 |
| CMA-ES | 840 | −8 |
| Differential Evolution | 8 225 | +14 (fails) |

Ours reaches the deepest null at the **lowest** oracle-call cost — that is the
performance claim, expressed rigorously.

---

## 6. Baseline families (one modern representative each)

All are implemented in [`baselines.py`](../beamformer/baselines.py) and tuned to
roughly comparable budgets (`budgets()` in
[`sota_families_benchmark.py`](../experiments/sota_families_benchmark.py)).

- **Local — Projected Gradient Descent (PGD).** Finite-difference gradient of
  `J(V)` with projection onto box bounds. Pays `N+1` oracle calls per step.
- **Local — Coordinate Descent.** Exhaustive per-coordinate grid search over `V`,
  swept to convergence. Grid-native (robust to discreteness) but expensive
  (`sweeps·N·G` oracle calls).
- **Local — Trust-Region / Levenberg–Marquardt.** Least-squares (`scipy`
  `trf`) on the residual form of `J`. *Needs a smooth Jacobian* — see §8.
- **Manifold — Riemannian Conjugate Gradient.** Minimizes the *same penalized
  objective* over ideal unit-modulus phases (analytic Riemannian gradient,
  Polak–Ribière+, backtracking line search), **then projects** onto `M_n`.
  Cannot exploit amplitude–phase coupling by construction.
- **Splitting — Scaled ADMM (unimodular).** Quadratic `w`-update + unimodular
  projection + dual ascent, with a matched-filter gain anchor (without it the
  unimodular minimizer of pure null power collapses the beam), **then projects**.
- **Global / DFO — Simulated Annealing, Basin Hopping, Differential Evolution,
  CMA-ES, Cross-Entropy.** Black-box search over `V ∈ R^N`; thousands of oracle
  calls. Dimension *is* `N`, so they do not scale (see Fig. 7).
- **Quantum-inspired — Ballistic Simulated Bifurcation (bSB).** Adiabatic
  bifurcation dynamics over phase offsets, then projects.
- **Relaxation — Semidefinite Relaxation (SDR) with Gaussian randomization.** The
  unimodular nulling QCQP relaxed to an SDP; its rank-1 solution is the MVDR/LCMV
  continuous optimum (projected matched filter), recovered to a constant-modulus
  vector by Gaussian randomization, **then projected**. A standard strong baseline
  for constant-modulus array problems; like the other design-then-project methods
  it cannot exploit amplitude–phase coupling.
- **Hardware-aware — Perturbation nulling.** Sequential small-perturbation grid
  search on the measured manifold with a gain guard. Grid-native and the
  strongest non-trivial baseline here.
- **Prior proposed — OBH-ZKD.** Orbital basin hopping + zero-knowledge coordinate
  descent (a representative prior heuristic).

**Canonical-name mapping (for the related-work section).** The family
representatives above correspond to recognized methods in the literature; attach
your specific citations to: Alternating Projection / Gerchberg–Saxton (the legacy
`ap` backbone), Projected Gradient Descent, block/cyclic Coordinate Descent,
Levenberg–Marquardt / trust-region least squares, Riemannian Conjugate Gradient on
the complex-circle (oblique) manifold, ADMM for unimodular beamforming, Simulated
Annealing, Basin Hopping, Differential Evolution, CMA-ES, the Cross-Entropy Method,
ballistic Simulated Bifurcation, Semidefinite Relaxation with Gaussian
randomization, and phase-only perturbation nulling. The contribution is **not** a
new member of one of these families but a gain-first, grid-native solver
(§7) that the discreteness theory (`THEORY.md`) shows is the right tool on a
*measured* manifold.

---

## 7. Our method: MR-LCMV + GLCP

**Manifold-Retracted LCMV with Dual Correction**, followed by **Gain-Locked
Coordinate Polish**. The design principle is *gain-first*: unlike the
Alternating-Projection (AP) backbone — which minimizes a pattern-matching
residual and lets main-beam gain become collateral damage (normalized gain as
low as −20 dB) — every stage of our method holds the coherent beam explicitly.

Conventions (matching the metrics module): steering vector
`a(u)_n = exp(+jπ n u)`; matched ("conjugate") weight
`s(u)_n = exp(−jπ n u)`, so `c = s(u_t) ⇒ AF(u_t) = N`.

### 7.1 Closed-form LCMV anchor

For a *continuous* (unconstrained-modulus) array, the maximum-gain weight that
perfectly nulls is the matched filter projected off the null subspace:

```
        A   = [ a(u_1), …, a(u_M) ]            (N × M)
        P   = I − conj(A) (Aᵀ conj(A))^{-1} Aᵀ  (orthogonal projector off the null subspace)
        c0  = P ( e^{jψ} s_t )
```

This is the optimal dual of the continuous problem and gives a perfect-null,
near-max-gain starting point in **one shot**. `ψ` is a global phase gauge (§7.3).

### 7.2 Manifold retraction

Each element is snapped onto its manifold independently by a **linear**
inner-product maximization — which is the *global* per-element optimum for a
fixed target alignment, and crucially uses the **full-amplitude matched field**
as the target, never a shrunken IFFT pattern, so it preserves coherent gain:

```
        c_n = argmax_{c ∈ M_n}  Re( c · conj(target_n) ).
```

Vectorized over the grid this is a single `argmax`; it is grid-native and thus
**invariant** to smooth-vs-discrete (§8).

### 7.3 Gauss–Newton dual correction + PTNR gauge

Retraction perturbs the nulls. We repair them with a few steps of the same
`M × M` solve, accounting for null–null coupling, while the gain term in the
target keeps the beam locked:

```
        target = e^{jψ} s_t − conj(A) · μ
        c      = retract(target)
        μ     += G^{-1} · (Aᵀ c)            # drive null leakage Aᵀc → 0
```

`M` (number of nulls) is typically 1–4, so each step is a trivial small solve.

**The gauge ψ matters more than it looks.** ψ does *not* change the
inter-element taper — only each element’s absolute point on the lossy manifold.
Empirically the post-refinement *gain* is nearly ψ-invariant (≈ 0.6 dB span),
but the achievable *null depth* varies by **50+ dB** with ψ, and the
gain-optimal ψ sits ≈ 3 rad away from the PTNR-optimal ψ. So scoring ψ by gain
(the legacy default) lands on the wrong peak and leaves 10–16 dB of null depth on
the table. We therefore default to `gauge='ptnr'`: evaluate the *refined*
solution at ~24 candidate ψ and keep the best peak-to-null ratio.

### 7.4 Gain-Locked Coordinate Polish (GLCP) — the novel refinement

Greedy coordinate descent on the manifold **grid** that deepens nulls while the
main-beam magnitude is held by a **hard constraint**:

```
        for each element i:
            choose grid point  argmin_c  Σ_m |AF(u_m)|^2
            subject to  |AF(u_t)|  ≥  gain_floor · |AF(u_t)|_current
```

This is the key difference from a soft-PTNR-penalty local refinement: gain
*cannot* drop below the floor (default `gain_floor = 0.995`), so the optimizer
spends all of its freedom on the nulls. Implemented with running sums
`S_t = Σ c_n a_t,n` and `S_m = Aᵀ c`, each element trial is an **O(G + G·M)
rank-1 incremental update** — no FFT, no full array-factor recomputation — for a
total of `O(rounds · N · G · M)` arithmetic but **zero** full oracle calls.

### 7.5 Cost and guarantees

- **Cost:** `gauge_samples · (n_dual_steps + 2)` oracle calls (≈ 240 for the
  defaults) plus `rounds·N` committed GLCP moves implemented incrementally.
  Independent of any iteration-to-convergence; see Fig. 7(b) — the oracle cost is
  flat in `N` while local/DFO baselines grow to `10^3–10^4`.
- **Gain floor:** GLCP provably never drops `|AF(u_t)|` below the floor (unit
  test `test_gain_locked_polish_preserves_gain`).
- **Optional SLL active set:** `suppress_sll=True` iteratively adds the worst
  sidelobe peaks as virtual nulls.

---

## 8. The decisive finding: discrete manifold realism

Real hardware is a **measured lookup table**, not a differentiable analytic
function. We therefore evaluate every method twice:

- **smooth** (`discrete=False`): treat the analytic `c(V)` as differentiable — an
  optimistic upper bound you never actually have from measurements;
- **discrete** (`discrete=True`, the realistic default): snap `V` to `V_grid`
  before evaluating — the faithful model of a measured device.

The result (Fig. 8) is stark. Continuous gradient / least-squares / black-box
methods (Trust-Region/LM, PGD, Basin Hopping, Simulated Annealing, Differential
Evolution) dig **fantasy** nulls on the smooth model (down to −120…−150 dB) that
**collapse by 50–150 dB** — to roughly **0 dB** — on the quantized device. Their
deep nulls were a smoothness artifact: they relied on finite-differencing below
the voltage quantization step.

**Grid-native methods are invariant.** Coordinate Descent, Perturbation nulling,
and **ours** give essentially the *same* null depth smooth or discrete (collapse
≈ 0). MR-LCMV+GLCP retains its deep null on the real device because every one of
its stages (linear retraction, dual correction, hard-locked coordinate polish)
operates *on the grid itself*. This is why "design-ideal-then-project" (RCG,
ADMM) and finite-difference local/LS methods are **not credible** on measured
hardware, and it is the central reason a gain-first, grid-native solver is the
right tool.

> This collapse is **formalized and quantified** in [`THEORY.md`](./THEORY.md):
> Proposition 1 (phantom-null lemma) explains the mechanism, Proposition 2
> (quantization null floor) predicts the achievable null depth as ≈ `ε√N`, and
> `fig10` confirms the predicted **−20 dB/decade** recession of the floor with
> hardware resolution (measured −19.8 dB/decade).

---

## 9. Figure → claim map

| Figure | File | Supports |
|---|---|---|
| 1 | `fig1_distributions` | **Statistical**: distribution (median/IQR/spread) of PTNR, gain-loss, worst-null over random tasks — not a cherry-picked task. |
| 2 | `fig2_success_heatmap` | **Reliability**: `P(worst null ≤ −20/−30/−40/−50 dB)`; ours is the only method at 100% down to −50 dB. |
| 3 | `fig3_pareto` | **Trade-off**: gain retention vs nulling; ours occupies the top-left (best) corner. |
| 4 | `fig4_winrate` | **Statistical**: per-instance PTNR win-rate. |
| 5 | `fig5_perf_profile` | **Performance**: Dolan–Moré profile keyed on **oracle calls**; ours is most reliable *and* cheapest (100% at τ=1). |
| 6 | `fig6_cost_quality` | **Performance**: efficiency frontier — PTNR per oracle call; ours top-left. |
| 7 | `fig7_scaling` | **Performance/scaling**: PTNR and oracle cost vs `N`; ours’ cost is ~flat in `N`. |
| 8 | `fig8_manifold_realism` | **Decisive mechanism**: smooth-vs-discrete null collapse; ours invariant, gradient/DFO are artifacts. |
| 9 | `fig9_beam_patterns` | **Qualitative**: representative array factors; ours keeps the beam and digs both nulls. |
| 10 | `fig10_quantization_floor` | **Theory**: achievable null floor vs hardware resolution; ours tracks `ε√N` at −19.8 dB/decade, phantom is flat & unreachable (`THEORY.md`). |
| 11 | `fig11_robustness_calibration` | **Robustness**: worst-null vs per-element calibration error; ours keeps the deepest null at every error level. |
| 12 | `fig12_robustness_dead_elements` | **Robustness**: worst-null vs number of dead/stuck elements. |
| 13 | `fig13_significance` | **Significance**: paired Wilcoxon (Holm-corrected) + Cliff's δ; ours beats every baseline by 14–46 dB median PTNR (p < 10⁻³). |
| 14 | `fig14_ablation` | **Ablation**: marginal value of the PTNR gauge (+8 dB) and GLCP (+32 dB); full method 93 dB. |
| 15 | `fig15_convergence` | **Anytime cost**: null depth vs oracle-call budget; no baseline reaches ours' null at any budget. |

Tables: `pub_runs.csv`, `pub_summary.csv`, `pub_profile.csv`, `pub_scaling.csv`,
`pub_realism.csv` (main); `pub_quantization.csv` (theory); `pub_robustness.csv`,
`pub_significance.csv` (robustness/stats); `pub_ablation.csv`, `pub_convergence.csv`
(ablation/convergence).

### Robustness, significance, and ablation (summary of findings)

- **Robustness (`fig11`/`fig12`).** Deep nulls are calibration-limited: under
  per-element phase jitter all methods degrade toward a common floor, but ours
  starts far deeper and **retains the most null margin at every error level** —
  the figure doubles as a calibration spec (e.g. ≤ 0.5° RMS to keep a ≥ 30 dB null
  at `N = 32`). Graceful degradation with dead elements.
- **Significance (`fig13`).** Across 16 paired random tasks, MR-LCMV+GLCP exceeds
  **every** baseline on PTNR with a one-sided Wilcoxon signed-rank
  p = 1.5×10⁻⁵ (Holm-corrected p < 10⁻³ for all) and Cliff's δ = 0.71–1.00;
  median advantage 14–46 dB. It also beats the **bare anchor** by 42 dB, so the
  refinement is not cosmetic.
- **Ablation (`fig14`).** Median PTNR climbs 48 → 56 (PTNR gauge) → 80 (GLCP) → 93
  dB (full): both novel components contribute and combine super-additively.

---

## 10. Reproducibility

```bash
# Full statistical figure set + CSV tables (vector PDF + PNG into experiments/figures/):
./.venv/Scripts/python.exe experiments/publication_figures.py --tasks 10 --seeds 4 \
    --scale-N 16 32 64 128 256

# Fast smoke (a few tasks, one seed):
./.venv/Scripts/python.exe experiments/publication_figures.py --quick

# Theory (Stream 1): quantization null-floor figure + CSV (fig10):
./.venv/Scripts/python.exe experiments/quantization_floor.py

# Robustness + significance (Stream 2): fig11/12/13 + CSVs:
./.venv/Scripts/python.exe experiments/robustness_study.py --tasks 16 --draws 200

# Ablation + anytime convergence (Stream 4): fig14/15 + CSVs:
./.venv/Scripts/python.exe experiments/ablation_convergence.py --tasks 12

# Regression tests for the solver, the statistical claims, and the theory:
./.venv/Scripts/python.exe -m pytest tests/test_coherent_lcmv.py \
    tests/test_baselines_statistical.py tests/test_quantization_floor.py -q
```

The task set is seeded (`TASK_SEED` in `sota_families_benchmark.py`); the element
is `SyntheticVaractor(beta=0.8, folding=True)`. PDFs are tracked in git; PNGs and
CSVs are `.gitignored` (regenerate locally).

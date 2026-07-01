# Theory: Why Deep Nulls on a Smooth Model Are Unrealizable on Quantized Hardware

This note formalizes the empirical "collapse" finding of
[`ALGORITHMS.md` §8](./ALGORITHMS.md) into two propositions and validates them
quantitatively. It is the theoretical core of the contribution: it explains *why*
gradient / least-squares / black-box optimizers report fantastically deep nulls
that vanish on real hardware, and it predicts the null depth a grid-native solver
can actually achieve as a function of hardware resolution.

Experiment: [`experiments/quantization_floor.py`](../experiments/quantization_floor.py)
→ `experiments/figures/fig10_quantization_floor.(pdf|png)` and
`pub_quantization.csv`.

---

## Setup and notation

`N`-element ULA, array factor `AF(u) = Σ_n c_n e^{jπ n u}`, steering vectors
`a(u)_n = e^{jπ n u}`. Each element weight is drawn from a **per-element manifold**
`M_n`. On real hardware `M_n` is a **finite set** of achievable complex points
(a measured lookup table indexed by control voltage), not a continuum.

Define the manifold **covering radius**

```
        ε  =  max_{w ∈ conv(M)}  min_{c ∈ M} |c − w|,
```

i.e. the worst-case distance from a desired weight to the nearest achievable
grid point. For a 1-D voltage-swept curve sampled at `R` points, `ε` scales like
the curve length over `R`, so **`ε ∝ 1/R`** (halving the step → halving `ε`).

The continuous (relaxed) problem replaces `M_n` by its convex hull / the full
circle; the LCMV projector (`ALGORITHMS.md` §7.1) achieves `AF(u_m) = 0` *exactly*
for any `M < N` nulls. Continuous null depth is therefore **unbounded**
(`→ −∞`). The questions are (1) what a finite-difference optimizer *does* with
that fact, and (2) what is actually *achievable* once `M_n` is finite.

---

## Proposition 1 (phantom-null lemma)

> Let a solver estimate descent directions by finite differences with probe step
> `h` in control-voltage space, and let `δ_V` be the local voltage spacing of the
> grid. Evaluated on the **quantized** device (weights snapped to the grid before
> the objective), if `h < δ_V` then every probe `V ± h` snaps to the same grid
> point as `V`, so the measured finite difference is **identically zero** and the
> solver observes a vanishing gradient: it cannot descend below the grid. Evaluated
> on the **analytic** model `c(V)` (no snapping), the same finite difference is
> nonzero and the solver drives the null residual toward machine zero. Hence any
> null deeper than the grid-induced floor (Prop. 2) that a finite-difference /
> least-squares method reports on the analytic model is **unrealizable** on the
> device.

*Mechanism.* The analytic model grants the optimizer an infinitely fine handle on
each weight; the device does not. Methods that *design in the continuum and then
project* (Riemannian CG, ADMM) inherit a weaker version of the same defect: they
optimize the residual where `c_n` is free, then round once — they never jointly
optimize the **rounded** residual, so they do not even reach the achievable floor.

*Evidence.* On the analytic model Trust-Region/LM digs nulls of `−120…−190 dB`
**independent of grid resolution** (flat purple curve in `fig10`); on the device
the identical solver collapses to `≈ 0…+8 dB` (`fig8`). The reported depth was a
property of the model, not the hardware.

---

## Proposition 2 (quantization null floor)

> Let `c*` be the continuous optimum (`AF*(u_m) = 0`) and let `ĉ_n ∈ M_n` be the
> realized grid weights with per-element error `e_n = ĉ_n − c*_n`, `|e_n| ≤ ε`.
> The realized null residual is
> ```
>         AF(u_m) = Σ_n ĉ_n a(u_m)_n = AF*(u_m) + Σ_n e_n a(u_m)_n = Σ_n e_n a(u_m)_n .
> ```
> Therefore
> * **worst case** (adversarial alignment): `|AF(u_m)| ≤ N ε`;
> * **generic case** (errors with incoherent phases across elements): the residual
>   is a sum of `N` terms of magnitude `≤ ε` with quasi-random phases, so
>   `|AF(u_m)| ≈ ε √N` (a random walk).
>
> Relative to the coherent peak `N`, the generic achievable **worst-null depth** is
> ```
>         20·log10|AF(u_m)|  ≈  20·log10(ε) + 10·log10(N) + const,
> ```
> and since `ε ∝ 1/R`, the floor **recedes by ≈ 20 dB per decade (≈ 6 dB per
> octave / per added phase bit) of grid resolution `R`.** A solver that minimizes
> the residual *directly over the grid* (ours: linear retraction + dual correction
> + hard-locked coordinate polish) attains this floor; design-then-project methods
> do not.

This converts "deeper nulls need finer hardware" from folklore into a slope you
can predict, measure, and budget against (e.g. *to guarantee a −50 dB null at
`N = 32` you need ≳ 8 effective bits*).

---

## Quantitative validation

`experiments/quantization_floor.py` fixes `N = 32` and a set of random nulling
tasks, sweeps the manifold resolution `R` from 16 to 4096 points
(≈ 4 → 12 effective bits), and measures the achievable worst-null for three
behaviors (mean over tasks). Summary (`pub_quantization.csv`, see `fig10`):

| Resolution `R` | ≈ bits | `ε` | **Ours** (grid-native, device) | RCG (project) | Trust-Region (analytic phantom) |
|---:|---:|---:|---:|---:|---:|
| 16   | 4  | 0.139  | **−33 dB** | −5 | −120 |
| 64   | 6  | 0.034  | **−46 dB** | −6 | −137 |
| 256  | 8  | 0.008  | **−62 dB** | −5 | −162 |
| 1024 | 10 | 0.002  | **−66 dB** | −5 | −140 |
| 4096 | 12 | 0.0005 | **−82 dB** | −5 | −125 |

- **Ours tracks the `ε√N` floor with a measured slope of −19.8 dB/decade**, against
  the predicted **−20 dB/decade** — a direct confirmation of Prop. 2.
- **RCG is flat at ≈ −5 dB** for every resolution: designing ideal phases and
  projecting once cannot exploit the amplitude–phase-coupled grid, so finer
  resolution does not help it (Prop. 1, weak form).
- **The analytic phantom is flat at ≈ −120…−160 dB**, resolution-independent and
  unreachable — exactly what Prop. 1 predicts, and exactly the nulls the
  smooth-model gradient/DFO methods "win" with in `fig8` before they collapse.

---

## Consequences for the paper's claims

1. The smooth-vs-discrete collapse (`fig8`) is **not** an implementation accident;
   it is forced by Prop. 1. Any method whose null depth depends on differentiating
   below the grid step is reporting a model artifact.
2. The right figure of merit is the **achievable grid floor** (Prop. 2), and the
   right solver is one that **optimizes on the grid**. Ours attains the floor;
   continuous-design-then-project does not.
3. Resolution becomes a *spec*: `fig10` lets a system designer read off the bits
   required for a target null depth at a given `N`, instead of trusting an analytic
   simulation that promises arbitrarily deep nulls.

See also [`ALGORITHMS.md`](./ALGORITHMS.md) (§4 cost model, §7 our method, §8 the
discrete-manifold realism argument).

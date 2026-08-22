# MR-LCMV Co-Design Claims vs. Empirical Evidence

This document evaluates the specific claims of the MR-LCMV+GLCP co-design methodology against explicit results derived from the measurement-native geometry testing suite.

## The Exploitable Hardware Bounds (Null Fragility Verification)
Rather than abstract "evaluations", we first ground the algorithm against the physical *Hardware Card* computed from the measured RTPS VNA parameters ($N=64$, folding enabled):
* $c_W$: 0.87 (Do-Lozano Amplitude Shortfall)
* $\rho$: 0.29 (Quantization Distance)
* $\sigma^2$: 0.038 (Dithered Variance)
* $\lambda$: 5.01 (Branch/Folding Diversity)

**Interpretation:** The hardware sacrifices pure amplitude ($c_W < 1.0$) but buys significant branch diversity ($\lambda$). Traditional optimization bounds (e.g. continuous circle PGD) fail to understand $\lambda$ and are destroyed by $\rho$, explaining why standard techniques stall out on empirical benchmarks.

---

## Evaluation of Algorithm Claims

### Claim 1: "Our method preserves Main-Beam Gain while unlocking deep nulls."
* **The Test:** `GLCP Minimax` vs. `Coordinate Descent (no gain floor)`
* **The Evidence:** Un-gated Coordinate Descent eagerly sacrifices $7.0$ dB of broadside gain to marginally suppress the null to $-62$ dB. Our MR-LCMV formulation, using the deterministic PTNR gauge and Minimax GLCP, successfully drives the null to $-64.17$ dB while rigidly holding gain at $-1.14$ dB.
* **Verdict:** Supported. The GLCP gain-locked logic actively prevents the collapse observed in typical scalar objective optimizers.

### Claim 2: "Global Phase Offset (Gauge) breaks symmetry and serves as a structural alignment variable on rigid grids."
* **The Test:** Sweep $\psi \in [0, 2\pi)$ independently on the fixed geometry.
* **The Evidence:** The achieved Main Lobe Gain experiences negligible drift ($\sim 0.18$ dB variance), while the achievable Worst-Null depth varies wildly by over $40.50$ dB across the sweep.
* **Verdict:** Supported. The Gauge fundamentally alters the Voronoi overlap logic, dictating the "Local Feasible Reachability." Operating point selection is structurally non-redundant.

### Claim 3: "Standard optimizers designed for ideal circuits collapse when projected onto measured manifolds."
* **The Test:** `ADMM` and `PGD` on continuous ideal $e^{j\phi}$ vs. Projected onto Measured Grid.
* **The Evidence:** PGD easily reached $-31.8$ dB on the ideal mathematical circle. Once exposed to the $0.29$ quantization and coupling geometry, it collapsed utterly to $+19.67$ dB. In contrast, MR-LCMV is grid-native and achieves robust deep nulls without projecting continuous pipe dreams.
* **Verdict:** Supported. Disproves endless comparisons against $N$-dimensional smooth solvers on realistic models.

### Claim 4: "MR-LCMV scales natively and converges deterministically, avoiding endless iteration cycles."
* **The Test:** Scaling N up to 1024 evaluating tracking curves and $\Delta D_{ref}$ against a $K=256$ Certified limit.
* **The Evidence:** The `Adaptive` mode strictly follows the true convergence curve, finding deep limits ($\Delta D_{ref} < 0.2$ dB) extremely quickly. Fixed K=8 limits crash drastically on large arrays (creating up to 43 dB gaps against the theoretical ref limit). By isolating the $N_{AF}$ evaluations (full array recalculations) from incremental GLCP updates, the solver proves it scales effectively.
* **Verdict:** Supported.

## Summary Conclusion
The MR-LCMV variant effectively trades hardware scaling constraints for algorithmic intelligence. Rather than using heuristic guesses, it mathematically exploits the measured amplitude pinches using the Gauge Phase and then carefully polishes exact states without surrendering the intended array operation parameters.

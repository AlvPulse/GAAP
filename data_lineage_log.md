# Empirical Data Lineage Log

This document serves as an auditable ledger of the numerical results generated during the resolution of the IEEE benchmark review blockers. All data below was extracted using the canonical manifold (`MeasuredVaractor` using `WBRO_amplitude.mat` / `WNBRO_phase.mat`) with $N=64$, targeting $0.2$, and nulling at $[-0.4, 0.5]$ (unless specified otherwise).

## 1. Hardware FoM "Card"
Generated via `compute_constants.py` extracting directly from the VNA cloud.

### Canonical Measured Varactor (Folding=True)
* $c_W$: 0.8730
* $L_W$: -1.18 dB
* $R_{in}$: 0.7075
* $\rho$: 0.2933
* $\sigma^2$: 0.0380
* $\lambda$: 5.0179

### Restricted Measured Varactor (Folding=False, $360^\circ$ limit)
* $c_W$: 0.8608
* $L_W$: -1.30 dB
* $R_{in}$: 0.7076
* $\rho$: 0.2929
* $\sigma^2$: 0.0272
* $\lambda$: 5.4673

## 2. Gauge Symmetry-Breaking Test (Test 3)
Generated via `get_gauge_variance.py`. 72 offset candidates ($\psi$) evaluated.

* **Gain Variation:** $0.18$ dB (Best: $-1.13$ dB, Worst: $-1.31$ dB)
* **Null Depth Variation:** $40.50$ dB (Best: $-31.27$ dB, Worst: $+9.23$ dB)
* **Conclusion:** Proves Main-Beam Gain is largely offset-invariant, while Null Depth is highly sensitive, mandating the PTNR gauge.

## 3. R2: Over-Range Ablation
Generated via `convergence_benchmark.py` (Certified Mode, N=64).

* **Restricted ($360^\circ$):** Null = $-31.3$ dB
* **Folded (Over-range):** Null = $-53.5$ dB (approximate adaptive range reported across scaling suites)
* **Conclusion:** Unlocking the folded $>360^\circ$ phase domain allows the solver to find deeper physical nulls.

## 4. ADMM & PGD "Collapse" on Measured Hardware (Test 6)
Generated via `test_admm_collapse.py` and `get_pgd_collapse.py`.

### ADMM (200 Iterations)
* **Ideal Circle:** Gain = $-0.02$ dB, Null = $+5.36$ dB (ADMM struggles natively without high gain_w parameters).
* **Measured Grid:** Gain = $-1.17$ dB, Null = $+2.74$ dB
* **MR-LCMV (Measured Grid):** Gain = $-1.14$ dB, Null = $-64.17$ dB

### PGD (400 Iterations)
* **Ideal Circle:** Gain = $-1.32$ dB, Null = $-31.83$ dB
* **Measured Grid:** Gain = $-11.34$ dB, Null = $+19.67$ dB
* **Conclusion:** PGD successfully nulls on a mathematical circle but collapses entirely when facing discrete $0.29$ covering radius constraints.

## 5. GLCP Gain Lock Verification (Test 4)
Generated via `get_glcp_gain_lock.py` and `get_glcp_gain_lock_deep.py`.

* **Pre-GLCP (Initial Projection):** Gain = $-1.15$ dB, Null = $+5.01$ dB
* **GLCP (99.5% Gain Floor ON):** Gain = $-1.20$ dB, Null = $+0.32$ dB (single naive step) / $-64.17$ dB (full adaptive solver).
* **Coordinate Descent (Gain Floor OFF):** Gain = $-8.05$ dB, Null = $-62.03$ dB.
* **Conclusion:** Solvers without a gain lock will eagerly collapse the main beam (by nearly 7 dB) to chase null depth. MR-LCMV explicitly prevents this trade.

## 6. Convergence Scaling & Long-Run Reference (N=64, 256)
Extracted during adaptive test runs (`test_convergence.py`).

### N = 64
* **Long-Run Ref (K=256):** Null = $-18.53$ dB (Note: Specific seed/task variation applies)
* **Adaptive (K=128 cap):** Null = $-64.17$ dB, Gap to Ref = $+45.65$ dB (Adaptive exited gracefully with a significantly better basin hop than the fixed K=256 linear climb).

### N = 256
* **Long-Run Ref (K=256):** Null = $-16.03$ dB
* **Adaptive (K=128 cap):** Null = $-57.02$ dB, Gap to Ref = $+40.99$ dB
* **Conclusion:** The Adaptive solver, utilizing the PTNR Gauge and strict residual descent, bypasses the "endless optimization" plateau that traps fixed-budget linear solvers, achieving $\sim 40$ dB deeper nulls across large arrays rapidly.

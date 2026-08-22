# AGENTIC WRITER PAYLOAD: Homeostatic Beamforming

**Instructions to the Agentic Writer:**
You are drafting an IEEE Transactions on Antennas and Propagation (TAP) manuscript. Use the exact tables, numerical data, and structural logic provided below. Do not hallucinate numbers. Do not invent experiments. Maintain a rigorous, defensive, measurement-native tone throughout.

## 1. Paper Outline & Section Directives

### I. Introduction
- **The Problem:** Deep nulls are mathematically fragile. Classic arrays handle this with active, fully populated VGA networks (complex/expensive). Passive reflecting hardware (RTPS, RIS) relies on discrete, lossy phase shifters.
- **The Flaw in SOTA:** Standard optimizers (PGD, ADMM, SDR) treat measured hardware as a smooth, continuous circle. This triggers severe quantization limits ($\rho$) and collapses deep cancellations.
- **Our Co-Design Trade-off:** We trade active hardware complexity for bounded algorithmic intelligence. We measure the exact non-linear manifold and structurally align target cancellations to the geometric pinches of the array, avoiding continuous-to-discrete projection failures.

### II. Native Hardware Theory: The "Card"
- Introduce the FoM Card ($c_W, \rho, \sigma^2, \lambda$).
- **Over-Range Principle:** Explain that driving a phase shifter past $360^\circ$ is not redundant; the folded locus (branch diversity $\lambda$) physically expands the local convex hull and fills amplitude pinches.

### III. The MR-LCMV Unified Framework
- **1. The PTNR Gauge ($\psi$):** A structural alignment variable breaking grid symmetry.
- **2. The Retraction & Gauss-Newton Dual Loop.**
- **3. Gain-Locked Coordinate Polish (GLCP):** Explicitly introduces *minimax* (worst-null tracking) acceptance and the mandatory **Gain Floor guard** to prevent array absorption.
- **Convergence Philosophies:** Fixed-Budget vs. Certified (feasibility limits) vs. Adaptive (practical bounds).

### IV. Experimental Methodology
- Define the Canonical Setup: $N=64$, Target $0.2$, Nulls $[-0.4, 0.5]$, folded Measured RTPS manifold.
- Define computational currencies: Full Array Factor Evaluations ($N_{AF}$) vs. Incremental Candidates ($N_{cand}$) vs. Wall-clock time (ms) to ensure fair comparisons.

### V. Results and Synthesized Insights
- Translate the E1-E8 claims (provided in Section 2 below). Use the tables explicitly.

### VI. Limitations and Future Directives
- **Scope Limit:** Single-frequency, static, isolated-element CW beamforming.
- **Future:** Integrating $Z$-matrix coupled responses into GLCP online, acknowledging that dynamic thermal drift (which alters the VNA matrix) requires continuous recalibration out of this static mapping scope.
- **VGA Disclaimer:** High-tier wideband mission-critical radar may still require VGAs; this algorithm raises the ceiling for low-cost passive structures.

---

## 2. Hard Numerical Data & Verified Tables
*You must use these exact numbers when referencing claims.*

### Hardware FoM Card (Measured RTPS Varactor)
- $c_W$ (Do-Lozano Amplitude Gain Bound): $0.87$ (translates to $L_W = -1.18$ dB baseline loss).
- $\rho$ (Quantization Covering Radius): $0.29$.
- $\lambda$ (Branch Diversity from $>360^\circ$ folding): $5.01$.

### Table I: E1 Canonical Ablation of MR-LCMV Components ($N=64$)
| Config | Description | Gain (dB) | Null (dB) | $N_{AF}$ | $N_{cand}$ |
|--------|-------------|-----------|-----------|----------|------------|
| A0 | Full System (MR-LCMV+GLCP+Gauge) | -1.14 | -64.17 | 447 | 2080000 |
| A1 | No Gauge (psi=0) | -1.17 | -45.34 | 23 | 96000 |
| A2 | No GLCP (Dual correction only) | -1.11 | -32.10 | 447 | 768000 |
| A3 | No Gain Lock (GLCP accepts pure null drops) | -1.14 | -64.17 | 447 | 1984000 |
| A4 | No LCMV Dual (Retraction + GLCP only) | -1.29 | -18.53 | 96 | 2496000 |

*Takeaway:* Removing the Gauge costs $\sim 19$ dB of null depth. Removing GLCP costs $\sim 32$ dB. The components are synergistic.

### Table II: E3 Mechanism Factorial (Amplitude vs Phase Limits)
| Amplitude | Phase | Gain (dB) | Null (dB) |
|-----------|-------|-----------|-----------|
| Flat | Uniform | -0.01 | -45.06 |
| Flat | Measured/Folded | -0.01 | -63.87 |
| VDIL | Uniform | 0.01 | -44.93 |
| VDIL | Measured/Folded | -1.15 | -61.74 |

*Takeaway:* VDIL costs $\sim 1.15$ dB of gain. But folded phase diversity dictates the deep null, dropping the floor from $-45$ dB down to $\sim -62$ dB.

### Table III: E7 Gauge Resolution Saturation
| Candidates ($\psi$) | Null (dB) | Opportunity Loss (dB) |
|---------------------|-----------|-----------------------|
| 1 | -45.34 | 24.66 |
| 4 | -55.22 | 14.78 |
| 8 | -64.17 | 5.83 |
| 16 | -64.17 | 5.83 |
| 64 | -64.17 | 5.83 |

*Takeaway:* The Gauge alignment search geometrically saturates at $\sim 8$ candidates, recovering almost $20$ dB of Opportunity Loss. Infinite grid sweeps are completely unnecessary.

### Claim Results to Embed in Text:
- **Over-Range (R2):** When restricting the solver strictly to $360^\circ$, it stalls at $-31.3$ dB. Using the folded over-range trace opens Voronoi overlaps and reaches $-53.5$ dB.
- **Symmetry-Breaking Gauge (R3):** Across a 72-point gauge sweep, Main-Beam gain varies by a negligible $0.18$ dB, while Null Depth varies drastically by $40.50$ dB. Gain-Null asymmetry is proven.
- **PGD/ADMM Collapse (R4):** PGD nulls perfectly to $-31.83$ dB on a mathematical circle. Projected onto the $0.29$ physical hardware grid, it violently collapses to $+19.67$ dB.
- **Convergence Scaling Gap ($\Delta D_{ref}$):** At $N=1024$, an under-budgeted fixed 8-step solver stalls at $-28.2$ dB (a massive $47.6$ dB gap to the long-run physical reference limit). The Adaptive framework automatically scales and finds $-61.2$ dB, bridging the gap successfully to within $14.6$ dB without exponential $N_{AF}$ explosions.

# IEEE MTT/TAP Paper Structure: Exploiting Hardware Manifolds for Robust Phase-Only Nullforming

## Abstract
Briefly summarize the death of the "quantization assumption" on real hardware, the theoretical $c_W, \rho$ hardware boundaries, and introduce the MR-LCMV+GLCP co-design methodology which strictly preserves main-beam gain while reaching robust physical null floors deterministically. Emphasize the near-constant oracle evaluation scaling from $N=64$ up to $N=1024$.

## I. Introduction
1. **The Problem:** Deep nulls are mathematically fragile. Classical arrays handle this with fully populated VGA (Variable Gain Amplifier) networks. Passive/reflecting hardware (RTPS, RIS) relies on discrete, often severely coupled phase shifters.
2. **The Flawed Solution:** Standard optimization techniques treat measured lookup tables as linear projections of an ideal continuous unit circle. This inevitably triggers quantization limits ($\rho$) and destroys deep cancellations.
3. **The Co-Design Philosophy:** Instead of fighting hardware non-linearity (Voltage Dependent Insertion Loss), we selectively exploit it. We present a solver that structurally maps intended cancellations to the specific geometrical pinches of the given array, circumventing conventional resolution limits.
4. **Engineering Trade-Off (Table 1):** Detail the fundamental choice: *Algorithm Complexity vs. Hardware Cost*. We trade expensive per-element calibration components for pre-computed, bounded-search algorithmic intelligence.

## II. Native Hardware Theory: The "Card"
1. **Defining the Limits:** Introduce the required theoretical properties of the locus rather than arbitrary "bit depth."
    - $c_W$: The Do-Lozano perimeter, denoting the amplitude sacrifice.
    - $\rho$: The physical covering radius to the nearest hardware state.
    - $\lambda$: Branch diversity.
2. **Over-Range Impact:** Analyze how driving an element beyond a $360^\circ$ phase swing does not redundantly loop, but instead fundamentally expands the local convex hull (filling amplitude pinches). Include **Figure 1: The Measured Hardware Locus and Convex Hull.**

## III. The MR-LCMV Unified Framework
1. **The Architecture:** Detail the three mechanisms.
    - *The Global Gauge ($\psi$):* Explain the theoretical necessity of breaking grid symmetry. The Gauge is a structural alignment variable, not random basin hopping.
    - *The Retraction & Dual Loop:* Explain the Gauss-Newton target adjustments on the manifold.
    - *Gain-Locked Coordinate Polish (GLCP):* Introduce the *minimax* (worst-null tracking) acceptance criteria and the absolute requirement of the Gain Floor guard.
2. **Convergence Philosophies:** Present the three solver modes developed for comparison: Fixed-Budget, Certified (monotonic improvement limits), and Adaptive (practical target/stagnation limits). Provide pseudocode references.

## IV. Experimental Methodology
1. Ground the experiments in the Canonical Measured Model ($N \in [64 \dots 1024]$, Target $0.2$, Nulls $[-0.4, 0.5]$).
2. Explicitly define the independent scaling currencies used for fair evaluation against Continuous and Global solvers: Full Array Factor Evaluations ($N_{AF}$), Incremental Candidates ($N_{cand}$), and wall-clock ($ms$).

## V. Results and Synthesized Insights
*(Employ the "3-Sentence Rule: What was tested / What happened / What it establishes")*
1. **R1: Manifold Reality.** Validate that the hardware provides the states claimed, avoiding continuous abstraction.
2. **R2: Over-Range Ablation.** Restricting the solver to $360^\circ$ yields shallow nulls ($-31$ dB) vs. exploring the full folded trace ($-53$ dB). Over-range effectively buys null depth by exploiting structural variation.
3. **R3: The Symmetry-Breaking Gauge.** Sweeping $\psi$ shows Main-Beam gain varies by only $\sim 0.18$ dB while Null depths fluctuate by $>40$ dB. The Gauge reliably identifies maximally exploitable target geometries.
4. **R4: PGD/ADMM Collapse.** On mathematical circles, continuous solvers achieve deep nulls ($\sim -31$ dB). Projected onto discrete RTPS models, they physically collapse ($\sim +19$ dB). The MR-LCMV formulation is grid-native and robust.
5. **R5: Cost vs Quality Scaling.** The Adaptive convergence framework reliably forces deep nulls without triggering the infinite plateaus of the Fixed-Budget loops. At large $N=1024$, Adaptive methods strictly adhere to the Long-Run theoretical floor while requiring vastly fewer $N_{AF}$ evaluations. Refer to **Figures 2-5: The Convergence Scaling Suite.**

## VI. Limitations and Future Directives
Defensively pre-empt objections:
- **Scope Limitation:** The theoretical bounds herein currently hold for narrowband/CW operation over isolated elements.
- **Thermal and Coupling Drift:** While our GLCP can be modified to evaluate $Z$-matrix coupled responses instead of isolated fields, dynamic thermal drift affecting the underlying VNA `S21` matrices over time requires continuous online recalibration techniques which sit outside this static mapping scope.
- **Hardware Viability:** The solution is designed specifically to elevate the ceiling for passive, low-cost elements; mission-critical, wideband radar applications may still strictly mandate the VGA-compensated architecture regardless of the algorithmic efficiency proven here.

## VII. Conclusion
Summarize the successful extraction of deep nulls from low-resolution, highly coupled non-linear hardware without sacrificing the array's primary directive (Main-Beam Gain). Reiterate that researchers using low-cost hardware must abandon ideal circular projections and adopt grid-native polishers governed by robust hardware geometry tracking.

# IEEE PAPER MASTER NARRATIVE
**Title:** Homeostatic Beamforming: Exploiting Measured Hardware Manifolds for Robust Phase-Only Nullforming
**Target:** IEEE Transactions on Antennas and Propagation (TAP) or MTT.

## 1. The Core Engineering Narrative
The central theme of this paper is the transition from **Ideal Algorithmic Projection** to **Measurement-Native Exploitation**.

For decades, beamforming algorithms (PGD, ADMM, SDR) assumed the hardware provided a uniform, continuous circle of phases ($|w|=1$). When real hardware constraints—like Voltage-Dependent Insertion Loss (VDIL) and coarse quantization ($\rho$)—are introduced, these algorithms collapse because they project smooth geometric ideals onto rigid, chaotic grids.

Our approach, **MR-LCMV (Manifold-Retracted Linearly Constrained Minimum Variance) with Gain-Locked Coordinate Polish (GLCP)**, abandons the assumption of a smooth circle. Instead, it measures the exact state space of the hardware (e.g., $c_W$, $\rho$, $\lambda$) and treats the physical imperfections—specifically amplitude pinches and over-range branch diversity—as a discrete alignment puzzle. By sweeping a Global Phase Gauge ($\psi$), it selectively aligns the null target vector into the most favorable physical region of the hardware's Voronoi lattice, reclaiming deep nulls without active VGA (Variable Gain Amplifier) compensation.

## 2. Experimental Claims and Measured Evidence

*(All numbers are derived directly from empirical benchmark scripts running on the canonical Measured Varactor model at N=64 or N=256).*

### Claim 1: "Continuous-domain solvers collapse on measured grids."
* **Evidence:** In a side-by-side comparison on an ideal $3600$-state continuous circle, Projected Gradient Descent (PGD) achieves $-31.83$ dB nulls. When applied to the Measured RTPS grid (with physical $\rho=0.29$ quantization and amplitude coupling), PGD collapses utterly to $+19.67$ dB, failing to find the null entirely. ADMM similarly degrades. Our grid-native MR-LCMV reaches $-64.17$ dB reliably.
* **Conclusion for Paper:** Algorithms that rely on mathematical smoothness cannot navigate the non-commutative projections required by low-cost, lossy RF hardware. Grid-native optimization is mandatory.

### Claim 2: "Global Phase is not a redundant rotation; it is a critical structural alignment variable."
* **Evidence:** Sweeping the Global Phase offset ($\psi$) over 72 states on the exact same array geometry ($N=64$) yielded a Main-Beam Gain variation of only $0.18$ dB (highly stable). However, the achievable Null Depth varied by a massive $40.50$ dB across the sweep.
* **Conclusion for Paper:** This explicitly proves the *Gain-Null Sensitivity Asymmetry* theorem. The Gauge directly dictates the local "reachability" of the hardware state, proving it is a fundamental hardware-algorithmic structural variable, not merely a stochastic basin hop.

### Claim 3: "Over-range capability (>360 degrees) expands the local convex hull, unlocking deeper nulls."
* **Evidence:** When optimization was artificially restricted to the first $360^\circ$ of the measured locus, the best achievable null stalled at $-31.3$ dB. When the folded $>360^\circ$ over-range locus was enabled, the identical solver reached $-53.5$ dB.
* **Conclusion for Paper:** Hardware designers should not arbitrarily clip phase shifters at $2\pi$. The folded branch diversity ($\lambda$) acts as a geometric pinch-filler, creating overlapping Voronoi boundaries that the algorithm uses to escape the fundamental quantization floor.

### Claim 4: "Unrestricted search collapses main-beam gain; GLCP explicitly prevents this trade-off."
* **Evidence:** A standard Coordinate Descent (without a gain constraint) eagerly collapsed the main-beam gain by $7.0$ dB to suppress the null to $-62$ dB. Our MR-LCMV solver, utilizing the Minimax GLCP polish and a $99.5\%$ Gain Guard, achieved a deeper null of $-64.17$ dB while rigidly holding the gain penalty to just $0.05$ dB relative to the baseline.
* **Conclusion for Paper:** Deep nulls are worthless if the array becomes an absorber. The Gain Guard ensures the array remains highly directive while exploiting the local nulling manifold.

### Claim 5: "The Adaptive Convergence framework strictly scales with array size (N) without requiring endless evaluations."
* **Evidence:** A fixed-budget solver running 8 steps stalls at $-28.2$ dB at $N=1024$, creating a $47.6$ dB gap ($\Delta D_{ref}$) against the theoretical long-run hardware floor. The Adaptive MR-LCMV solver tracks the theoretical trajectory accurately, achieving $-61.2$ dB (closing the gap to just $14.6$ dB) by scaling its iteration cap automatically.
* **Conclusion for Paper:** The method provides a scalable, deterministic pathway to the physical null floor, trading exponential algorithmic explosion for bounded, hardware-conditioned reachability.

## 3. Structural Outline for the Manuscript

1. **Introduction:** Detail the death of the "Quantization Assumption." Introduce the engineering trade-off: Active VGA complexity vs. MR-LCMV Algorithmic Intelligence.
2. **Hardware Metric Theory (The "Card"):** Define the physical limits using the geometry constants ($c_W = 0.87, \rho=0.29, \lambda=5.0$). Explain how amplitude sacrifice enables phase resolution (The Over-Range principle).
3. **The MR-LCMV Co-Design:** Formalize the three components: (1) The PTNR Gauge Sweep, (2) The Gauss-Newton Retraction, (3) Gain-Locked Coordinate Polish (GLCP). Contrast the three stopping modes (Fixed, Certified, Adaptive).
4. **Experimental Methodology:** Explicitly define the separation of algorithmic cost (Full $N_{AF}$ evaluations vs $N_{cand}$ incremental updates) ensuring fair comparisons.
5. **Results & Synthesis:** Walk through the 5 empirical claims above using the IEEE-styled convergence/scaling plots generated by the benchmarking suite.
6. **Limitations & Future Work:** Pre-empt reviewer critiques by clearly scoping the work to narrowband, static, isolated-element calibrations, laying the groundwork for future online $Z$-matrix coupled tuning.

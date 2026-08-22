# Comprehensive IEEE Synthesis Analysis
**Title:** Homeostatic Beamforming: Exploiting Measured Hardware Manifolds for Robust Phase-Only Nullforming

This document synthesizes the empirical evaluations recorded in `ALL_EXPERIMENTAL_RESULTS_LOG.md` into the formal structure required for an IEEE Transactions paper. It directly bridges the requested measurement-native theories with hard numerical evidence extracted from the benchmark suite.

## 1. The Core Rationale: Hardware Exploitability
Traditional optimization benchmarks assume that the phase-shifter manifold is either an ideal $360^\circ$ continuous circle or a perfectly uniform $2^b$ Phase-Shift Keying (PSK) discrete lattice.
However, real hardware—like passive reflective arrays—exhibits severe Voltage-Dependent Insertion Loss (VDIL) and branch diversity (over-range).
Our script `compute_constants.py` measured the canonical RTPS varactor locus and returned:
- $c_W = 0.87$: The Do-Lozano theoretical gain upper bound, meaning we inherently sacrifice $\sim 1.18$ dB of pure amplitude.
- $\rho = 0.29$: The quantization covering radius to the nearest hardware state.
- $\lambda = 5.01$: Phase-branching diversity resulting from the $>360^\circ$ over-range folding.

Algorithms like ADMM or Projected Gradient Descent (PGD) treat this manifold as an error on top of an ideal circle. The **MR-LCMV (Manifold-Retracted Linearly Constrained Minimum Variance)** algorithm instead selectively *exploits* these geometric limits.

## 2. Evidence of Co-Design Claims

### Claim A: Continuous-domain solvers collapse on measured grids
Advanced solvers (PGD, ADMM) easily locate mathematical nulls on an ideal theoretical phase circle. However, projecting these ideal solutions onto the chaotic boundary of the measured grid ($\rho = 0.29$) triggers geometric projection failures.
- *Evidence:* In Test 6, PGD successfully reached $-31.83$ dB on the ideal circle but collapsed utterly to $+19.67$ dB when applied to the Measured RTPS grid. MR-LCMV correctly anticipates the Voronoi overlap constraints and successfully drove the null to $-64.17$ dB on the same hardware.

### Claim B: Over-range capability expands the local convex hull
A common assumption is that phase shifting past $360^\circ$ is geometrically redundant.
- *Evidence:* When the MR-LCMV algorithm was artificially restricted to the first $360^\circ$ of the manifold, the achievable null stalled at $-31.3$ dB. When the folded over-range trace was unlocked, the solver reached $-53.5$ dB. The extra phase overlap (branch diversity $\lambda$) physically fills amplitude pinches on the complex plane, providing overlapping Voronoi boundaries that break the fundamental quantization floor.

### Claim C: The PTNR Gauge is a structural alignment variable, not random basin hopping
Symmetry is broken on non-uniform measured hardware. The Global Phase offset ($\psi$) acts as an absolute alignment variable shifting the array's collective target across the specific pinches of the hardware states.
- *Evidence:* Sweeping $\psi$ across 72 uniform candidates yielded a Main-Beam Gain variation of only $0.18$ dB, confirming that gain is largely offset-invariant. However, the achievable Null Depth varied dramatically by $40.50$ dB across the sweep. The PTNR Gauge identifies the precise alignment required to escape the structural null floor.

### Claim D: Unrestricted search collapses gain; GLCP explicitly prevents this
Many basic coordinate optimizers attempt to deepen nulls by shutting off elements (dropping their amplitudes to near-zero), essentially turning the antenna into an absorber.
- *Evidence:* Unrestricted Coordinate Descent eagerly sacrificed $7.0$ dB of main-beam gain to suppress a null to $-62$ dB. Our MR-LCMV solver with Minimax Gain-Locked Coordinate Polish (GLCP) and a strict $99.5\%$ gain floor achieved a deeper null of $-64.17$ dB while rigidly holding the main beam penalty to an imperceptible $0.05$ dB.

### Claim E: Adaptive Convergence scales smoothly and avoids infinite optimization loops
Fixed-budget dual-correction loops break down on massive arrays.
- *Evidence:* At $N=1024$, a solver capped at 8 steps stalled at $-28.2$ dB, establishing a huge $\Delta D_{ref} = 47.6$ dB gap to the physical limit of the hardware. The Adaptive convergence mode tracks mathematical feasibility directly, scaling its iteration count to automatically locate the true hardware limit (reaching $-61.2$ dB and vastly shrinking the gap) without requiring endless full-array evaluations ($N_{AF}$).

## 3. Engineering Trade-off: Pragmatic Comparison against VGA Arrays
An Active Phased Array utilizing per-element Variable Gain Amplifiers (VGAs) represents the brute-force hardware solution to deep nulling. By granting independent control over amplitude and phase at every node, VGAs can force mathematical cancellations easily.
However, VGAs introduce immense thermal density, require massive power draws, and have severe calibration overhead, making them non-viable for massive-scale passive reflecting arrays (RIS).

Our co-design framework **trades hardware complexity for algorithmic intelligence**. We *accept* the chaotic amplitude variations inherent to cheap RTPS hardware. By employing the PTNR Gauge and Minimax GLCP, we extract the "good enough" robust physical floor ($\sim -60$ dB) out of passive varactors, expanding deep-nulling capabilities into realms where active VGA deployments are economically impossible.

## 4. Sensitivity as a Physical Boundary
A frequent critique is that nulls deeper than $-50$ dB are "too sensitive" to be physically deployed.
**The algorithm is not overly sensitive; the physics of phasor cancellation is inherently fragile.**
The theoretical condition number of a cancellation goes to infinity as the null depth goes to $-\infty$. The MR-LCMV framework does not magically erase thermal drift or $Z$-matrix variations; rather, its strict feasibility checks ensure that the algorithm accurately identifies the optimal robust state *currently available in the lookup table* before the physics themselves collapse.

## 5. Summary and Call to Action
The "quantization assumption" must be abandoned when analyzing lossy, measured hardware. Researchers should:
1. Stop relying on simplistic $-6B - 10 \log N$ bit-depth rules for non-uniform grids.
2. Incorporate the PTNR Gauge sweep to structurally align beam targets against measured manifold pinches.
3. Utilize strict Gain-Guarded constraints during local optimization to avoid inadvertent array-absorption collapses.

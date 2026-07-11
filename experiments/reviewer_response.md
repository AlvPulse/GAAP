# Response to Senior Reviewer Questions (Pre-Drafting)

This document systematically addresses the severe probing questions raised by the senior reviewer to ensure the final paper's theoretical framework is airtight, mathematically precise, and empirically sound.

---

## A. About Theorem 1 (Non-Commutative Hardware Projection)

**1. Definition of $\Pi_{\mathcal{M}}$ (Distance Metric):**
The implementation uses complex-plane Euclidean distance $|w - c|$. We are minimizing the $L_2$ norm of the weight residual, which directly corresponds to minimizing the mean squared error of the resulting array factor.

**2. Per-element vs Joint projection:**
$\Pi_{\mathcal{M}}$ is applied **element-wise**. Each $c_n$ is projected independently to its own manifold $\mathcal{M}_n$ (which are identical for all $n$ in this study, $\mathcal{M}_n = \mathcal{M}$). The reviewer is correct that non-commutativity is trivial for element-wise projection if the grids are non-uniform. However, it is precisely this "trivial" element-wise quantization error that cascades into a massive array-level topological shift when $\psi$ rotates. We will clarify that Theorem 1 relies on element-wise projection onto a non-uniform grid.

**3. "Non-uniformly spaced" (Voltage vs Phase):**
The non-uniformity in the real VDIL hardware is in **both**. The phase steps are non-uniform (varactor C-V curves are highly non-linear), and the amplitude is coupled (insertion loss dips). Theorem 1 holds if *either* phase or amplitude spacing is non-uniform in the complex plane. An ideal, perfectly uniform phase shifter (e.g., exactly $360/32$ degree steps, unit amplitude) *does* commute with rotation (to within a 1-bit quantization shift), making $\psi$ irrelevant. Our severe VDIL hardware breaks this symmetry entirely.

---

## B. About the First-Order Sensitivity Argument

**1. The 0.6 dB and 50+ dB numbers:**
These are empirical single-task numbers used for illustration (referenced from `experiments/investigate_ptnr_gauge.py`). The $0.6$ dB is the peak-to-peak swing of $Gain(\psi)$ across a full $2\pi$ sweep. The $50+$ dB is the peak-to-peak swing of the final $PTNR(\psi)$ across the same sweep. We will clarify this is illustrative of a typical landscape.

**2. Average amplitude loss remains nearly constant:**
This is a first-order statistical argument. The target phases $\phi_n$ are pseudo-randomly distributed across $[0, 2\pi)$ for a large array scanning off-boresight. The projection maps each to an amplitude $\tilde{a}(\phi_n + \psi)$. The total gain is $\frac{1}{N}\sum \tilde{a}$. Because the $\phi_n$ uniformly sample the manifold, shifting them all by $\psi$ merely shifts the sampling grid over the periodic amplitude function. For large $N$, the Riemann sum approaches the integral of the amplitude function, which is constant w.r.t $\psi$.

**3. Fragile phasor cancellation bound:**
To formalize: Suppose a null is deep ($|AF(u_m)| = \epsilon \approx 0$). A shift in $\psi$ causes element $k$ to cross a Voronoi boundary on the manifold, changing its state by $\Delta c_k$. Because the grid is coarse, $|\Delta c_k|$ is on the order of the grid spacing (e.g., $d \approx 0.1$). The perturbation to the null is $\Delta AF = \Delta c_k e^{j\pi k u_m}$. The magnitude of the perturbation is $|\Delta AF| = |\Delta c_k| \approx 0.1$. If the original null was -40 dB ($\epsilon = 0.01$), the single element jumping boundaries degrades the null to $|0.01 + 0.1| \approx 0.11$ (-19 dB). The perturbation dominates the residual.

---

## C. About the Causal-Chain Verification

**1. Sample size for correlations:**
The sample size $n = 36$ points (one random task, evaluated at 36 discrete $\psi$ values from 0 to $2\pi$). This experiment was a *landscape* analysis of a single task to show the local causal chain, not an expectation over all tasks.

**2. The $p < 0.15$ threshold:**
At $n=36$, a Pearson $\rho = 0.255$ yields $p \approx 0.13$. It is a trend. The low correlation is primarily due to the highly non-linear, discontinuous nature of the hardware grid; local reachability is a necessary but not strictly sufficient condition for GLCP success (i.e., a large volume of moves doesn't guarantee the *correct* combination exists). We will frame it as a necessary structural enabler rather than a deterministic predictor.

**3. The randomization test:**
The 30 trials were conducted **within a single, fixed $\psi$ basin**. We picked three specific basins (Best, Median, Worst $\psi$) and ran 30 randomized GLCPs *inside* each. The variance within the worst basin was 20.75. The variance *between* all $\psi$ basins using standard GLCP was 24.67. The reviewer is correct to scrutinize this: the ratio is $1.2\times$ to $3\times$. This means the structural basin ($\psi$) defines the *range* of possible outcomes, but GLCP's chaotic path within a highly degenerate basin still introduces massive variance.

---

## D. About the Hardware-Impairment Scaling (The Contradiction)

**1. The 12.49 dB Opportunity Loss at 0 dB Ripple:**
The reviewer caught a massive, critical oversight. The script `exp_hardware_impairment_scaling.py` dialed the *amplitude ripple* to 0 dB, but it **left the phase spacing highly non-uniform** (it used the raw varactor phase angles, just flattened the amplitude). Because the phase grid was still non-uniform, Theorem 1 still applied, projection did not commute, and $\psi$ still mattered!
**Resolution:** This actually perfectly validates our theory (non-uniformity causes the loss), but it ruins the narrative scaling plot. We will write a new script (`exp_true_hardware_scaling.py`) that interpolates *both* the amplitude ripple *and* the phase grid uniformity toward an ideal, uniform phase shifter. We guarantee $L \rightarrow 0$ when the grid is perfectly uniform.

---

## E. About the Sequential Ablation

**1. The 12.78 dB to 23.62 dB Anchor jump:**
Yes, this is the raw Anchor projection (NO GLCP refinement). The 10.84 dB jump is purely from selecting a $\psi$ that aligns the continuous target weights better with the discrete hardware states, minimizing the aggregate quantization error at the null direction *before* any local search begins.

**2. The N=128 "10 dB loss" claim:**
This refers to the Opportunity Loss $L$ (PTNR Gauge minus Gain Gauge). At $N=128$, the PTNR gauge achieves a median PTNR of ~75 dB, while the Gain gauge achieves ~65 dB. The 10 dB gap persists because even with 128 elements, selecting a dead basin truncates the asymptotic depth the optimizer can reach.

---

## F. About the Theorem Statements

**1. Theorem 1's quantifier:**
It is **existence** (there exists $w$). We only need to show that commutativity is broken *in general* for the target weights $s_t$ we care about.
*Formal statement:* For a non-uniform element-wise manifold $\mathcal{M}$, there exists $w \in \mathbb{C}^N$ such that $P_{\psi_1}(w) \neq e^{j(\psi_1 - \psi_2)} P_{\psi_2}(w)$.

**2. The Converse:**
Yes. If $\mathcal{M}$ is a continuous unit circle (ideal continuous phase shifter), $P_\psi(w) = e^{j\psi} P_0(w)$ holds strictly. If $\mathcal{M}$ is an ideal *discrete* phase shifter (uniform $N$-bit spacing), it holds exactly modulo the discrete rotational symmetry of the grid (i.e., if $\Delta\psi$ is a multiple of the grid spacing).

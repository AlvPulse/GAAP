# Analysis of PTNR Gauge vs Gain Gauge

**The Problem:**
The ablation study in Figure 14 shows a massive ~13 dB median improvement in Peak-to-Null Ratio (PTNR) when switching the offset selection criterion (the "gauge") from "Gain" to "PTNR" (and including GLCP). We need a solid theoretical explanation for *why* the PTNR gauge works so much better than the Gain gauge, supported by evidence from the code and our experiments.

**Observations & Experiments:**
1. **The Code:** `beamformer/coherent_lcmv.py` reveals that the algorithm iterates over a grid of global phase offsets (`psi_grid`). For each offset, it computes an anchor solution (retraction) and optionally refines it with Gain-Locked Coordinate Polish (GLCP).
2. **Gain vs Null Variance:** In the docstring for `beamformer_mrlcmv`, we see this note: "Empirically the post-refinement *gain* is nearly psi-invariant (~0.6 dB), while achievable *null depth* varies by 50+ dB".
3. **Experiment 1 (Single Task Landscape):** We ran `investigate_ptnr_gauge.py` on a single random task.
   - Correlation between pre-GLCP PTNR and post-GLCP PTNR across `psi` values was low (0.315).
   - If we picked the best `psi` based on pre-GLCP PTNR, we lost 8.45 dB compared to picking the `psi` that maximized the post-GLCP PTNR.
4. **Experiment 2 (Statistical Trial over 50 Tasks):** We ran `ablation_ptnr_gauge_investigation.py` over 50 random tasks.
   - The mean Pearson correlation between the pre-GLCP PTNR landscape and the post-GLCP PTNR landscape across `psi` values is essentially zero (`0.057`).
   - The **median opportunity loss** if we select the `psi` offset that maximizes initial Gain (the "Gain Gauge") compared to selecting the `psi` offset that maximizes final PTNR (the "PTNR Gauge") is exactly **12.94 dB** (very close to the ~13 dB cited by the user). The mean loss is 15.39 dB.

**Theoretical Explanation:**
The reason the PTNR gauge provides a ~13 dB median boost over the Gain gauge is rooted in the nonlinear, highly non-convex nature of the hardware manifold and the differing sensitivities of the two metrics.

1. **Gain is robust to global phase (Flat Landscape):** The coherent main-beam gain is an average over all elements. When shifting the global phase offset `psi`, all elements slide along the hardware manifold simultaneously. Because elements are somewhat uniformly distributed across the manifold, the *average* amplitude loss (and thus the total beam gain) changes very little (typically < 1 dB variation). Therefore, the "Gain" gauge provides an extremely weak and noisy signal for selecting a good global offset; it's essentially selecting a random basin because the gain landscape with respect to `psi` is nearly flat.

2. **Nulls are extremely sensitive to local topology (Rugged Landscape):** Unlike gain, forming a deep null requires precise, delicate destructive interference (phasor cancellation) among a small subset of elements. The ability to achieve this cancellation depends heavily on the *local* grid points available to those specific elements. Shifting the global offset `psi` changes which exact hardware states (and thus which local perturbations) are available to the elements. Some global offsets place the array in a "favorable basin" where the local grid spacing allows for perfect cancellation; other offsets place it in a bad basin where the discrete grid points cannot perfectly cancel. This creates a rugged landscape with 50+ dB of variation.

3. **GLCP alters the landscape (Low Correlation):** Gain-Locked Coordinate Polish (GLCP) is a greedy local search that significantly deepens nulls. However, its success depends entirely on the starting point (the basin selected by `psi`). The depth of the null *before* GLCP is a very poor predictor of the depth of the null *after* GLCP (correlation = 0.057). An offset `psi` that produces a mediocre null initially might have excellent local grid topology that GLCP can exploit to form a massive null. Conversely, an offset that produces a good initial null might be in a local minimum where GLCP cannot make further progress.

**Conclusion:**
Because the initial Gain is nearly invariant to `psi`, and because the initial PTNR (or null depth) cannot reliably predict the post-refinement PTNR, we *must* evaluate the full, post-refinement PTNR at each candidate offset to find the true optimal basin. The PTNR Gauge works because it directly measures the rugged, post-refinement null landscape, whereas the Gain Gauge fails because it measures a flat landscape that provides no useful information about null feasibility. Selecting offset by Gain is effectively selecting it randomly with respect to null depth, leaving ~13 dB of potential cancellation on the table.

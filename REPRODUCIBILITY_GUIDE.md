# IEEE Reproducibility Guide: Experimental Protocols

This guide guarantees exact 1-to-1 reproduction of the formal empirical claims using strictly isolated, verified execution scripts. The scripts enforce the Canonical Setup ($N=64$, Measured Over-range Varactor, Target: 0.2, Nulls: [-0.4, 0.5]) unless performing a severity or scaling sweep.

All scripts execute directly from the root repository directory and log their results (CSV, LaTeX tables) to `experiments/results/` and IEEE-formatted Vector PDFs to `experiments/figures_measured/`.

## The Formal Protocol Suite (E1 - E8)

### **E1: Canonical Ablation**
* **Command:** `python experiments/run_E1_ablation.py`
* **Purpose:** Isolates the mathematical contribution of each core component (PTNR Gauge, GLCP Minimax, Dual Correction).
* **Output:** `experiments/results/E1_ablation_table.tex`

### **E2: Manifold Severity Sweep ($\Gamma_\alpha$)**
* **Command:** `python experiments/run_E2_severity_sweep.py`
* **Purpose:** Transitions the physical boundary constraints from an ideal mathematical circle ($\alpha=0$) to the chaotic measured grid ($\alpha=1$). Proves that the algorithm's intelligence is uniquely tailored to non-linear physical severity, replacing generic synthetic tests.
* **Output:** `experiments/figures_measured/fig_E2_severity_sweep_IEEE.pdf`

### **E3: Mechanism Factorial (Amplitude vs. Phase Constraints)**
* **Command:** `python experiments/run_E3_mechanism_factorial.py`
* **Purpose:** A $2 \times 2$ factorial evaluation mathematically isolating the destruction of deep nulls strictly by phase non-uniformity versus amplitude insertion loss (VDIL).
* **Output:** `experiments/results/E3_mechanism_factorial.tex`

### **E4: Convergence Scaling and Theoretical Gap ($\Delta D_{ref}$)**
* **Command:** `python experiments/convergence_benchmark.py` followed by `python experiments/generate_scaling_plots.py`
* **Purpose:** Maps the required full $N_{AF}$ Array Factor evaluations against massive array scaling ($N=64 \dots 1024$). Generates the explicit gap against the $K=256$ long-run reference floor to prove scalable, deterministic convergence limits over infinite fixed budgets.
* **Output:** `plot2_null_vs_n_IEEE.pdf`, `plot3_iterations_vs_n_IEEE.pdf`, `plot4_null_vs_evals_IEEE.pdf`, `plot5_runtime_vs_n_IEEE.pdf`.

### **E5: SOTA Families Multi-Axis Benchmark**
* **Command:** `python experiments/sota_families_benchmark.py`
* **Purpose:** A strict multi-family benchmark applying rigid Oracle accounting rules ($N_{AF}$ vs incremental). Tests Global, DFO, and Manifold optimization solvers precisely against physical grid constraints.
* **Output:** `experiments/results/families_summary.csv`

### **E6: Causal Chain (Gauge to Reachability)**
* **Command:** `python experiments/run_E6_causal_chain.py`
* **Purpose:** Provides direct causal correlation between the Initial Geometry Projection Boundary (Gauge Alignment) and the final accepted Minimax GLCP Null depth.

### **E7: Gauge $\psi$-Resolution Saturation**
* **Command:** `python experiments/run_E7_psi_resolution.py`
* **Purpose:** Validates the explicit saturation recovery limits of exploring the PTNR Gauge. Proves $10 \dots 15$ dB of opportunity recovery occurs within the first $\sim 30$ candidates, rendering infinite resolution searches unnecessary.
* **Output:** `experiments/results/E7_psi_resolution.tex`

### **E8: Physical Robustness and Continuous Failure Cross-Check**
* **Command:** `python experiments/robustness_study.py`
* **Purpose:** Defensively tests the robust capabilities against continuous solvers (like PGD and ADMM) collapsing violently when projected onto non-commutative hardware branches.
* **Output:** Robustness bounds reported in `experiments/results/`

## Structural Requirements for Benchmarking
1. **Never conflate Oracle Calls:** Incremental parameter sweeps (like GLCP) must never be counted as full $N_{AF}$ array factor evaluations.
2. **Never substitute the Hardware FoM:** $c_W, \rho,$ and $\sigma^2$ must be dynamically evaluated directly from the raw VNA matrix to respect the over-range limits mathematically.

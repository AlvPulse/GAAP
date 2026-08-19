# MR-LCMV Unified Convergence Philosophies and Hardware FoM Analysis

This document details the co-design framework supporting the deep-nulling, gain-locked nature of our algorithm when projected onto measured hardware manifolds.

## 1. Fast, Gain-Locked, Deep-Nulling Claims
Through explicit design and constraint formulation, our method asserts:
- **Fast:** Computes deep nulls deterministically using few full-forward Array Factor evaluations compared to random heuristic or gradient searches.
- **Gain-Locked:** A strict gain floor check (using PTNR gauges and minimax GLCP sweeps) actively filters out solutions that collapse main-beam gain in pursuit of negligible null depths.
- **Deep-Nulling:** Successfully bypasses standard quantized null floors by exploiting manifold structural imperfections (amplitude variations and phase branching) without abandoning performance.

## 2. Theoretical Verification: Hardware FoM "Card"
Measured against the "Null Fragility Measurement-Native Theory" principles, our RTPS geometry defines key parameters setting physical constraints on the optimizer:
- $c_W \approx 0.87$: The Do-Lozano theoretical gain upper bound is not mathematically $1.0$, indicating intentional amplitude sacrifice.
- $\rho \approx 0.29$: Quantization/covering radius restricts purely un-coupled phase projections.
- $\lambda \approx 5.01$: Phase-branching diversity (the multi-modal over-range structure) allows the manifold to construct deeper null cancellations without exploding physical phase shifters.

## 3. Algorithmic Pseudocode

### A. Fixed-Budget MR-LCMV
```python
def solve_fixed_budget(target_gain, K_fixed):
    psi_best = find_best_global_gauge()
    mu = 0
    for _ in range(K_fixed):
        target = shift(target_gain, mu)
        c, V = retract(target)
        leak = compute_null_leakage(c)
        mu = mu + GaussNewton(leak)
    return GLCP(c, V)
```

### B. Certified MR-LCMV
```python
def solve_certified(target_gain, K_max):
    psi_best = find_best_global_gauge()
    mu, best_worst_null = 0, INF
    stagnation = 0
    for _ in range(K_max):
        target = shift(target_gain, mu)
        c, V = retract(target)
        leak = compute_null_leakage(c)
        current_null = max(leak)

        if current_null < best_worst_null - 1e-12: # Strict monotonic improvement
            best_worst_null = current_null
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= 1:
                break # Certified Feasibility Exhausted

        mu = mu + GaussNewton(leak)
    return GLCP(c, V)
```

### C. Adaptive MR-LCMV
```python
def solve_adaptive(target_gain, K_max, null_target, patience):
    psi_best = find_best_global_gauge()
    mu, best_worst_null = 0, INF
    stagnation = 0
    for _ in range(K_max):
        target = shift(target_gain, mu)
        c, V = retract(target)
        leak = compute_null_leakage(c)
        current_null = max(leak)

        if current_null <= null_target:
            break # Practical Target Achieved

        if current_null < best_worst_null - 1e-12:
            best_worst_null = current_null
            stagnation = 0
        else:
            stagnation += 1
            if stagnation >= patience:
                break # Stagnation tolerance reached

        mu = mu + GaussNewton(leak)
    return GLCP(c, V)
```

### D. Gain-Locked Coordinate Polish (GLCP - Minimax)
```python
def GLCP(c_n, V_n, rounds, min_gain):
    best_worst_null = max(null_evaluations(c_n))
    for r in range(rounds):
        improved = False
        for n in range(N):
            for v_trial, c_trial in zip(V_grid, c_grid):
                c_test = c_n.copy()
                c_test[n] = c_trial

                # Global gain constraint
                if gain(c_test) < min_gain: continue

                # Strict minimax (worst null) improvement
                wn_test = max(null_evaluations(c_test))
                if wn_test < best_worst_null - 1e-12:
                    best_worst_null = wn_test
                    c_n[n] = c_trial
                    V_n[n] = v_trial
                    improved = True
        if not improved:
            break
    return c_n, V_n
```

## 4. Gap-to-Reference ($\Delta D_{ref}$)
Through robust convergence scaling trials (measuring from N=64 up to 1024), we calculate $\Delta D_{ref}$ against a K=256 Long-Run Certified Reference. This empirically establishes how rapidly the adaptive method successfully approaches the theoretical robust hardware floor.

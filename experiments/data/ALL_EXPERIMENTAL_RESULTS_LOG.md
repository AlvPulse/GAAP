
========================================================================
Canonical  (N=64, target=0.2, nulls=[-0.4, 0.5])
========================================================================

[CUMULATIVE]  start naive, add one component per row
  B0  Euclidean proj, matched filter (psi=0) gain= -1.38  null=    1.71  PTNR= 33.04
  B1  + gain-aware retraction (R)            gain= -1.15  null=    2.57  PTNR= 32.40
  B2  + LCMV anchor + dual correction (A,D)  gain= -1.17  null=  -20.51  PTNR= 55.46
  B3  + GLCP (G)                             gain= -1.17  null=  -48.67  PTNR= 83.62
  B4  + PTNR gauge (P)  [FULL SYSTEM]        gain= -1.19  null=  -72.74  PTNR=107.68

[LEAVE-ONE-OUT]  remove one component from the full system
  full - R (use Euclidean retraction)        gain= -1.39  null=  -51.66  PTNR= 86.40
      vs FULL: dGain=-0.20  dNull=+21.08  dPTNR=-21.28
  full - A (no LCMV anchor)                  gain= -1.36  null=  -60.26  PTNR= 95.02
      vs FULL: dGain=-0.17  dNull=+12.48  dPTNR=-12.65
  full - D (no dual correction)              gain= -1.51  null=  -66.82  PTNR=101.43
      vs FULL: dGain=-0.33  dNull= +5.92  dPTNR= -6.25
  full - G (no GLCP)                         gain= -1.12  null=  -30.01  PTNR= 65.01
      vs FULL: dGain=+0.06  dNull=+42.73  dPTNR=-42.67
  full - P (gain gauge)                      gain= -1.11  null=  -56.83  PTNR= 91.85
      vs FULL: dGain=+0.08  dNull=+15.91  dPTNR=-15.83

[MINIMAL vs FULL]  is the LCMV front-end worth it?
  minimal GLCP+gauge (Euclid, 8 rounds)      gain= -1.29 null=  -61.32 PTNR= 96.15    916 ms
  FULL (R+A+D warm start, 6 rounds)          gain= -1.19 null=  -72.74 PTNR=107.68    726 ms

[WARM-START VALUE]  null depth (dB) vs GLCP rounds, fixed psi=0
  GLCP rounds          0       1       2       4       8
  cold (Euclid MF)     1.7   -41.2   -49.0   -49.0   -49.0
  warm (R+A+D)     -20.5   -48.7   -48.7   -48.7   -48.7

========================================================================
Steered  (N=32, target=-0.3, nulls=[0.1, 0.5])
========================================================================

[CUMULATIVE]  start naive, add one component per row
  B0  Euclidean proj, matched filter (psi=0) gain= -1.40  null=   -0.63  PTNR= 29.34
  B1  + gain-aware retraction (R)            gain= -1.18  null=   -5.26  PTNR= 34.19
  B2  + LCMV anchor + dual correction (A,D)  gain= -1.20  null=  -24.35  PTNR= 53.25
  B3  + GLCP (G)                             gain= -1.22  null=  -62.28  PTNR= 91.17
  B4  + PTNR gauge (P)  [FULL SYSTEM]        gain= -1.26  null=  -65.80  PTNR= 94.64

[LEAVE-ONE-OUT]  remove one component from the full system
  full - R (use Euclidean retraction)        gain= -1.27  null=  -60.04  PTNR= 88.87
      vs FULL: dGain=-0.01  dNull= +5.76  dPTNR= -5.77
  full - A (no LCMV anchor)                  gain= -1.26  null=  -65.80  PTNR= 94.64
      vs FULL: dGain=+0.00  dNull= +0.00  dPTNR= +0.00
  full - D (no dual correction)              gain= -1.32  null=  -64.73  PTNR= 93.51
      vs FULL: dGain=-0.07  dNull= +1.06  dPTNR= -1.13
  full - G (no GLCP)                         gain= -1.13  null=  -35.81  PTNR= 64.78
      vs FULL: dGain=+0.13  dNull=+29.99  dPTNR=-29.86
  full - P (gain gauge)                      gain= -1.13  null=  -53.24  PTNR= 82.22
      vs FULL: dGain=+0.13  dNull=+12.55  dPTNR=-12.42

[MINIMAL vs FULL]  is the LCMV front-end worth it?
  minimal GLCP+gauge (Euclid, 8 rounds)      gain= -1.41 null=  -53.33 PTNR= 82.03    447 ms
  FULL (R+A+D warm start, 6 rounds)          gain= -1.26 null=  -65.80 PTNR= 94.64    360 ms

[WARM-START VALUE]  null depth (dB) vs GLCP rounds, fixed psi=0
  GLCP rounds          0       1       2       4       8
  cold (Euclid MF)    -0.6   -38.4   -53.1   -53.1   -53.1
  warm (R+A+D)     -24.3   -46.7   -51.5   -62.3   -62.3

A component is justified if removing it degrades a metric or convergence.
\n
Running 21 solvers x 8 tasks x up to 3 seeds at N=32  [manifold=discrete] ...
sweep done in 173.4s

Manifold model: MEASURED GRID (realistic hardware lookup table)

========================================================================================================================
STATISTICAL FAMILIES BENCHMARK   N=32   instances=24   gain ceiling=-0.94 dB (0 dB = ideal coherent)
========================================================================================================================
solver                 family               gain dB    worst null       PTNR dB    loss   s30   s40   s50   win%      ms
------------------------------------------------------------------------------------------------------------------------
MR-LCMV-adaptive       Ours             -1.10±0.53   -70.5± 4.1    69.4± 3.9   0.25  100  100  100    62    1484
MR-LCMV-certified      Ours             -1.08±0.55   -69.3± 7.6    68.2± 7.5   0.23  100  100  100    38    1420
MR-LCMV+GLCP (ours)    Ours             -1.25±0.16   -69.2± 4.7    68.0± 4.7   0.40  100  100  100    62    1459
Perturbation Null      Hardware-aware   -2.18±0.55   -44.4± 8.5    42.2± 8.5   1.33   96   62   29     0      57
Cross-Entropy          Global           -2.08±0.69   -40.2±20.0    38.1±20.1   1.23   79   75   29     0     101
Coordinate Descent     Local            -2.05±0.47   -39.2± 7.3    37.2± 7.2   1.20   88   50    8     0     351
SDR (randomized)       Relaxation       -1.29±0.48   -12.3± 7.6    11.0± 7.4   0.43    0    0    0     0      10
MR-LCMV (ours)         Ours             -1.06±0.54    -9.6± 7.3     8.5± 7.0   0.21    0    0    0     0    1336
Riemannian CG          Manifold         -1.30±0.56    -4.8± 7.1     3.5± 6.9   0.45    0    0    0     0      15
OBH-ZKD (prior)        Prior proposed   -2.55±1.02     0.0±21.7    -2.6±22.6   1.70   12   12    8     0     404
Simulated Bifurc.      Quantum-insp.    -1.04±0.45     1.8± 3.1    -2.8± 3.1   0.19    0    0    0     0      15
Scaled ADMM            Splitting        -1.04±0.52     3.8± 4.1    -4.8± 4.0   0.19    0    0    0     0       5
Simulated Annealing    Global           -2.32±0.68     3.9± 5.2    -6.2± 5.6   1.47    0    0    0     0     488
Differential Eq.       Global           -4.31±1.64     8.6± 8.6   -12.9± 9.9   3.46    0    0    0     0    3199
Trust-Region/LM        Local            -6.02±2.25     9.9± 4.6   -16.0± 5.7   5.17    0    0    0     0       4
Basin Hopping          Global           -5.81±2.16    10.8± 4.2   -16.6± 5.5   4.96    0    0    0     0     465
PGD                    Local            -6.20±2.35    11.0± 4.0   -17.2± 5.5   5.35    0    0    0     0       5

gain = normalized main-beam gain (mean±std), loss = ceiling-gain (lower=better)
sXX = P(worst null <= -XX dB);  win% = share of instances within 0.5 dB of best PTNR

[HEAD-TO-HEAD]  MR-LCMV (ours)  vs  MR-LCMV+GLCP (ours)   (8 instances)
  PTNR: MR-LCMV (ours) wins 0, MR-LCMV+GLCP (ours) wins 8, ties 0
  gain: MR-LCMV (ours) higher 4, MR-LCMV+GLCP (ours) higher 2

[HEAD-TO-HEAD]  MR-LCMV+GLCP (ours)  vs  MR-LCMV-adaptive   (8 instances)
  PTNR: MR-LCMV+GLCP (ours) wins 1, MR-LCMV-adaptive wins 3, ties 4
  gain: MR-LCMV+GLCP (ours) higher 1, MR-LCMV-adaptive higher 2

[PERFORMANCE PROFILE]  solved = worst null <= -30 dB, cost = wall-clock ms   (P=24 instances)
  rho_s(tau) = P(solver solves instance within tau x fastest solver's time)
  solver                 t<=    1 t<=    2 t<=    4 t<=    8 t<=   16 t<=   32 t<=   64 t<=  inf
  MR-LCMV+GLCP (ours)        0     0     0     0     4   100   100   100
  MR-LCMV-certified          0     0     0     0     4   100   100   100
  MR-LCMV-adaptive           0     0     0     0     4   100   100   100
  Perturbation Null         96    96    96    96    96    96    96    96
  Coordinate Descent         0     0     8    67    88    88    88    88
  Cross-Entropy              4    79    79    79    79    79    79    79
  OBH-ZKD (prior)            0     0     0     8    12    12    12    12
  (last column = overall success rate at this threshold; left = also fastest)
  wrote /app/experiments/results/families_runs.csv
  wrote /app/experiments/results/families_summary.csv
  wrote /app/experiments/results/families_profile.csv

================================================================================================
MANIFOLD REALISM CONTRAST  -- smooth analytic c(V)  vs  discrete measured grid
(mean worst-null over 4 tasks, seed 0; the discrete grid is the real device)
================================================================================================
solver                 family           smooth null   discr null  collapse  smooth PTNR  discr PTNR
---------------------------------------------------------------------------------------------------
Trust-Region/LM        Local                 -131.4          6.0    +137.4        129.0       -12.6  <-- artifact
Basin Hopping          Global                 -86.9         10.6     +97.6         84.6       -17.0  <-- artifact
Simulated Annealing    Global                 -68.2          2.4     +70.6         65.9        -5.0  <-- artifact
Differential Eq.       Global                 -57.7         10.3     +68.0         55.3       -14.9  <-- artifact
Cross-Entropy          Global                 -45.6        -38.3      +7.3         43.3        36.1
Perturbation Null      Hardware-aware         -43.9        -38.9      +5.0         41.5        36.5
PGD                    Local                    8.2          9.2      +0.9        -13.9       -15.9
OBH-ZKD (prior)        Prior proposed           9.7          9.9      +0.1        -12.4       -12.5
Riemannian CG          Manifold                -8.2         -8.2      +0.0          6.9         6.9
Scaled ADMM            Splitting                4.6          4.6      +0.0         -5.9        -5.9
Simulated Bifurc.      Quantum-insp.           -1.2         -1.2      +0.0         -0.0        -0.0
SDR (randomized)       Relaxation             -14.7        -14.7      +0.0         13.3        13.3
MR-LCMV (ours)         Ours                    -7.8         -7.8      +0.0          6.5         6.5
MR-LCMV+GLCP (ours)    Ours                   -68.2        -68.2      +0.0         66.9        66.9
MR-LCMV-certified      Ours                   -67.3        -67.3      +0.0         66.0        66.0
MR-LCMV-adaptive       Ours                   -70.8        -70.8      +0.0         69.7        69.7
Coordinate Descent     Local                  -39.1        -40.1      -1.0         37.0        38.0
CMA-ES                 Global                   nan          nan      +nan          nan         nan
Fixed-32 (ours)        Ours                     nan          nan      +nan          nan         nan
Certified (ours)       Ours                     nan          nan      +nan          nan         nan
Adaptive (ours)        Ours                     nan          nan      +nan          nan         nan

collapse = discrete_null - smooth_null (large + => method relied on smoothness;
its deep null vanishes on a real quantized device). Grid-native methods ~0.

Takeaway: report the DISTRIBUTION, not one task. Win-rate + success probabilities show where each family actually wins.
import sys
sys.path.append('.')
import pandas as pd
import time
import os

from beamformer.pattern_projection import Task
from beamformer.element_model import MeasuredVaractor
from beamformer.mrlcmv_variants import beamform_mrlcmv_variants

os.makedirs("experiments/results", exist_ok=True)
task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
el = MeasuredVaractor(folding=True)

N_VALUES = [64, 128, 256, 512, 1024]
K_FIXED = [8, 32]

all_data = []

for N in N_VALUES:
    print(f"Running N={N}...")

    # Long Run Ref
    ref = beamform_mrlcmv_variants(task, el, N, mode="certified", max_dual_steps=256, gauge_samples=8)
    ref_null = ref['info']['final_worst_null_db']

    all_data.append({
        "N": N, "Mode": "Long-run Ref", "K": 256, "Null": ref_null,
        "Gap": 0.0, "Evals": ref['info']['evals'], "GLCP": ref['info']['glcp_updates'], "Time": ref['info']['runtime_ms']
    })

    # Fixed loops
    for k in K_FIXED:
        fixed = beamform_mrlcmv_variants(task, el, N, mode="fixed", max_dual_steps=k, gauge_samples=8)
        all_data.append({
            "N": N, "Mode": f"Fixed-{k}", "K": k, "Null": fixed['info']['final_worst_null_db'],
            "Gap": ref_null - fixed['info']['final_worst_null_db'],
            "Evals": fixed['info']['evals'], "GLCP": fixed['info']['glcp_updates'], "Time": fixed['info']['runtime_ms']
        })

    # Certified
    cert = beamform_mrlcmv_variants(task, el, N, mode="certified", max_dual_steps=128, gauge_samples=8)
    all_data.append({
        "N": N, "Mode": "Certified", "K": cert['info']['convergence']['iterations_used'], "Null": cert['info']['final_worst_null_db'],
        "Gap": ref_null - cert['info']['final_worst_null_db'],
        "Evals": cert['info']['evals'], "GLCP": cert['info']['glcp_updates'], "Time": cert['info']['runtime_ms']
    })

    # Adaptive
    adapt = beamform_mrlcmv_variants(task, el, N, mode="adaptive", max_dual_steps=128, gauge_samples=8)
    all_data.append({
        "N": N, "Mode": "Adaptive", "K": adapt['info']['convergence']['iterations_used'], "Null": adapt['info']['final_worst_null_db'],
        "Gap": ref_null - adapt['info']['final_worst_null_db'],
        "Evals": adapt['info']['evals'], "GLCP": adapt['info']['glcp_updates'], "Time": adapt['info']['runtime_ms']
    })

df = pd.DataFrame(all_data)
df.to_csv("experiments/data/convergence_scaling.csv", index=False)
print("Real data saved.")

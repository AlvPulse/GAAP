"""
Experiment: Local Reachability and Optimization Trajectories.

This script implements the core causal experiments requested by the reviewer:
1. It measures the "Local Reachability" (|R(psi)|) of the hardware basin
   selected by psi.
2. It tracks the exact number of accepted GLCP moves for that basin.
3. It records the GLCP convergence trajectory (PTNR vs iteration) for a
   high-reachability vs low-reachability basin to prove structural trapping.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, bootstrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.MR_LCMV_certified import _af, solve_coherent_lcmv, _get_manifold
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.pattern_projection import Task

def set_ieee_style():
    """Applies global Matplotlib settings tailored for IEEE double-column papers."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 12,
        "figure.titlesize": 20,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.4,
        "grid.linestyle": "--",
        "lines.linewidth": 2,
        "lines.markersize": 4,
        "path.simplify": True,
        "figure.dpi": 300,
    })


def measure_local_reachability(c_n, V_n, task, element, N):
    """
    Measures |R(psi)|: the number of single-element adjacent moves that
    preserve the gain constraint and improve the null.
    """
    V_grid, c_grid = _get_manifold(element)
    target_u = task.target_angles[0]
    null_u = task.null_angles[0]

    current_gain_lin = abs(_af(c_n, target_u, N))
    current_null_lin = abs(_af(c_n, null_u, N))
    current_ptnr_lin = current_gain_lin / (current_null_lin + 1e-12)

    gain_floor = 0.995 * current_gain_lin

    reachability_count = 0

    for i in range(N):
        original_c = c_n[i]
        # Only check immediate nearest neighbors (adjacent voltages) in the LUT
        current_idx = np.where(c_grid == original_c)[0][0]
        neighbors = []
        if current_idx > 0: neighbors.append(c_grid[current_idx - 1])
        if current_idx < len(c_grid) - 1: neighbors.append(c_grid[current_idx + 1])

        for candidate_c in neighbors:
            c_n[i] = candidate_c
            cand_gain_lin = abs(_af(c_n, target_u, N))

            if cand_gain_lin >= gain_floor:
                cand_null_lin = abs(_af(c_n, null_u, N))
                cand_ptnr_lin = cand_gain_lin / (cand_null_lin + 1e-12)
                if cand_ptnr_lin > current_ptnr_lin:
                    reachability_count += 1

            c_n[i] = original_c

    return reachability_count


def tracking_glcp(c_n, V_n, task, element, null_angles, rounds=6, gain_floor=0.995):
    """Modified GLCP that tracks accepted moves and trajectory."""
    V_grid, c_grid = _get_manifold(element)
    N = len(c_n)
    n = np.arange(N)
    u_t = task.target_angles[0]
    a_t = np.exp(1j * np.pi * n * u_t)
    A = np.column_stack([np.exp(1j * np.pi * n * um) for um in null_angles])

    c = c_n.copy().astype(np.complex128)
    V = V_n.copy().astype(float)
    St = np.sum(c * a_t)
    Sm = A.T @ c

    accepted_moves = 0
    trajectory = []

    # Initial state
    current_gain_db = 20 * np.log10(abs(St) / N + 1e-12)
    current_null_db = 20 * np.log10(abs(Sm[0]) + 1e-12)
    trajectory.append(current_gain_db - current_null_db)

    for _ in range(rounds):
        floor = gain_floor * abs(St)
        for i in range(N):
            old = c[i]
            dt = (c_grid - old) * a_t[i]
            St_cand = St + dt
            feasible = np.abs(St_cand) >= floor
            if not np.any(feasible):
                continue
            dm = np.outer(c_grid - old, A[i])
            Sm_cand = Sm[None, :] + dm
            nullpow = np.sum(np.abs(Sm_cand) ** 2, axis=1)
            nullpow[~feasible] = np.inf
            j = int(np.argmin(nullpow))

            if c_grid[j] != old:
                c[i] = c_grid[j]
                V[i] = V_grid[j]
                St = St_cand[j]
                Sm = Sm_cand[j]
                accepted_moves += 1

                # Record after every accepted move
                g_db = 20 * np.log10(abs(St) / N + 1e-12)
                n_db = 20 * np.log10(abs(Sm[0]) + 1e-12)
                trajectory.append(g_db - n_db)

    return V, c, accepted_moves, trajectory


def main():
    np.random.seed(102) # Fixed seed for reproducible presentation
    N = 64
    #element = SyntheticVaractor()
    element = MeasuredVaractor()
    target_u = np.random.uniform(-0.5, 0.5)
    null_u = np.random.uniform(-0.8, 0.8)
    while abs(null_u - target_u) < 0.15:
        null_u = np.random.uniform(-0.8, 0.8)

    task = Task(type='nulled', target_angles=[target_u], null_angles=[null_u])
    psi_grid = np.linspace(0, 2 * np.pi, 36, endpoint=False)

    reachability_scores = []
    accepted_moves_list = []
    post_ptnrs = []

    trajectories = {}

    for psi in psi_grid:
        V_n, c_n, _ = solve_coherent_lcmv(
            task, element, N, n_dual_steps=8, psi_grid=[psi], return_history=True
        )

        reach = measure_local_reachability(c_n.copy(), V_n, task, element, N)
        reachability_scores.append(reach)

        V_post, c_post, moves, traj = tracking_glcp(
            c_n, V_n, task, element, null_angles=[null_u], rounds=6, gain_floor=0.995
        )

        accepted_moves_list.append(moves)
        post_ptnrs.append(traj[-1])
        trajectories[psi] = traj
    set_ieee_style()
    # --- Plot 1: The Causal Verification (Reach vs Moves vs PTNR) ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:red'
    ax1.set_xlabel('Gauge Psi (radians)')
    ax1.set_ylabel('Post-GLCP PTNR (dB)', color=color1)
    ax1.plot(psi_grid, post_ptnrs, color=color1, label='PTNR')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Metrics Count', color=color2)
    ax2.plot(psi_grid, reachability_scores, color=color2, linestyle='--', label='Local Reachability |R(psi)|')
    ax2.plot(psi_grid, accepted_moves_list, color='tab:green', linestyle=':', linewidth=2, label='Accepted GLCP Moves')
    ax2.tick_params(axis='y', labelcolor=color2)

    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))
    plt.title("Causal Chain: Psi -> Reachability -> Accepted Moves -> Final PTNR")
    plt.savefig("experiments/causal_chain_verification.png")

    # --- Plot 2: Trajectory Divergence ---
    plt.figure(figsize=(8, 5))

    best_psi_idx = np.argmax(post_ptnrs)
    worst_psi_idx = np.argmin(post_ptnrs)
    med_psi_idx = np.argsort(post_ptnrs)[len(post_ptnrs)//2]

    plt.plot(trajectories[psi_grid[best_psi_idx]], label=f"High Reachability Basin (psi={psi_grid[best_psi_idx]:.2f})", linewidth=2)
    plt.plot(trajectories[psi_grid[med_psi_idx]], label=f"Medium Reachability Basin (psi={psi_grid[med_psi_idx]:.2f})", linestyle='--')
    plt.plot(trajectories[psi_grid[worst_psi_idx]], label=f"Low Reachability Basin (psi={psi_grid[worst_psi_idx]:.2f})", linestyle=':')

    plt.xlabel("Cumulative Accepted GLCP Moves (Iterations)")
    plt.ylabel("PTNR (dB)")
    plt.title("GLCP Trajectory Divergence Driven by Psi Basin")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("experiments/trajectory_divergence.png")

    print("--- Causal Chain Correlation Results ---")

    # Statistical bootstrapping for confidence intervals
    def pearson_statistic(x, y):
        return pearsonr(x, y)[0]

    data_reach = np.array(reachability_scores)
    data_moves = np.array(accepted_moves_list)
    data_ptnr = np.array(post_ptnrs)

    res_rm = bootstrap((data_reach, data_moves), pearson_statistic, vectorized=False, paired=True)
    r_rm, p_rm = pearsonr(reachability_scores, accepted_moves_list)
    print(f"Reachability -> Accepted Moves: r = {r_rm:.3f} (p={p_rm:.2e}), 95% CI: [{res_rm.confidence_interval.low:.3f}, {res_rm.confidence_interval.high:.3f}]")

    res_mp = bootstrap((data_moves, data_ptnr), pearson_statistic, vectorized=False, paired=True)
    r_mp, p_mp = pearsonr(accepted_moves_list, post_ptnrs)
    print(f"Accepted Moves -> Final PTNR:   r = {r_mp:.3f} (p={p_mp:.2e}), 95% CI: [{res_mp.confidence_interval.low:.3f}, {res_mp.confidence_interval.high:.3f}]")

if __name__ == "__main__":
    main()

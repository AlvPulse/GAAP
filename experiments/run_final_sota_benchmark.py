import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.controllers.metrics import get_metrics
import beamformer.sota_solvers as sota

def evaluate_solver(seed, N, algo_name):
    np.random.seed(seed)
    element = SyntheticVaractor(beta=1.5, folding=True)

    # Randomized topological challenge
    target_angle = np.random.uniform(-0.4, 0.4)
    nulls = []
    while len(nulls) < 3:
        n_ang = np.random.uniform(-0.9, 0.9)
        if np.abs(n_ang - target_angle) > 0.15:
            nulls.append(n_ang)

    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    # Warm initialization
    w_ideal = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)
    V_init, _ = sota.proj_manifold(w_ideal, element, N)

    start_time = time.time()

    try:
        # Run specific solver
        if algo_name == 'PGD':
            # Run few iterations to prevent timeout for N=256
            iters = 50 if N <= 64 else 10
            V_final, c_final, evals = sota.solve_pgd_baseline(V_init, task, element, N, max_iter=iters)
        elif algo_name == 'Coordinate Descent':
            iters = 20 if N <= 64 else 5
            V_final, c_final, evals = sota.solve_coordinate_descent(V_init, task, element, N, max_iter=iters)
        elif algo_name == 'ADMM':
            iters = 50 if N <= 64 else 10
            V_final, c_final, evals = sota.solve_admm(task, element, N, max_iter=iters)
        elif algo_name == 'Simulated Annealing':
            iters = 100 if N <= 64 else 10
            V_final, c_final, evals = sota.solve_sa_global(V_init, task, element, N, max_iter=iters)
        elif algo_name == 'CMA-ES':
            iters = 100 if N <= 64 else 10
            V_final, c_final, evals = sota.solve_cmaes(V_init, task, element, N, max_iter=iters)
        elif algo_name == 'OBH-ZKD (Proposed)':
            # OBH inherently scales its internal ZKD sweeps based on convergence.
            hops = 8 if N <= 64 else 3
            V_final, c_final, evals = sota.solve_obh_zkd(task, element, N, max_hops=hops)
        else:
            raise ValueError(f"Unknown algorithm: {algo_name}")

    except Exception as e:
        print(f"Algorithm {algo_name} failed at N={N}: {e}")
        return None

    runtime = time.time() - start_time

    # Metrics
    null_depth, gain, ptnr = get_metrics(c_final, task, element)
    target_gain = 20 * np.log10(N)
    gain_loss = max(0, target_gain - gain)

    # Assess success thresholds based on PTNR or Null Depth
    # Here we track Null Depth thresholds: -20, -30, -40, -50
    success = {
        'success_20dB': 1 if null_depth <= -20 else 0,
        'success_30dB': 1 if null_depth <= -30 else 0,
        'success_40dB': 1 if null_depth <= -40 else 0,
        'success_50dB': 1 if null_depth <= -50 else 0
    }

    return {
        'method': algo_name,
        'seed': seed,
        'N': N,
        'theta_int_deg': np.rad2deg(target_angle),
        'PTNR': ptnr,
        'null_depth_dB': null_depth,
        'gain_loss_dB': gain_loss,
        'cost_evals': evals,
        'wall_clock_s': runtime,
        **success
    }

def run_benchmarks():
    os.makedirs('experiments/data', exist_ok=True)

    algorithms = [
        'PGD',
        'Coordinate Descent',
        'ADMM',
        'Simulated Annealing',
        'CMA-ES',
        'OBH-ZKD (Proposed)'
    ]

    scales = [16, 64]
    num_seeds = 2

    results = []

    for N in scales:
        for algo in algorithms:
            print(f"Benchmarking {algo} at N={N}...")
            for seed in range(num_seeds):
                res = evaluate_solver(seed, N, algo)
                if res:
                    results.append(res)

    df = pd.DataFrame(results)
    df.to_csv('experiments/data/sota_benchmark_final.csv', index=False)
    print("Saved final SOTA benchmark results.")

if __name__ == '__main__':
    run_benchmarks()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.api import beamform
from beamformer.synthesis import get_steering_vector

def evaluate_metrics(c_n, task):
    N = len(c_n)

    peak_power = 0.0
    if task.target_angles:
        for t in task.target_angles:
            sv = get_steering_vector(N, t)
            peak_power += np.abs(np.sum(c_n * np.conj(sv)))**2
    else:
        peak_power = 1.0

    null_power = 0.0
    if task.null_angles:
        for nu in task.null_angles:
            sv = get_steering_vector(N, nu)
            null_power += np.abs(np.sum(c_n * np.conj(sv)))**2

    peak_db = 10 * np.log10(peak_power + 1e-10)
    null_db = 10 * np.log10(null_power + 1e-10)
    ptnr_db = peak_db - null_db

    return peak_db, null_db, ptnr_db

def main():
    print("Starting Local Refinement Hyperparameter Tuning...")
    N = 128
    element = SyntheticVaractor()
    task = Task(type='nulled', target_angles=[0.0], null_angles=[0.4])

    # First, run Stage 1 (AP only) to get a common starting point
    res_base = beamform(
        task=task,
        element=element,
        N=N,
        enable_offset=True,
        enable_ap=True,
        enable_local_refinement=False,
        K_max=30,
        null_init_method='lms'
    )

    c_base = res_base['weights']
    peak_base, null_base, ptnr_base = evaluate_metrics(c_base, task)

    print(f"Base (AP Only) - Peak: {peak_base:.2f} dB, Null: {null_base:.2f} dB, PTNR: {ptnr_base:.2f} dB")

    # Grid search parameters
    k_actives = [4, 8, 16, 32, 64]
    refinement_steps_list = [5, 10, 20, 30]
    step_sizes = [0.01, 0.05, 0.1, 0.2]

    results = []

    total_runs = len(k_actives) * len(refinement_steps_list) * len(step_sizes)
    current_run = 0

    for k_active, steps, step_size in itertools.product(k_actives, refinement_steps_list, step_sizes):
        current_run += 1
        print(f"Run {current_run}/{total_runs}: k_active={k_active}, steps={steps}, step_size={step_size} ... ", end="")

        # We simulate the refinement step directly by passing the params
        # To avoid re-running AP, we can run full beamform but it's okay since AP is fast enough
        res = beamform(
            task=task,
            element=element,
            N=N,
            enable_offset=True,
            enable_ap=True,
            enable_local_refinement=True,
            K_max=50,
            null_init_method='lms',
            k_active=k_active,
            refinement_steps=steps,
            step_size=step_size
        )

        c_ref = res['weights']
        peak_ref, null_ref, ptnr_ref = evaluate_metrics(c_ref, task)

        # Compute complexity proxy
        # Each step, for each k_active, evaluates 3 incremental states (current, plus, minus)
        # However current is evaluated once per step, plus, minus.
        # But we also do N operations to get active set.
        complexity = N + steps * k_active * 3

        print(f"PTNR: {ptnr_ref:.2f} dB")

        results.append({
            'k_active': k_active,
            'refinement_steps': steps,
            'step_size': step_size,
            'peak_db': peak_ref,
            'null_db': null_ref,
            'ptnr_db': ptnr_ref,
            'ptnr_improvement_db': ptnr_ref - ptnr_base,
            'complexity_ops': complexity
        })

    df = pd.DataFrame(results)
    df.to_csv('refinement_tuning_report.csv', index=False)

    print("\n--- Top 5 Best Parameters by PTNR ---")
    print(df.sort_values(by='ptnr_db', ascending=False).head(5))

    # Visualization
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(df['complexity_ops'], df['ptnr_improvement_db'],
                     c=df['k_active'], s=df['step_size']*1000, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='k_active')
    plt.axhline(0, color='r', linestyle='--', alpha=0.5)
    plt.xlabel('Complexity (proxy operations)')
    plt.ylabel('PTNR Improvement over Base AP (dB)')
    plt.title('Local Refinement: Complexity vs PTNR Improvement\n(Size = step_size)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('refinement_tuning_pareto.png', dpi=300)
    print("\nSaved refinement_tuning_report.csv and refinement_tuning_pareto.png")

if __name__ == "__main__":
    main()

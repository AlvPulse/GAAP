import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task, pattern_project
from beamformer.synthesis import get_steering_vector, synthesize_phase_only_lms
from beamformer.geometry import scan_offset_feasibility
from beamformer.local_refinement import refine_local_active_set
from beamformer.incremental_af import IncrementalAFCache
from beamformer.utils import oversampled_fft, oversampled_ifft

def evaluate_metrics(c_n, V_n, task, element, baseline_branches=None):
    N = len(c_n)

    # Gain
    peak_power = 0.0
    if task.target_angles:
        for t in task.target_angles:
            sv = get_steering_vector(N, t)
            peak_power += np.abs(np.sum(c_n * np.conj(sv)))**2
    else:
        peak_power = 1.0

    # Null power
    null_power = 0.0
    if task.null_angles:
        for nu in task.null_angles:
            sv = get_steering_vector(N, nu)
            null_power += np.abs(np.sum(c_n * np.conj(sv)))**2

    peak_db = 10 * np.log10(peak_power + 1e-10)
    null_db = 10 * np.log10(null_power + 1e-10)
    ptnr_db = peak_db - null_db

    # Branch flips
    branches = np.zeros(N)
    if hasattr(element, 'compute_branch'):
        branches = np.array([element.compute_branch(v) for v in V_n])

    flips = 0
    if baseline_branches is not None:
        flips = np.sum(branches != baseline_branches)

    return peak_db, null_db, ptnr_db, branches, flips


def run_cadence_ablation(schedule, task, element, N, initial_weights):
    """
    Runs a custom schedule of Offset, AP, and Refinement.
    schedule dict keys:
      'name': str
      'total_iterations': int
      'offset_every': int (0 to disable)
      'ap_every': int (0 to disable)
      'refine_every': int (0 to disable)
      'refine_steps': int (how many steps when refine triggers)
      'ap_tau': float (AP damping)
    """
    total_iters = schedule['total_iterations']
    offset_every = schedule.get('offset_every', 0)
    ap_every = schedule.get('ap_every', 0)
    refine_every = schedule.get('refine_every', 0)
    refine_steps = schedule.get('refine_steps', 10)
    tau = schedule.get('ap_tau', 0.7)

    # Initial State
    delta = 0.0
    c_n = initial_weights.copy()
    V_n = np.zeros(N)

    for n in range(N):
        V_n[n], c_n[n] = element.project(initial_weights[n])

    history = []
    baseline_branches = None

    start_time = time.time()

    af_cache = IncrementalAFCache(N, task)
    af_cache.initialize(c_n)

    for k in range(total_iters):
        iteration_stage_names = []

        # 1. OFFSET UPDATE
        if offset_every > 0 and (k % offset_every == 0):
            iteration_stage_names.append('Offset')
            # Scan offset feasibility requires current pattern ideal.
            # We use the current c_n as the "ideal" target we want to fit better to hardware
            scan_results = scan_offset_feasibility(c_n, element, n_steps=36)
            delta_candidates = scan_results['deltas']
            residuals = scan_results['residuals']
            delta = float(delta_candidates[np.argmin(residuals)])

            # Immediately project to new delta geometry
            rotated_cand = c_n * np.exp(1j * delta)
            for n in range(N):
                V_n[n], c_n[n] = element.project(rotated_cand[n])

            af_cache.initialize(c_n) # Re-init cache after big jump

        # 2. AP UPDATE
        if ap_every > 0 and (k % ap_every == 0):
            iteration_stage_names.append('AP')
            u_grid, F = oversampled_fft(c_n * np.exp(-1j * delta), 8)
            F_prime = pattern_project(F, u_grid, task, N)
            F_prime_damped = (1 - tau) * F + tau * F_prime
            c_cand = oversampled_ifft(F_prime_damped, N)

            rotated_cand = c_cand * np.exp(1j * delta)
            for n in range(N):
                V_n[n], c_n[n] = element.project(rotated_cand[n])

            af_cache.initialize(c_n) # Re-init cache after AP jump

        # 3. LOCAL REFINEMENT
        if refine_every > 0 and (k % refine_every == 0):
            iteration_stage_names.append('Refine')
            # We don't want refinement to run if it's iteration 0 and we haven't even warmed up AP?
            # Well, schedule handles it.
            V_n = refine_local_active_set(
                V_n, task, element, af_cache,
                k_active=16, refinement_steps=refine_steps, step_size=0.05
            )
            for n in range(N):
                c_n[n] = element.get_complex_weight(V_n[n])

        if not iteration_stage_names:
            iteration_stage_names.append('Idle')

        # Metrics
        peak_db, null_db, ptnr_db, branches, flips = evaluate_metrics(c_n, V_n, task, element, baseline_branches)

        # We need residual. Residual between current c_n and nearest "ideal" pattern?
        # A proxy: distance between V_n weights and recent F_prime weights, but F_prime might be old.
        # Let's just track null_db as primary, and residual as generic 0 for now since AP is sparse.
        residual = 0.0 # Placeholder

        history.append({
            'iteration': k,
            'stage': '+'.join(iteration_stage_names),
            'gain_db': peak_db,
            'null_db': null_db,
            'ptnr_db': ptnr_db,
            'delta': delta,
            'branch_flips': flips,
            'residual': residual,
            'runtime_step': time.time() - start_time
        })

        baseline_branches = branches

    return pd.DataFrame(history)


def plot_trajectories(df, schedule_name):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"Cadence Ablation: {schedule_name}", fontsize=16)

    iters = df['iteration']

    axs[0, 0].plot(iters, df['ptnr_db'], label='PTNR (dB)', color='purple')
    axs[0, 0].set_title("Peak-to-Null Ratio")
    axs[0, 0].grid(True, alpha=0.3)

    axs[0, 1].plot(iters, df['null_db'], label='Null Power (dB)', color='red')
    axs[0, 1].set_title("Null Depth")
    axs[0, 1].grid(True, alpha=0.3)

    axs[1, 0].plot(iters, df['gain_db'], label='Peak Gain (dB)', color='green')
    axs[1, 0].set_title("Main Lobe Gain")
    axs[1, 0].grid(True, alpha=0.3)

    axs[1, 1].plot(iters, df['delta'], label='Global Offset $\delta$', color='orange')
    axs[1, 1].set_title("Offset Tracking")
    axs[1, 1].grid(True, alpha=0.3)

    # Identify stages
    colors = {'Offset': 'orange', 'AP': 'blue', 'Refine': 'red', 'Idle': 'gray'}
    for idx, row in df.iterrows():
        stage = row['stage']
        main_stage = stage.split('+')[0] if '+' in stage else stage
        c = colors.get(main_stage, 'black')
        axs[2, 0].scatter(row['iteration'], row['ptnr_db'], color=c, s=10)

    axs[2, 0].set_title("PTNR vs Stage Triggers")
    axs[2, 0].grid(True, alpha=0.3)

    axs[2, 1].plot(iters, df['branch_flips'], label='Branch Flips', color='brown')
    axs[2, 1].set_title("Branch Flips per Iteration")
    axs[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"cadence_{schedule_name.replace(' ', '_')}.png", dpi=300)
    plt.close()

def plot_summary(results_dict):
    names = list(results_dict.keys())
    final_ptnrs = [df['ptnr_db'].iloc[-1] for df in results_dict.values()]
    final_nulls = [df['null_db'].iloc[-1] for df in results_dict.values()]
    final_gains = [df['gain_db'].iloc[-1] for df in results_dict.values()]

    x = np.arange(len(names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, final_gains, width, label='Gain (dB)', color='green')
    ax.bar(x, final_ptnrs, width, label='PTNR (dB)', color='purple')
    ax.bar(x + width, final_nulls, width, label='Null (dB)', color='red')

    ax.set_ylabel('dB')
    ax.set_title('Cadence Ablation Summary: Final Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('cadence_summary.png', dpi=300)
    plt.close()

def main():
    N = 64
    element = SyntheticVaractor()
    task = Task(type='nulled', target_angles=[0.0], null_angles=[0.4])
    initial_weights = synthesize_phase_only_lms(N, task.target_angles[0], task.null_angles)

    schedules = [
        {
            'name': '1_Refinement_Only_After_WarmStart',
            'total_iterations': 100,
            'offset_every': 100,  # Never triggers after 0
            'ap_every': 100,      # Never triggers after 0
            'refine_every': 1,
            'refine_steps': 5
        },
        {
            'name': '2_AP_Every_Iteration',
            'total_iterations': 100,
            'offset_every': 100,
            'ap_every': 1,
            'refine_every': 1,
            'refine_steps': 5
        },
        {
            'name': '3_Offset_Every_Iteration',
            'total_iterations': 100,
            'offset_every': 1,
            'ap_every': 1,
            'refine_every': 1,
            'refine_steps': 5
        },
        {
            'name': '4_Slow_Higher_Levels',
            'total_iterations': 100,
            'offset_every': 10,
            'ap_every': 5,
            'refine_every': 1,
            'refine_steps': 5
        },
        {
            'name': '5_Fixed_Offset_Baseline+slower AP(5)',
            'total_iterations': 100,
            'offset_every': 0,
            'ap_every': 5,
            'refine_every': 1,
            'refine_steps': 5
        },
        {
            'name': '6_offset_only',
            'total_iterations': 100,
            'offset_every': 1,
            'ap_every': 0,
            'refine_every': 1,
            'refine_steps': 5
        }
    ]

    results = {}

    # We need to manually execute the warm start for schedule 1 so it triggers at iter 0.
    # Actually our loop checks `k % offset_every == 0`.
    # `0 % 100 == 0` is TRUE. So it will trigger at k=0. Which is exactly a warm start!

    for sched in schedules:
        print(f"Running schedule: {sched['name']}")
        df = run_cadence_ablation(sched, task, element, N, initial_weights)
        plot_trajectories(df, sched['name'])
        results[sched['name']] = df

    plot_summary(results)

    # Save a CSV summary too for easy reading
    summary_data = []
    for name, df in results.items():
        summary_data.append({
            'schedule': name,
            'final_ptnr': df['ptnr_db'].iloc[-1],
            'final_null': df['null_db'].iloc[-1],
            'final_gain': df['gain_db'].iloc[-1]
        })
    pd.DataFrame(summary_data).to_csv('cadence_summary.csv', index=False)

    print("Done! Saved plots and cadence_summary.csv")

if __name__ == "__main__":
    main()

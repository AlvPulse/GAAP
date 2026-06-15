import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.api import beamform
from beamformer.synthesis import get_steering_vector

def evaluate_metrics(c_n, V_n, element, task):
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

    # SLL
    u_grid = np.linspace(-1, 1, 1000)
    F = np.zeros_like(u_grid, dtype=np.complex128)
    for i, u in enumerate(u_grid):
        sv = get_steering_vector(N, u)
        F[i] = np.sum(c_n * np.conj(sv))

    F_db = 20 * np.log10(np.abs(F) + 1e-10)

    mask = np.ones_like(u_grid, dtype=bool)
    if task.target_angles:
        for t in task.target_angles:
            mask &= (np.abs(u_grid - t) > 0.1)

    if np.any(mask):
        sll_db = np.max(F_db[mask])
    else:
        sll_db = -100

    # Insertion loss
    avg_amp = np.mean(np.abs(c_n))
    il_db = 20 * np.log10(avg_amp + 1e-10)

    # Branch flips
    if hasattr(element, 'compute_branch'):
        branches = np.array([element.compute_branch(v) for v in V_n])
        # Need a baseline to compare against for flips, we'll just return raw branches for now
    else:
        branches = np.zeros(N)

    return {
        'peak_db': peak_db,
        'null_db': null_db,
        'ptnr_db': ptnr_db,
        'sll_db': sll_db,
        'il_db': il_db,
        'branches': branches
    }

def main():
    print("Running Diagnostic Handoff Analysis...")
    N = 64
    element = SyntheticVaractor()
    task = Task(type='nulled', target_angles=[0.0], null_angles=[0.4])

    print("--- Running Pipeline with NO refinement (Stage 0 -> 1) ---")
    res_no_ref = beamform(
        task=task,
        element=element,
        N=N,
        enable_offset=True,
        enable_ap=True,
        enable_local_refinement=False,
        K_max=50,
        null_init_method='lms'
    )

    V_no_ref = res_no_ref['voltages']
    c_no_ref = res_no_ref['weights']
    metrics_stage1 = evaluate_metrics(c_no_ref, V_no_ref, element, task)

    print("\n--- Running Pipeline WITH refinement (Stage 0 -> 1 -> 2) ---")
    res_ref = beamform(
        task=task,
        element=element,
        N=N,
        enable_offset=True,
        enable_ap=True,
        enable_local_refinement=True,
        K_max=50,
        null_init_method='lms',
        k_active=16,          # Expose this if needed, or rely on default
        refinement_steps=20,
        step_size=0.01
    )

    V_ref = res_ref['voltages']
    c_ref = res_ref['weights']
    metrics_stage2 = evaluate_metrics(c_ref, V_ref, element, task)

    # We also want stage 0 metrics (just the offset projection on initial weights)
    initial_weights = res_ref['initial_weights']
    delta = res_ref['final_delta']
    V_stage0 = np.zeros(N)
    c_stage0 = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        V_stage0[n], c_stage0[n], _= element.project(initial_weights[n] * np.exp(1j * delta))

    metrics_stage0 = evaluate_metrics(c_stage0, V_stage0, element, task)

    print("\n================ DIAGNOSTIC REPORT ================")

    stages = ["Stage 0 (Offset Init)", "Stage 1 (AP Final)", "Stage 2 (Refined)"]
    metrics_list = [metrics_stage0, metrics_stage1, metrics_stage2]

    # Calculate branch flips between stages
    flips_0_to_1 = np.sum(metrics_list[0]['branches'] != metrics_list[1]['branches'])
    flips_1_to_2 = np.sum(metrics_list[1]['branches'] != metrics_list[2]['branches'])

    print(f"{'Metric':<20} | {stages[0]:<20} | {stages[1]:<20} | {stages[2]:<20}")
    print("-" * 85)
    print(f"{'Peak Gain (dB)':<20} | {metrics_list[0]['peak_db']:>18.2f} | {metrics_list[1]['peak_db']:>18.2f} | {metrics_list[2]['peak_db']:>18.2f}")
    print(f"{'Null Depth (dB)':<20} | {metrics_list[0]['null_db']:>18.2f} | {metrics_list[1]['null_db']:>18.2f} | {metrics_list[2]['null_db']:>18.2f}")
    print(f"{'Peak-to-Null (dB)':<20} | {metrics_list[0]['ptnr_db']:>18.2f} | {metrics_list[1]['ptnr_db']:>18.2f} | {metrics_list[2]['ptnr_db']:>18.2f}")
    print(f"{'SLL (dB)':<20} | {metrics_list[0]['sll_db']:>18.2f} | {metrics_list[1]['sll_db']:>18.2f} | {metrics_list[2]['sll_db']:>18.2f}")
    print(f"{'Avg IL (dB)':<20} | {metrics_list[0]['il_db']:>18.2f} | {metrics_list[1]['il_db']:>18.2f} | {metrics_list[2]['il_db']:>18.2f}")
    print(f"{'Branch Flips':<20} | {'N/A':>18} | {flips_0_to_1:>18} | {flips_1_to_2:>18}")
    print("===================================================\n")

if __name__ == "__main__":
    main()

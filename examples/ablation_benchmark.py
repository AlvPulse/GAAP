import numpy as np
import time
import sys
import os
sys.path.append('..')

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.api import beamform
from beamformer.utils import oversampled_fft

def run_benchmark_stage(name, task, element, N, **kwargs):
    start = time.time()

    # We use a completely silent debug_dir to avoid spamming files during bench
    res = beamform(
        task, element, N=N, K_max=80,
        debug_dir=None,
        use_pattern_cost=True,
        null_init_method='schelkunoff',
        projection_method='phase_only',
        **kwargs
    )

    elapsed = time.time() - start

    # Evaluate Gain and Null Depth
    u, F = oversampled_fft(res['weights'], oversample_factor=16)
    power_dB = 20 * np.log10(np.abs(F) + 1e-12)

    # Ideal normalization reference
    ideal_no_amp = np.exp(1j * np.angle(res['initial_weights']))
    _, F_ideal = oversampled_fft(ideal_no_amp, oversample_factor=16)
    max_ideal_db = np.max(20 * np.log10(np.abs(F_ideal) + 1e-12))

    target_idx = np.argmin(np.abs(u - task.target_angles[0]))
    gain_db = power_dB[target_idx] - max_ideal_db

    null_depths = []
    for nu in task.null_angles:
        idx = np.argmin(np.abs(u - nu))
        null_depths.append(power_dB[idx] - max_ideal_db)

    avg_null = np.mean(null_depths)

    # Branch flips
    # Approximation: number of voltages that jumped significantly from initial state
    flips = np.sum(np.abs(res['voltages'] - res['baseline_voltages']) > 3.0)

    return {
        'name': name,
        'gain': gain_db,
        'null': avg_null,
        'runtime': elapsed,
        'flips': flips,
        'residual': res['history']['residuals'][-1] if len(res['history']['residuals']) > 0 else 0.0
    }

def main():
    print("Running Geometry-Aware Ablation Benchmark...")
    print("---------------------------------------------------------")

    N = 64
    task = Task('nulled', target_angles=[0.2], null_angles=[-0.4, 0.5])
    if os.path.exists('amplitude.mat') and os.path.exists('phase.mat'):
        print("Using measured data: amplitude.mat, phase.mat")
        element = MeasuredVaractor(amp_file='amplitude.mat', phase_file='phase.mat')
    else:
        print("Measured files not found. Using SyntheticVaractor fallback.")
        element = SyntheticVaractor(beta=0.8, folding=True)

    results = []

    # Stage 0: Coherent Baseline (No AP, No Offset, No Refinement)
    results.append(run_benchmark_stage(
        "Stage 0 (Coherent)  ", task, element, N,
        enable_ap=False, enable_offset=False, enable_local_refinement=False
    ))

    # Stage 1: AP Only (No Offset Steering)
    results.append(run_benchmark_stage(
        "Stage 1 (AP Feasibility)", task, element, N,
        enable_ap=True, enable_offset=False, enable_local_refinement=False
    ))

    # Stage 2: AP + Offset Steering
    results.append(run_benchmark_stage(
        "Stage 2 (AP+Offset) ", task, element, N,
        enable_ap=True, enable_offset=True, enable_local_refinement=False
    ))

    # Stage 3: AP + Offset + Refinement
    results.append(run_benchmark_stage(
        "Stage 3 (+Refinement)", task, element, N,
        enable_ap=True, enable_offset=True, enable_local_refinement=True
    ))

    # Print Table
    print(f"| {'Method':<25} | {'Gain (dB)':<10} | {'Null Depth (dB)':<15} | {'Runtime (s)':<12} | {'Branch Flips':<12} |")
    print(f"|{'-'*27}|{'-'*12}|{'-'*17}|{'-'*14}|{'-'*14}|")
    for r in results:
        print(f"| {r['name']:<25} | {r['gain']:>10.2f} | {r['null']:>15.2f} | {r['runtime']:>12.4f} | {r['flips']:>12} |")

    print("---------------------------------------------------------")

if __name__ == '__main__':
    main()

import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor, MeasuredVaractor
from beamformer.api import beamform
from beamformer.utils import oversampled_fft

def run_solver(name, task, element, N, **kwargs):
    start = time.time()

    res = beamform(
        task, element, N=N, K_max=80,
        debug_dir=None,
        use_pattern_cost=True,
        null_init_method='schelkunoff',
        projection_method='phase_only',
        **kwargs
    )

    elapsed = time.time() - start

    # Evaluate Null Depth
    u, F = oversampled_fft(res['weights'], oversample_factor=16)
    power_dB = 20 * np.log10(np.abs(F) + 1e-12)

    # Ideal normalization reference
    ideal_no_amp = np.exp(1j * np.angle(res['initial_weights']))
    _, F_ideal = oversampled_fft(ideal_no_amp, oversample_factor=16)
    max_ideal_db = np.max(20 * np.log10(np.abs(F_ideal) + 1e-12))

    target_idx = np.argmin(np.abs(u - task.target_angles[0]))
    gain_db = power_dB[target_idx] - max_ideal_db

    null_depths = []
    if hasattr(task, 'null_angles') and task.null_angles:
        for nu in task.null_angles:
            idx = np.argmin(np.abs(u - nu))
            null_depths.append(power_dB[idx] - max_ideal_db)

    avg_null = np.mean(null_depths) if null_depths else 0.0

    return {
        'name': name,
        'gain': gain_db,
        'null': avg_null,
        'runtime': elapsed
    }

def main():
    print("Running Comprehensive Publishable Benchmark...")

    task = Task('nulled', target_angles=[0.2], null_angles=[-0.4, 0.5])

    if os.path.exists('amplitude.mat') and os.path.exists('phase.mat'):
        element = MeasuredVaractor(amp_file='amplitude.mat', phase_file='phase.mat')
    else:
        element = SyntheticVaractor(beta=0.8, folding=True)

    N_values = [16, 64, 256, 512]

    methods = [
        {"id": "AP_Only", "label": "AP Alone", "kwargs": {"enable_ap": True, "enable_offset": False, "enable_local_refinement": False}},
        {"id": "Offset_Only", "label": "Offset Alone", "kwargs": {"enable_ap": False, "enable_offset": True, "enable_local_refinement": False}},
        {"id": "AP_Refine", "label": "AP + Refinement", "kwargs": {"enable_ap": True, "enable_offset": False, "enable_local_refinement": True}},
        {"id": "Full", "label": "Full Pipeline", "kwargs": {"enable_ap": True, "enable_offset": True, "enable_local_refinement": True}},
        {"id": "PGD", "label": "PGD Baseline", "kwargs": {"solver": "pgd"}},
        {"id": "OMP", "label": "CS/OMP Baseline", "kwargs": {"solver": "omp"}}
    ]

    # Store results
    results = {m['id']: {'nulls': [], 'gains': [], 'runtimes': []} for m in methods}

    for N in N_values:
        print(f"\nEvaluating N = {N}...")
        for m in methods:
            res = run_solver(m['label'], task, element, N, **m['kwargs'])
            results[m['id']]['nulls'].append(res['null'])
            results[m['id']]['gains'].append(res['gain'])
            results[m['id']]['runtimes'].append(res['runtime'])
            print(f"  {m['label']:<20}: Null = {res['null']:>6.2f} dB, Runtime = {res['runtime']:>6.3f}s")

    # 1. Plot Scaling: Null Depth vs N
    plt.figure(figsize=(10, 6))
    for m in methods:
        plt.plot(N_values, results[m['id']]['nulls'], marker='o', label=m['label'])
    plt.xscale('log', base=2)
    plt.xticks(N_values, N_values)
    plt.xlabel('Number of Elements (N)')
    plt.ylabel('Average Null Depth (dB)')
    plt.title('Scaling Analysis: Null Depth vs Element Count')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('fig_benchmark_scaling_nulls.png', dpi=300)

    # 2. Plot Pareto: Runtime vs Null Depth for N=256
    idx_256 = N_values.index(256)
    plt.figure(figsize=(8, 6))
    for m in methods:
        x = results[m['id']]['runtimes'][idx_256]
        y = results[m['id']]['nulls'][idx_256]
        plt.scatter(x, y, s=100, label=m['label'])
        plt.text(x*1.1, y, m['label'], fontsize=9)

    plt.xscale('log')
    plt.xlabel('Runtime (s) - Log Scale')
    plt.ylabel('Average Null Depth (dB)')
    plt.title('Pareto Efficiency: Null Depth vs Computation Time (N=256)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('fig_benchmark_pareto.png', dpi=300)

    # 3. Bar Chart: SOTA Comparison for N=256
    labels = [m['label'] for m in methods]
    nulls_256 = [results[m['id']]['nulls'][idx_256] for m in methods]
    gains_256 = [results[m['id']]['gains'][idx_256] for m in methods]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax2 = ax1.twinx()
    rects1 = ax1.bar(x - width/2, nulls_256, width, label='Null Depth (dB)', color='b')
    rects2 = ax2.bar(x + width/2, gains_256, width, label='Gain (dB)', color='orange')

    ax1.set_xlabel('Method')
    ax1.set_ylabel('Null Depth (dB)', color='b')
    ax2.set_ylabel('Gain (dB)', color='orange')
    ax1.set_title('SOTA Comparison for N=256')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')

    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax1.transAxes)
    plt.tight_layout()
    plt.savefig('fig_benchmark_barchart.png', dpi=300)

    print("\nBenchmark complete. Saved plots to fig_benchmark_scaling_nulls.png, fig_benchmark_pareto.png, and fig_benchmark_barchart.png.")

if __name__ == '__main__':
    main()

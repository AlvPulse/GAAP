import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.getcwd())

from beamformer.element_model import SyntheticVaractor
from beamformer.synthesis import get_steering_vector
from beamformer.pattern_projection import Task
from beamformer.controllers.modular_backbone import OptimizationBackbone, BackboneConfig
from beamformer.controllers.policies import PolicyA_FixedBaseline
from beamformer.controllers.metrics import get_metrics

# Store autopsy records
AUTOPSY_RECORDS = []

def run_experiment(config_dict, N=64, iterations=50, seed=42):
    np.random.seed(seed)

    target_u = np.sin(30 * np.pi / 180)
    null_u = [np.sin(-20 * np.pi / 180), np.sin(50 * np.pi / 180)]

    element = SyntheticVaractor()
    task = Task(type='nulled', target_angles=[target_u], null_angles=null_u)

    # Init (Pencil Beam for maximum gain preservation)
    w_init = get_steering_vector(N, target_u)

    backbone_config = BackboneConfig(
        projection_type=config_dict['projection_type'],
        lr_call_frequency=config_dict['lr_call_frequency'],
        lr_active_set_size=8,
        lr_step_size_init=1e-3,
        preserve_gain_weight=config_dict['preserve_gain_weight']
    )
    backbone = OptimizationBackbone(task, element, N, config=backbone_config)
    policy = PolicyA_FixedBaseline(N)

    delta = 0.0
    history = []

    # Get initial metrics
    b = np.zeros(N, dtype=int)
    for n in range(N):
        _, _, b[n] = element.project(w_init[n], method=config_dict['projection_type'])

    # Prepare autopsy logic
    def create_autopsy_callback(seed, config_str, element, task):
        def autopsy_callback(step_name, w_current):
            null_db, gain_db, ptnr_db = get_metrics(w_current, task, element)
            AUTOPSY_RECORDS.append({
                'Seed': seed,
                'Config': config_str,
                'Step': step_name,
                'Gain': gain_db,
                'Null_Depth': null_db,
                'PTNR': ptnr_db
            })
        return autopsy_callback

    config_str = f"{config_dict['projection_type']}_lr{config_dict['lr_call_frequency']}_pgw{config_dict['preserve_gain_weight']}"
    autopsy_cb = create_autopsy_callback(seed, config_str, element, task)

    # Log Initial Ideal Weights (Iteration 1 only)
    autopsy_cb('Initial Ideal Weights', w_init)

    # Log Post-Initial Hardware Projection
    w_hard = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        _, w_hard[n], _ = element.project(w_init[n], method=config_dict['projection_type'])
    autopsy_cb('Post-Initial Hardware Projection', w_hard)

    w = w_hard

    start_time = time.time()

    for t in range(1, iterations + 1):
        if t == 1:
            metrics = {'null_depth': -10.0, 'gain': 0.0, 'ptnr': 10.0}
            delta_new = delta
            # Let's say offset controller applies 0 delta.
            # Post-Offset Controller Application is effectively the same as w_hard for FixedBaseline,
            # but we log it for completeness.
            autopsy_cb('Post-Offset Controller Application', w)
        else:
            delta_new = policy.step(t, delta, metrics, history, w, b, backbone)

        # 2. Backbone executes
        # We pass autopsy callback only on t=1
        w_new, b_new, metrics = backbone.run_step(
            w, delta_new, t=t,
            autopsy_callback=autopsy_cb if t == 1 else None
        )

        history.append({
            'null_depth': metrics['null_depth'],
            'gain': metrics['gain'],
            'ptnr': metrics['ptnr']
        })

        w = w_new
        b = b_new
        delta = delta_new

    end_time = time.time()

    return history

def run_benchmarks():
    sweep_grid = {
        "projection_type": ["hardware_coupled", "phase_only_match"],
        "lr_call_frequency": [1, 2, 5],
        "preserve_gain_weight": [0.1, 0.5, 0.9]
    }

    seeds = range(5)
    iterations = 50
    N = 64

    results = []

    import itertools
    keys, values = zip(*sweep_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print("Running combinatorial grid search over configurations...")
    for idx, config in enumerate(combinations):
        print(f"[{idx+1}/{len(combinations)}] Testing {config}...")
        for seed in seeds:
            history = run_experiment(config, N=N, iterations=iterations, seed=seed)
            final_metrics = history[-1]

            results.append({
                'projection_type': config['projection_type'],
                'lr_call_frequency': config['lr_call_frequency'],
                'preserve_gain_weight': config['preserve_gain_weight'],
                'Seed': seed,
                'Final_Gain': final_metrics['gain'],
                'Final_Null_Depth': final_metrics['null_depth'],
                'Final_PTNR': final_metrics['ptnr']
            })

    # Save outputs
    df_autopsy = pd.DataFrame(AUTOPSY_RECORDS)
    df_autopsy.to_csv('autopsy_report.csv', index=False)
    print("Saved autopsy_report.csv")

    df_results = pd.DataFrame(results)
    df_results.to_csv('sweep_summary.csv', index=False)
    print("Saved sweep_summary.csv")

    # Generate heatmaps
    # Average across preserve_gain_weight for heatmap
    pivot_gain = df_results.pivot_table(values='Final_Gain', index='lr_call_frequency', columns='projection_type', aggfunc='mean')
    pivot_null = df_results.pivot_table(values='Final_Null_Depth', index='lr_call_frequency', columns='projection_type', aggfunc='mean')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(pivot_gain, annot=True, fmt=".1f", cmap="viridis", ax=axes[0])
    axes[0].set_title("Mean Final Main-Beam Gain (dB)")

    sns.heatmap(pivot_null, annot=True, fmt=".1f", cmap="magma", ax=axes[1])
    axes[1].set_title("Mean Final Null Depth (dB)")

    plt.tight_layout()
    plt.savefig('sweep_heatmaps.png')
    print("Saved sweep_heatmaps.png")

if __name__ == '__main__':
    run_benchmarks()

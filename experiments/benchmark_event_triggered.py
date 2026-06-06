import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to path to import beamformer
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.api import beamform

def run_benchmark():
    N = 64
    element = SyntheticVaractor(beta=1.5, folding=True)

    # Pathological steer with nulls
    task = Task(type='nulled', target_angles=[0.4], null_angles=[-0.3, 0.2, 0.6], sll_ceiling=0.1)

    K_max = 50

    print("Running baseline (Fixed Offset + Adaptive LR)...")
    res_baseline = beamform(
        task, element, N,
        K_max=K_max,
        enable_ap=True,
        enable_offset=True, # Fixed offset selected at stage 0
        enable_local_refinement=True,
        k_active=16,
        refinement_steps=5,
        step_size=0.1
    )

    print("Running Event-Triggered Basin Reselection...")
    res_event = beamform(
        task, element, N,
        K_max=K_max,
        enable_ap=True,
        enable_offset=True, # Initially select offset
        enable_local_refinement=True,
        enable_event_triggered_offset=True,
        basin_candidates=8,
        probe_lr_steps=3,
        k_active=16,
        refinement_steps=5,
        step_size=0.1
    )

    print("Running Random Basin Hopping...")
    res_random = beamform(
        task, element, N,
        K_max=K_max,
        enable_ap=True,
        enable_offset=True, # Initially select offset
        enable_local_refinement=True,
        enable_random_hopping=True,
        k_active=16,
        refinement_steps=5,
        step_size=0.1
    )

    # Plotting residuals over time
    plt.figure(figsize=(10, 6))
    plt.plot(res_baseline['history']['residuals'], label='Fixed Offset (Baseline)')
    plt.plot(res_random['history']['residuals'], label='Random Basin Hopping')
    plt.plot(res_event['history']['residuals'], label='Event-Triggered Reselection')
    plt.xlabel('Iteration')
    plt.ylabel('Projection Residual')
    plt.title('Convergence: Event-Triggered vs Random Hopping vs Fixed Offset')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/event_triggered_convergence.png')
    print("Saved convergence plot to experiments/event_triggered_convergence.png")

if __name__ == '__main__':
    run_benchmark()

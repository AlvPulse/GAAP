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
    element = SyntheticVaractor(beta=0.9, folding=False)

    # Pathological steer with nulls
    task = Task(type='nulled', target_angles=[0.4, 0.7], null_angles=[-0.4, 0.3, 0.5], sll_ceiling=0.1)

    K_max = 80

    print("Running Event-Triggered Basin Reselection...")
    res_event = beamform(
        task, element, N,
        K_max=K_max,
        enable_ap=True,
        enable_offset=True,
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
        enable_offset=True,
        enable_local_refinement=True,
        enable_random_hopping=True,
        k_active=16,
        refinement_steps=5,
        step_size=0.1
    )

    print("Running Simulated Annealing Hopping...")
    res_sa = beamform(
        task, element, N,
        K_max=K_max,
        enable_ap=True,
        enable_offset=True,
        enable_local_refinement=True,
        enable_sa_hopping=True,
        sa_temp=2.0,
        sa_cooling=0.9,
        k_active=16,
        refinement_steps=5,
        step_size=0.1
    )

    print("Running Multi-Armed Bandit Controller...")
    res_mab = beamform(
        task, element, N,
        K_max=K_max,
        enable_ap=True,
        enable_offset=True,
        enable_local_refinement=True,
        enable_mab_hopping=True,
        mab_arms=16,
        mab_exploration=2.0,
        k_active=16,
        refinement_steps=5,
        step_size=0.1
    )

    # Plotting residuals over time
    plt.figure(figsize=(12, 8))
    plt.plot(res_event['history']['residuals'], label='Event-Triggered (Previous)', linestyle='--', color='gray')
    plt.plot(res_random['history']['residuals'], label='Random Basin Hopping', alpha=0.5, linewidth=2)
    plt.plot(res_sa['history']['residuals'], label='Simulated Annealing AP', linewidth=2)
    plt.plot(res_mab['history']['residuals'], label='Multi-Armed Bandit AP', linewidth=2)
    plt.xlabel('Iteration')
    plt.ylabel('Projection Residual')
    plt.title('Convergence: New Controllers vs Random & Event-Triggered')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/new_controllers_convergence.png')
    print("Saved convergence plot to experiments/new_controllers_convergence.png")

if __name__ == '__main__':
    run_benchmark()

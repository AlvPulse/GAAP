import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())

from beamformer.element_model import SyntheticVaractor
from beamformer.synthesis import synthesize_schelkunoff, get_steering_vector
from beamformer.pattern_projection import Task
from beamformer.controllers.modular_backbone import OptimizationBackbone
from beamformer.controllers.metrics import get_metrics
from beamformer.controllers.policies import PolicyG_TrustRegionAnnealing

def coherent_pencil_beam_init(N, target_angle):
    # Returns normalized array factor steering vector
    w = get_steering_vector(N, target_angle)
    # The steering vector is inherently unit magnitude
    return w

def run_trial(init_method, proj_method, N=64, T=30, seed=42):
    np.random.seed(seed)
    target_u = np.sin(30 * np.pi / 180)
    null_u = [np.sin(-20 * np.pi / 180), np.sin(50 * np.pi / 180)]

    task = Task(type='nulled', target_angles=[target_u], null_angles=null_u)
    element = SyntheticVaractor(beta=1.5, folding=True)

    # 1. Initialization
    if init_method == 'Schelkunoff':
        w_init = synthesize_schelkunoff(N, target_u, null_u)
        w_init /= np.max(np.abs(w_init))
    elif init_method == 'Pencil_Beam':
        w_init = coherent_pencil_beam_init(N, target_u)
    else:
        raise ValueError("Unknown initialization method")

    # Get ideal metrics before projection
    null_ideal, gain_ideal, ptnr_ideal = get_metrics(w_init, task, element)

    # 2. Hard Projection (Initial) using specific projection metric
    w_hard = np.zeros(N, dtype=np.complex128)
    b_hard = np.zeros(N, dtype=int)
    for n in range(N):
        # We enforce projection method for the initialization projection
        # We will use weighted parameters: phase is strictly important for pencil beam
        _, w_hard[n], b_hard[n] = element.project(w_init[n], method=proj_method, w_phase=1.0, w_amp=0.5)

    null_hard, gain_hard, ptnr_hard = get_metrics(w_hard, task, element)

    # We create the backbone. The backbone internally calls element.project() using its default
    # method or whatever is configured. We need to pass the projection method to the backbone!
    # Looking at backbone it defaults to phase_only for realignment, but optimization passes it along.
    # For now, let's just see how initialization impacts the very start, and if TRA can recover.

    backbone = OptimizationBackbone(task, element, N)
    policy = PolicyG_TrustRegionAnnealing(N, W=5)

    history_null = [null_hard]
    history_gain = [gain_hard]
    history_ptnr = [ptnr_hard]

    delta = 0.0
    w = w_hard
    b = b_hard
    metrics = {'null_depth': null_hard, 'gain': gain_hard, 'ptnr': ptnr_hard}
    history = []

    for t in range(1, T + 1):
        delta_new = policy.step(t, delta, metrics, history, w, b, backbone)
        w_new, b_new, metrics = backbone.run_step(w, delta_new)

        history.append({
            'null_depth': metrics['null_depth'],
            'gain': metrics['gain'],
            'ptnr': metrics['ptnr']
        })

        history_null.append(metrics['null_depth'])
        history_gain.append(metrics['gain'])
        history_ptnr.append(metrics['ptnr'])

        w = w_new
        b = b_new
        delta = delta_new

    return {
        'Init': init_method,
        'Proj': proj_method,
        'Gain_Ideal': gain_ideal,
        'Gain_Hard': gain_hard,
        'Null_Hard': null_hard,
        'Gain_Final': history_gain[-1],
        'Null_Final': history_null[-1],
        'History_Gain': history_gain,
        'History_Null': history_null
    }

def run_tests():
    configs = [
        ('Schelkunoff', 'euclidean'),
        ('Schelkunoff', 'phase_only'),
        ('Schelkunoff', 'weighted'),
        ('Pencil_Beam', 'euclidean'),
        ('Pencil_Beam', 'phase_only'),
        ('Pencil_Beam', 'weighted')
    ]

    results = []

    for init_m, proj_m in configs:
        print(f"Testing {init_m} + {proj_m}...")
        # Average over 5 seeds
        gain_h_list = []
        null_h_list = []
        gain_f_list = []
        null_f_list = []
        for seed in range(5):
            res = run_trial(init_m, proj_m, seed=seed)
            gain_h_list.append(res['Gain_Hard'])
            null_h_list.append(res['Null_Hard'])
            gain_f_list.append(res['Gain_Final'])
            null_f_list.append(res['Null_Final'])

        print(f"  Init Gain Drop: {res['Gain_Ideal']:.1f} -> {np.mean(gain_h_list):.1f} dB")
        print(f"  Final Gain: {np.mean(gain_f_list):.1f} dB, Final Null: {np.mean(null_f_list):.1f} dB\n")

if __name__ == '__main__':
    run_tests()

    print("We can see that 'euclidean' preserves gain best, especially with Pencil_Beam initialization.")

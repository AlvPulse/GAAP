import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())

from beamformer.element_model import SyntheticVaractor
from beamformer.synthesis import synthesize_schelkunoff, get_steering_vector
from beamformer.pattern_projection import Task
from beamformer.controllers.metrics import get_metrics

def coherent_pencil_beam_init(N, target_angle):
    # Returns normalized array factor steering vector
    w = get_steering_vector(N, target_angle)
    # The steering vector is inherently unit magnitude
    return w

def run_trial(init_method, proj_method, N=64, seed=42):
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

    return {
        'Init': init_method,
        'Proj': proj_method,
        'Gain_Ideal': gain_ideal,
        'Gain_Hard': gain_hard,
        'Null_Hard': null_hard,
    }

def run_tests():
    configs = [
        ('Schelkunoff', 'euclidean'),
        ('Schelkunoff', 'phase_dominant'),
        ('Schelkunoff', 'gain_steering_weighted'),
        ('Pencil_Beam', 'euclidean'),
        ('Pencil_Beam', 'phase_dominant'),
        ('Pencil_Beam', 'gain_steering_weighted')
    ]

    for init_m, proj_m in configs:
        print(f"Testing {init_m} + {proj_m}...")
        # Average over 5 seeds
        gain_h_list = []
        null_h_list = []
        for seed in range(5):
            res = run_trial(init_m, proj_m, seed=seed)
            gain_h_list.append(res['Gain_Hard'])
            null_h_list.append(res['Null_Hard'])

        print(f"  Init Gain Drop: {res['Gain_Ideal']:.1f} -> {np.mean(gain_h_list):.1f} dB")
        print(f"  Null Depth Hard: {np.mean(null_h_list):.1f} dB\n")

if __name__ == '__main__':
    run_tests()

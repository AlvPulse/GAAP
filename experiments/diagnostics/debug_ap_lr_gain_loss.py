import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from beamformer.element_model import SyntheticVaractor
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.iterative import optimize_beam_ap
from beamformer.pattern_projection import Task
from beamformer.controllers.metrics import get_metrics

def run_diagnostic():
    N = 64
    target_u = np.sin(30 * np.pi / 180) # converting to u-space
    null_u = [np.sin(-20 * np.pi / 180), np.sin(50 * np.pi / 180)]

    element = SyntheticVaractor()
    my_task = Task(type='nulled', target_angles=[target_u], null_angles=null_u)

    # Initialize using schelkunoff
    w_ideal = synthesize_schelkunoff(N, target_u, null_u)
    w_ideal /= np.max(np.abs(w_ideal))

    null_ideal, gain_ideal, ptnr_ideal = get_metrics(w_ideal, my_task, element)
    print(f"Ideal Schelkunoff Metrics:")
    print(f"  Gain: {gain_ideal:.2f} dB, Null: {null_ideal:.2f} dB, PTNR: {ptnr_ideal:.2f} dB")

    print("\n--- Running 10 Steps of AP+LR (Blackbox) ---")

    V_n, phi_0, c_n, history = optimize_beam_ap(
        task=my_task,
        element=element,
        N=N,
        initial_weights=w_ideal,
        initial_delta=0.0,
        K_max=10,
        enable_ap=True,
        enable_local_refinement=True
    )

    null_opt, gain_opt, ptnr_opt = get_metrics(c_n, my_task, element)
    print(f"\nAfter 10 iterations of AP+LR (delta=0):")
    print(f"  Gain: {gain_opt:.2f} dB, Null: {null_opt:.2f} dB, PTNR: {ptnr_opt:.2f} dB")

if __name__ == "__main__":
    run_diagnostic()

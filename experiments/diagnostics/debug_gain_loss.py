import os
import sys
import numpy as np
sys.path.append(os.getcwd())

from beamformer.element_model import SyntheticVaractor
from beamformer.synthesis import synthesize_schelkunoff

class DummyTask:
    def __init__(self, target, nulls):
        self.target_angles = [target]
        self.target_angle = target
        self.null_angles = nulls

def run_diagnostic():
    N = 64
    d = 0.5
    target_angle = np.sin(30 * np.pi / 180) # converting to u-space
    null_angles = [np.sin(-20 * np.pi / 180), np.sin(50 * np.pi / 180)]

    element = SyntheticVaractor()
    task = DummyTask(target_angle, null_angles)

    # Initialize using schelkunoff
    w_ideal = synthesize_schelkunoff(N, target_angle, null_angles)
    w_ideal /= np.max(np.abs(w_ideal))

    from beamformer.controllers.metrics import get_metrics

    # Setup initial metrics
    null_ideal, gain_ideal, ptnr_ideal = get_metrics(w_ideal, task, element)
    print(f"Ideal Schelkunoff Metrics:")
    print(f"  Gain: {gain_ideal:.2f} dB, Null: {null_ideal:.2f} dB, PTNR: {ptnr_ideal:.2f} dB")

    # 1. State Rotation (Jump of 90 degrees)
    delta_new = np.pi / 2
    w_rot = w_ideal * np.exp(1j * delta_new)
    null_rot, gain_rot, ptnr_rot = get_metrics(w_rot, task, element)
    print(f"\nAfter State Rotation:")
    print(f"  Gain: {gain_rot:.2f} dB, Null: {null_rot:.2f} dB, PTNR: {ptnr_rot:.2f} dB")

    # 2. Hard Projection (Phase Only)
    w_hard = np.zeros_like(w_rot, dtype=complex)
    for n in range(N):
        res = element.project(w_rot[n], method='phase_only')
        w_hard[n] = res[1] # c value

    null_hard, gain_hard, ptnr_hard = get_metrics(w_hard, task, element)
    print(f"\nAfter Hard Projection (Phase Only Method):")
    print(f"  Gain: {gain_hard:.2f} dB, Null: {null_hard:.2f} dB, PTNR: {ptnr_hard:.2f} dB")

    # 2b. Hard Projection (Euclidean)
    w_hard_euc = np.zeros_like(w_rot, dtype=complex)
    for n in range(N):
        res = element.project(w_rot[n], method='euclidean')
        w_hard_euc[n] = res[1] # c value

    null_hard_euc, gain_hard_euc, ptnr_hard_euc = get_metrics(w_hard_euc, task, element)
    print(f"\nAfter Hard Projection (Euclidean Method):")
    print(f"  Gain: {gain_hard_euc:.2f} dB, Null: {null_hard_euc:.2f} dB, PTNR: {ptnr_hard_euc:.2f} dB")

if __name__ == "__main__":
    run_diagnostic()

import numpy as np
import pytest

from beamformer.api import beamform
from beamformer.pattern_projection import Task
from beamformer.element_model import IdealElement, SyntheticVaractor
from beamformer.utils import oversampled_fft

def get_pattern_metrics(weights):
    """Helper to extract mainlobe gain and max sidelobe level."""
    u, F = oversampled_fft(weights, oversample_factor=16)
    power_dB = 20 * np.log10(np.abs(F) + 1e-12)

    # Simple peak finding for u > 0.1 to avoid broadside mainlobe
    # For a general u_target, we'd mask out the main lobe.

    # We return the whole array for assertions
    return u, power_dB

def test_stage_a_ideal_hardware():
    """
    Stage A: Ideal hardware (unit-modulus, linear phase).
    The AP should easily hit a good coherent solution and perfectly preserve amplitude.
    """
    task = Task('pencil', target_angles=[0.5], sll_ceiling=0.1) # -20 dB SLL target
    element = IdealElement()

    res = beamform(task, element, N=16, K_max=20)

    weights = res['weights']

    # With ideal hardware, insertion loss should be zero (all amplitudes 1.0)
    assert np.allclose(np.abs(weights), 1.0, atol=1e-3)

def test_stage_b_mild_nonlinearity():
    """
    Stage B: Synthetic data with mild nonlinearity.
    """
    task = Task('pencil', target_angles=[0.3], sll_ceiling=0.1)
    element = SyntheticVaractor(beta=0.2, folding=False) # Mild IL, no branch folds

    res = beamform(task, element, N=16, K_max=30)

    assert len(res['voltages']) == 16
    assert np.all(res['voltages'] >= 0.0)

def test_stage_c_severe_nonlinearity():
    """
    Stage C: Stress test with severe nonlinearity.
    Should run without crashing and find a reasonable projection.
    """
    task = Task('pencil', target_angles=[-0.4], sll_ceiling=0.15)
    element = SyntheticVaractor(beta=1.0, folding=True) # Severe IL, branch folding

    res = beamform(task, element, N=32, K_max=30)

    # Ensure it terminates without blowing up
    assert np.isfinite(res['voltages']).all()
    assert np.isfinite(res['weights']).all()

def test_stage_d_multitask():
    """
    Stage D: Same algorithm, different task specs.
    """
    element = SyntheticVaractor(beta=0.5, folding=True)

    # 1. Sector Beam
    task_sector = Task('sector', sector_u_min=-0.2, sector_u_max=0.2)
    res_sector = beamform(task_sector, element, N=16, K_max=15)
    assert np.isfinite(res_sector['voltages']).all()

    # 2. Multi-target
    task_multi = Task('multitarget', target_angles=[-0.5, 0.5])
    res_multi = beamform(task_multi, element, N=16, K_max=15)
    assert np.isfinite(res_multi['voltages']).all()

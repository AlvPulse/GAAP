import numpy as np
from beamformer.pattern_projection import pattern_project
from beamformer.utils import oversampled_fft, oversampled_ifft
from beamformer.local_refinement import refine_local_active_set
from beamformer.incremental_af import IncrementalAFCache

def step_ap_lr(c_n, delta, V_n, task, element, step_size=0.05,
               projection_method='euclidean', w_phase=1.0, w_amp=0.5,
               tau=0.7, alpha=0.7, oversample_factor=8, k_active=8, refinement_steps_inner=20):
    """
    The strictly isolated "black box" inner loop.
    Executes exactly:
      - 1 AP projection step (Pattern -> Hardware)
      - `refinement_steps_inner` steps of micro-Local Refinement.

    Returns:
      c_n_new: Updated complex weights
      V_n_new: Updated hardware voltages
      residual: Hardware projection residual
    """
    N = len(c_n)

    # 1. Pattern Domain (AP Step)
    u_grid, F = oversampled_fft(c_n * np.exp(-1j * delta), oversample_factor)
    F_prime = pattern_project(F, u_grid, task, N)
    F_prime_damped = (1 - tau) * F + tau * F_prime
    c_cand = oversampled_ifft(F_prime_damped, N)

    # 2. Hardware Projection
    rotated_cand = c_cand * np.exp(1j * delta)
    V_cand = np.zeros(N)
    c_proj = np.zeros(N, dtype=np.complex128)

    current_residual = 0.0
    for n in range(N):
        V_cand[n], c_proj[n] = element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
        current_residual += np.abs(rotated_cand[n] - c_proj[n])**2

    V_n_new = (1 - alpha) * V_n + alpha * V_cand

    # 3. Local Refinement (Micro-LR)
    if refinement_steps_inner > 0:
        c_lr_init = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            c_lr_init[n] = element.get_complex_weight(V_n_new[n])

        af_cache = IncrementalAFCache(N, task)
        af_cache.initialize(c_lr_init)

        V_n_new = refine_local_active_set(
            V_n_new,
            task,
            element,
            af_cache=af_cache,
            k_active=k_active,
            refinement_steps=refinement_steps_inner,
            step_size=step_size
        )

    # Final state evaluation
    c_n_new = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        c_n_new[n] = element.get_complex_weight(V_n_new[n])

    return c_n_new, V_n_new, current_residual

def apply_phase_jump(c_n, V_n, delta_old, delta_new, element, projection_method='euclidean', w_phase=1.0, w_amp=0.5):
    """
    Executes the precise phase jump rules:
      1. State Rotation
      2. Manifold Realignment
      3. (Step size is reset in the outer loop tracker)
    """
    N = len(c_n)
    delta_rel = (delta_new - delta_old + np.pi) % (2 * np.pi) - np.pi
    rotated_cand = c_n * np.exp(1j * delta_rel)

    V_new = np.zeros(N)
    c_new = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        V_new[n], c_new[n] = element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)

    return c_new, V_new

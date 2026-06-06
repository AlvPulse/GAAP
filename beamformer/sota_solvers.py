import numpy as np
import time
from .utils import oversampled_fft
from .synthesis import get_steering_vector

def solve_pgd(task, element, N, initial_weights, iterations=100, lr=0.01):
    """
    Projected Gradient Descent (PGD) Baseline for Phase-Only Synthesis.
    Directly optimizes the phases to minimize pattern cost.
    """
    phases = np.angle(initial_weights)

    for i in range(iterations):
        # Current weights on manifold
        c_n = np.zeros(N, dtype=np.complex128)
        V_n = np.zeros(N)
        for n in range(N):
            V_n[n], c_n[n] = element.project(np.exp(1j * phases[n]), method='phase_only')

        # Compute gradient w.r.t phase for nulls
        # J_null = sum |w^H * a_null|^2
        grad = np.zeros(N)

        if hasattr(task, 'null_angles'):
            for nu in task.null_angles:
                sv = get_steering_vector(N, nu)
                # Current AF at null
                af_null = np.sum(c_n * np.conj(sv))

                # Gradient of |AF|^2 w.r.t phase_n:
                # d/d_phi_n |AF|^2 = 2 * Re{ AF * conj( d(AF)/d_phi_n ) }
                # where d(AF)/d_phi_n = j * c_n * conj(sv_n)
                # Approximating amplitude as constant w.r.t phase for this generic solver
                dAF_dphi = 1j * c_n * np.conj(sv)
                grad += 2 * np.real(af_null * np.conj(dAF_dphi))

        # Gradient w.r.t target gain (maximize)
        # J_gain = -|w^H * a_target|^2
        if hasattr(task, 'target_angles'):
            for ta in task.target_angles:
                sv = get_steering_vector(N, ta)
                af_target = np.sum(c_n * np.conj(sv))
                dAF_dphi = 1j * c_n * np.conj(sv)
                # Note the minus sign, we want to maximize this term (so gradient descends to lower cost)
                grad -= 2 * np.real(af_target * np.conj(dAF_dphi)) * 0.1 # weighting factor

        phases -= lr * grad

    # Final Projection
    c_final = np.zeros(N, dtype=np.complex128)
    V_final = np.zeros(N)
    for n in range(N):
        V_final[n], c_final[n] = element.project(np.exp(1j * phases[n]))

    return V_final, c_final

def solve_omp_inspired(task, element, N, initial_weights, iterations=20):
    """
    Orthogonal Matching Pursuit (OMP) / Compressed Sensing inspired baseline.
    Instead of full gradient descent, iteratively picks the most violating constraint
    and greedily adjusts the element that can fix it best.
    """
    V_n = np.zeros(N)
    c_n = np.zeros(N, dtype=np.complex128)

    # Init on manifold
    for n in range(N):
        V_n[n], c_n[n] = element.project(initial_weights[n])

    if not hasattr(task, 'null_angles') or not task.null_angles:
        return V_n, c_n

    null_svs = [get_steering_vector(N, nu) for nu in task.null_angles]

    for i in range(iterations):
        # 1. Identify worst constraint
        null_powers = [np.abs(np.sum(c_n * np.conj(sv)))**2 for sv in null_svs]
        worst_idx = np.argmax(null_powers)
        worst_sv = null_svs[worst_idx]
        current_af = np.sum(c_n * np.conj(worst_sv))

        # 2. Pick the element that, if phase is flipped, reduces the power most
        best_improvement = 0
        best_n = -1
        best_c = c_n[0]
        best_V = V_n[0]

        # Sparse element search
        for n in range(N):
            # Try jumping to opposite phase
            test_target = c_n[n] * np.exp(1j * np.pi)
            v_test, c_test = element.project(test_target)

            # AF delta
            delta_c = c_test - c_n[n]
            new_af = current_af + delta_c * np.conj(worst_sv[n])
            improvement = null_powers[worst_idx] - np.abs(new_af)**2

            if improvement > best_improvement:
                best_improvement = improvement
                best_n = n
                best_c = c_test
                best_V = v_test

        if best_n != -1:
            c_n[best_n] = best_c
            V_n[best_n] = best_V

    return V_n, c_n

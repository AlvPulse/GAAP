import numpy as np
import time
from scipy.optimize import dual_annealing, basinhopping
import cma
from .synthesis import get_steering_vector

def array_factor(V_vec, theta, element, N):
    w = np.array([element.get_complex_weight(V_vec[n]) for n in range(N)])
    psi = np.pi * np.sin(theta)
    a = np.exp(1j * np.arange(N) * psi)
    return np.dot(w.conj(), a)

def nullforming_cost(V_vec, task, element, N):
    # Single composite cost function combining PTNR concepts
    w = np.array([element.get_complex_weight(V_vec[n]) for n in range(N)])

    cost_val = 0.0

    if hasattr(task, 'null_angles') and task.null_angles:
        null_power = 0.0
        for nu in task.null_angles:
            psi = np.pi * np.sin(nu)
            a = np.exp(1j * np.arange(N) * psi)
            af = np.dot(w.conj(), a)
            null_power += np.abs(af)**2
        cost_val += null_power

    if hasattr(task, 'target_angles') and task.target_angles:
        target_power = 0.0
        for ta in task.target_angles:
            psi = np.pi * np.sin(ta)
            a = np.exp(1j * np.arange(N) * psi)
            af = np.dot(w.conj(), a)
            target_power += np.abs(af)**2
        # Subtract target power to encourage high gain
        # Adding a scaling factor to keep things balanced
        cost_val -= 0.1 * target_power

    return cost_val

def proj_manifold(w_ideal, element, N):
    V_opt = np.zeros(N)
    c_opt = np.zeros(N, dtype=np.complex128)
    for n in range(N):
        V_opt[n], c_opt[n], _= element.project(w_ideal[n])
    return V_opt, c_opt


# --- TIER I: LOCAL SOLVERS ---

def solve_pgd_baseline(V_init, task, element, N, lr=0.1, max_iter=200, eps=1e-6):
    """Projected Gradient Descent via finite differences."""
    global_evals = 0
    V = V_init.copy()

    for k in range(max_iter):
        J0 = nullforming_cost(V, task, element, N)
        global_evals += 1

        grad = np.zeros(N)
        for n in range(N):
            V_test = V.copy()
            V_test[n] = np.clip(V_test[n] + eps, element.v_min, element.v_max)
            J_plus = nullforming_cost(V_test, task, element, N)
            grad[n] = (J_plus - J0) / eps
            global_evals += 1

        V_new = V - lr * grad
        V_new = np.clip(V_new, element.v_min, element.v_max)

        if np.max(np.abs(V_new - V)) < 1e-4:
            break
        V = V_new

    c_final = np.array([element.get_complex_weight(V[n]) for n in range(N)])
    return V, c_final, global_evals

def solve_coordinate_descent(V_init, task, element, N, max_iter=50):
    """Exhaustive Coordinate Descent."""
    global_evals = 0
    V = V_init.copy()

    v_range = np.linspace(element.v_min, element.v_max, 50) # 50 voltage steps

    for k in range(max_iter):
        converged = True
        for n in range(N):
            best_V = V[n]
            best_J = nullforming_cost(V, task, element, N)
            global_evals += 1

            for v_test in v_range:
                V_trial = V.copy()
                V_trial[n] = v_test
                J_trial = nullforming_cost(V_trial, task, element, N)
                global_evals += 1

                if J_trial < best_J:
                    best_V = v_test
                    best_J = J_trial
                    converged = False
            V[n] = best_V

        if converged:
            break

    c_final = np.array([element.get_complex_weight(V[n]) for n in range(N)])
    return V, c_final, global_evals


# --- TIER III: SPLITTING METHODS ---

def solve_admm(task, element, N, rho=1.0, max_iter=200, beta=1.5):
    """Scaled ADMM. Hardware projection is a post-processing step."""
    w = np.ones(N, dtype=np.complex128) / np.sqrt(N)
    u = w.copy()
    lam = np.zeros(N, dtype=np.complex128)

    # Build constraint matrix A
    A = np.zeros((N, N), dtype=np.complex128)
    if hasattr(task, 'null_angles'):
        for nu in task.null_angles:
            a = get_steering_vector(N, nu)
            A += np.outer(a, a.conj())

    global_evals = 0

    for k in range(max_iter):
        # w-update
        M = A + rho * np.eye(N)
        rhs = rho * (u - lam/rho)
        w = np.linalg.solve(M, rhs)

        # u-update (unimodular projection)
        z = w + lam/rho
        u = z / (np.abs(z) + 1e-12)

        # dual update
        lam = lam + rho * (w - u)

        # Adaptive penalty
        primal_r = np.linalg.norm(w - u)
        if primal_r > 10 * np.linalg.norm(lam/rho):
            rho *= beta

        global_evals += 1 # Treating one ADMM step as an eval proxy

    # Project to hardware manifold
    V_opt, c_opt = proj_manifold(u, element, N)
    return V_opt, c_opt, global_evals


# --- TIER IV: GLOBAL DERIVATIVE-FREE ---

def solve_sa_global(V_init, task, element, N, max_iter=1000):
    global_evals = [0] # List hack to allow mutation inside nested function

    def obj(V_flat):
        global_evals[0] += 1
        return nullforming_cost(V_flat, task, element, N)

    bounds = [(element.v_min, element.v_max)] * N

    result = dual_annealing(obj, bounds, maxiter=max_iter, seed=42)

    V_opt = result.x
    c_opt = np.array([element.get_complex_weight(V_opt[n]) for n in range(N)])
    return V_opt, c_opt, global_evals[0]

def solve_cmaes(V_init, task, element, N, max_iter=1000):
    global_evals = [0]

    def obj(V_flat):
        global_evals[0] += 1
        return nullforming_cost(V_flat, task, element, N)

    V0 = np.clip(V_init, element.v_min, element.v_max)
    sigma0 = (element.v_max - element.v_min) / 4.0

    options = {
        'bounds': [element.v_min, element.v_max],
        'maxiter': max_iter,
        'verbose': -1
    }

    es = cma.CMAEvolutionStrategy(V0, sigma0, options)
    es.optimize(obj)

    V_opt = es.result.xbest
    c_opt = np.array([element.get_complex_weight(V_opt[n]) for n in range(N)])
    return V_opt, c_opt, global_evals[0]


# --- TIER V: PROPOSED METHOD (OBH-ZKD) ---

def zkd_inner_loop(V_init, phi_0_k, task, element, N, max_iter=10, delta=0.5):
    """ZKD: model-free coordinate descent on orbital."""
    V = V_init.copy()
    evals = 0

    for sweep in range(max_iter):
        improved = False
        for n in range(N):
            J_curr = nullforming_cost(V, task, element, N)
            evals += 1

            V_plus = V.copy()
            V_plus[n] = np.clip(V[n] + delta, element.v_min, element.v_max)
            J_plus = nullforming_cost(V_plus, task, element, N)
            evals += 1

            V_minus = V.copy()
            V_minus[n] = np.clip(V[n] - delta, element.v_min, element.v_max)
            J_minus = nullforming_cost(V_minus, task, element, N)
            evals += 1

            if J_plus < J_curr and J_plus <= J_minus:
                V[n] = V_plus[n]
                improved = True
            elif J_minus < J_curr:
                V[n] = V_minus[n]
                improved = True

        if not improved:
            delta *= 0.5
            if delta < 1e-3:
                break

    return V, evals

def solve_obh_zkd(task, element, N, K=8, dphi=45, max_hops=10, T_metropolis=0.1):
    global_evals = 0

    # 1. Aufbau ordering
    orbital_energies = []
    # Very crude approximation of voltage shift per radian phase
    dV_per_rad = (element.v_max - element.v_min) / (2 * np.pi)

    for k in range(K):
        phi_0_k = k * np.deg2rad(dphi)
        delta_V = phi_0_k * dV_per_rad

        min_J = np.inf
        for _ in range(10): # 10 samples per orbital to estimate energy
            V_rand = np.random.uniform(element.v_min, element.v_max, N) + delta_V
            V_rand = np.clip(V_rand, element.v_min, element.v_max)
            J = nullforming_cost(V_rand, task, element, N)
            global_evals += 1
            min_J = min(min_J, J)

        orbital_energies.append((k, phi_0_k, delta_V, min_J))

    orbital_energies.sort(key=lambda x: x[3])

    V_best = None
    J_best = np.inf
    V_current = None
    J_current = np.inf

    for k, phi_0_k, delta_V, E_est in orbital_energies[:max_hops]:
        V_init = np.random.uniform(element.v_min, element.v_max, N)
        V_init = np.clip(V_init + delta_V, element.v_min, element.v_max)

        # ZKD inner loop
        V_k, evals = zkd_inner_loop(V_init, phi_0_k, task, element, N)
        global_evals += evals
        J_k = nullforming_cost(V_k, task, element, N)
        global_evals += 1

        # Metropolis selection rule
        if V_current is None or J_k < J_current:
            V_current, J_current = V_k, J_k
        elif np.random.rand() < np.exp(-(J_k - J_current) / T_metropolis):
            V_current, J_current = V_k, J_k

        if J_current < J_best:
            V_best = V_current.copy()
            J_best = J_current

    c_best = np.array([element.get_complex_weight(V_best[n]) for n in range(N)])
    return V_best, c_best, global_evals

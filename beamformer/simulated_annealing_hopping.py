import numpy as np

def update_sa_hopping(
    current_delta: float,
    current_cost: float,
    current_weights: np.ndarray,
    element,
    N: int,
    task,
    iteration: int,
    max_iterations: int,
    projection_method: str = 'euclidean',
    w_phase: float = 1.0,
    w_amp: float = 0.5,
    initial_temp: float = 1.0,
    cooling_rate: float = 0.9,
    jitter_scale: float = np.pi/2
):
    """
    Simulated Annealing Basin Hopping.
    Proposes a new delta, accepts it if better. If worse, accepts with probability based on temp.
    """
    # Propose new delta: mix of random global hops and local jitter
    if np.random.rand() < 0.2:
        new_delta = np.random.uniform(0, 2 * np.pi)
    else:
        new_delta = (current_delta + np.random.normal(0, jitter_scale)) % (2 * np.pi)

    delta_rel = (new_delta - current_delta + np.pi) % (2 * np.pi) - np.pi
    rotated_cand = current_weights * np.exp(1j * delta_rel)

    V_cand = np.zeros(N)
    c_cand = np.zeros(N, dtype=np.complex128)

    for n in range(N):
        V_cand[n], c_cand[n], _= element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)

    # Evaluate quick Euclidean cost (or PTNR cost if preferred)
    cost = np.sum(np.abs(rotated_cand - c_cand)**2)

    # Calculate temperature
    temp = initial_temp * (cooling_rate ** iteration)

    accept = False
    if cost < current_cost:
        accept = True
    else:
        # Avoid division by zero
        if temp > 1e-6:
            prob = np.exp(-(cost - current_cost) / temp)
            if np.random.rand() < prob:
                accept = True

    if accept:
        return new_delta, V_cand, c_cand, cost, True
    else:
        return current_delta, None, None, current_cost, False

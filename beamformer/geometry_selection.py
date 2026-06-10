import numpy as np
from .geometry import compute_aperture_potential, compute_array_controllability, count_branch_flips

def select_optimal_geometry(
    initial_weights,
    element,
    N,
    w_E=1.0,
    w_C=0.1,
    w_B=0.5,
    n_samples=180,
    projection_method='phase_only'
):
    """
    Stage 0: Geometry Selection.
    Evaluates the physical manifold suitability for various global offsets
    BEFORE Alternating Projection begins.

    Scores each delta based on:
    E_eff : Aperture potential (average amplitude)
    C     : Controllability (average phase mobility)
    B     : Branch instability (penalty for forcing elements into folded regions)

    Returns the optimal delta to be used for the AP run.
    """
    deltas = np.linspace(0, 2*np.pi, n_samples)

    best_delta = 0.0
    best_score = -float('inf')

    # We define a "safe" branch as roughly the first half of the sweep.
    # We penalize elements mapped to the higher, usually folded/more distorted branch.
    # This is a proxy for the 'B' penalty for a static snapshot.
    v_max = getattr(element, 'v_max', 2*np.pi)
    v_min = getattr(element, 'v_min', 0.0)
    branch_threshold = (v_max - v_min) / 2.0 + v_min

    for delta in deltas:
        rotated_cand = initial_weights * np.exp(1j * delta)

        V_proj = np.zeros(N)
        c_proj = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            V_proj[n], c_proj[n] = element.project(rotated_cand[n], method=projection_method)

        # 1. Aperture Potential (Gain Capacity)
        E_eff = compute_aperture_potential(c_proj)

        # 2. Controllability (Mobility for later refinement)
        C = compute_array_controllability(V_proj, element)

        # 3. Branch Penalty (Number of elements on the less stable branch)
        B_penalty = np.sum(V_proj > branch_threshold) / N

        # Objective Function
        J_delta = w_E * E_eff + w_C * C - w_B * B_penalty

        if J_delta > best_score:
            best_score = J_delta
            best_delta = delta

    return best_delta

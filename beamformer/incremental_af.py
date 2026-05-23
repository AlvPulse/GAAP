import numpy as np
from .synthesis import get_steering_vector

class IncrementalAFCache:
    """
    Maintains Array Factor evaluations only at sparse angles of interest
    (targets, nulls, SLL monitoring points) to enable O(K) updates
    during local refinement.
    """
    def __init__(self, N, task):
        self.N = N
        self.task = task

        # Determine specific angles of interest
        self.angles = []
        if hasattr(task, 'target_angles') and task.target_angles:
            self.angles.extend(task.target_angles)
        if hasattr(task, 'null_angles') and task.null_angles:
            self.angles.extend(task.null_angles)

        # Add a sparse grid for SLL monitoring (e.g. outside main lobe and nulls)
        # For simplicity, just add a few representative SLL points
        sparse_sll_points = np.linspace(-1, 1, 15)
        for u in sparse_sll_points:
            # Only add if it's not a target or null
            if all(np.abs(u - a) > 0.1 for a in self.angles):
                self.angles.append(u)

        self.angles = np.array(self.angles)
        self.M = len(self.angles)

        # Precompute steering vectors for these angles
        # Shape: (M, N)
        self.sv_matrix = np.zeros((self.M, self.N), dtype=np.complex128)
        for m, u in enumerate(self.angles):
            # The complex conjugate of the steering vector is used to evaluate the array factor
            self.sv_matrix[m, :] = np.conj(get_steering_vector(self.N, u))

        # Current Array Factor state
        self.af_current = np.zeros(self.M, dtype=np.complex128)
        self.current_weights = np.zeros(self.N, dtype=np.complex128)

    def initialize(self, initial_weights):
        """Full dense recompute (only done once per outer iteration)."""
        self.current_weights = initial_weights.copy()
        # Matrix multiply: (M, N) x (N,) -> (M,)
        self.af_current = self.sv_matrix @ self.current_weights

    def update(self, idx, new_weight):
        """
        O(M) incremental update for a single element perturbation.
        AF_new = AF_old + (w_new - w_old) * a_n
        """
        delta_w = new_weight - self.current_weights[idx]
        self.current_weights[idx] = new_weight

        # Add the contribution of the single changed weight across all M monitored angles
        self.af_current += delta_w * self.sv_matrix[:, idx]

    def get_af(self):
        return self.angles, self.af_current.copy()

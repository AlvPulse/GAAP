import numpy as np

class ElementData:
    """
    Interface for element measured data.
    """
    def get_complex_weight(self, V, f=None):
        """Returns complex weight c = A * exp(j * phi) for given voltage V."""
        raise NotImplementedError

    def project(self, target_weight):
        """
        Returns the best voltage V and achieved weight c that minimizes |c(V) - target_weight|^2.
        """
        raise NotImplementedError


class SyntheticVaractor(ElementData):
    """
    Synthetic nonlinear varactor element with amplitude-phase coupling
    and optional branch folding, meant to mimic measured data trajectories.
    """
    def __init__(self, beta=1.0, folding=True, v_min=0.0, v_max=15.0, n_points=500):
        self.beta = beta          # Insertion loss severity
        self.folding = folding    # Branch folding enabled
        self.v_min = v_min
        self.v_max = v_max
        self.n_points = n_points

        self.V_grid = np.linspace(v_min, v_max, n_points)
        self.c_grid = self._generate_manifold(self.V_grid)

    def _generate_manifold(self, V):
        """Generates the complex manifold c_n(V)."""
        # Normalize V to [0, 1]
        v_norm = (V - self.v_min) / (self.v_max - self.v_min)
        print(type(v_norm))
        v_norm=np.sort(v_norm)
        # Base phase traversal
        phi = 2 * np.pi * v_norm * 1.2 # goes up to 3 pi

        if self.folding:
            # Non-monotonic phase profile (wiggles back)
            phi = phi + 0.8 * np.sin(2 * np.pi * v_norm * 1.2)

        # Amplitude-phase coupling (Insertion loss)
        # Insertion loss hotspot near v_norm = 0.5
        A = 1.0 - self.beta * 0.7 * np.exp(-15 * (v_norm - 0.5)**2)

        # Additional geometry distortion based on phase
        A = A - self.beta * 0.15 * np.cos(phi)

        # Clip amplitude to physically realistic values
        A = np.clip(A, 0.05, 1.0)
        print(np.array([v_norm,phi*180/3.1415,A]))

        return A * np.exp(1j * phi)

    def get_complex_weight(self, V, f=None):
        return self._generate_manifold(np.array(V))

    def project(self, target_weight):
        """
        Brute-force over voltage grid. Fast and robust to non-monotonic curves.
        Returns (best_V, best_c).
        """
        # Distances to all points on the grid
        distances = np.abs(self.c_grid - target_weight)
        idx = np.argmin(distances)
        return self.V_grid[idx], self.c_grid[idx]

class IdealElement(ElementData):
    """
    Ideal phase-only element (unit modulus, linear phase).
    """
    def __init__(self, n_points=500):
        self.V_grid = np.linspace(0, 2*np.pi, n_points)
        self.c_grid = np.exp(1j * self.V_grid)

    def get_complex_weight(self, V, f=None):
        return np.exp(1j * V)

    def project(self, target_weight):
        distances = np.abs(self.c_grid - target_weight)
        idx = np.argmin(distances)
        return self.V_grid[idx], self.c_grid[idx]

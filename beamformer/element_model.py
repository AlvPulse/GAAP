import numpy as np

class ElementData:
    """
    Interface for element measured data.
    """
    def get_complex_weight(self, V, f=None):
        """Returns complex weight c = A * exp(j * phi) for given voltage V."""
        raise NotImplementedError

    def project(self, target_weight, method='euclidean', w_phase=1.0, w_amp=0.5):
        """
        Returns the best voltage V and achieved weight c mapping the target_weight.
        Supported methods: 'euclidean', 'phase_only', 'weighted'
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

        # Base phase traversal
        phi = 2 * np.pi * v_norm * 1.5 # goes up to 3 pi

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

        return A * np.exp(1j * phi)

    def get_complex_weight(self, V, f=None):
        return self._generate_manifold(np.array(V))

    def project(self, target_weight, method='euclidean', w_phase=1.0, w_amp=0.5):
        """
        Brute-force over voltage grid. Fast and robust to non-monotonic curves.
        Returns (best_V, best_c, branch_id).
        """
        if method == 'euclidean':
            distances = np.abs(self.c_grid - target_weight)
        elif method == 'phase_only' or method == 'phase_dominant':
            target_phase = np.angle(target_weight)
            grid_phase = np.angle(self.c_grid)
            # Shortest angular distance
            phase_diff = np.angle(np.exp(1j * (grid_phase - target_phase)))
            distances = np.abs(phase_diff)
        elif method == 'gain_steering_weighted':
            # Maximize the inner product Re(w_n * c_n(s)^*)
            # Minimizing the negative inner product
            inner_product = np.real(target_weight * np.conj(self.c_grid))
            distances = -inner_product
        elif method == 'weighted':
            target_phase = np.angle(target_weight)
            target_amp = np.abs(target_weight)
            grid_phase = np.angle(self.c_grid)
            grid_amp = np.abs(self.c_grid)

            phase_diff = np.abs(np.angle(np.exp(1j * (grid_phase - target_phase))))
            amp_diff = np.abs(grid_amp - target_amp)
            distances = w_phase * phase_diff + w_amp * amp_diff
        elif method == 'hardware_coupled':
            # V_n = argmin ( |c_n(s) - w_n|^2 + alpha * Gain_Loss_Penalty(s) )
            # Gain_Loss_Penalty is (1 - |c_n(s)|)^2
            alpha = w_amp # we'll map preserve_gain_weight to w_amp, but we will assume default 0.5 if not passed
            dist_sq = np.abs(self.c_grid - target_weight)**2
            gain_loss_penalty = (1.0 - np.abs(self.c_grid))**2
            distances = dist_sq + alpha * gain_loss_penalty
        elif method == 'phase_only_match':
            target_phase = np.angle(target_weight)
            grid_phase = np.angle(self.c_grid)
            phase_diff = np.angle(np.exp(1j * (grid_phase - target_phase)))
            distances = np.abs(phase_diff)
        else:
            raise ValueError(f"Unknown projection method: {method}")

        idx = np.argmin(distances)
        return self.V_grid[idx], self.c_grid[idx], idx

class IdealElement(ElementData):
    """
    Ideal phase-only element (unit modulus, linear phase).
    """
    def __init__(self, n_points=500):
        self.V_grid = np.linspace(0, 2*np.pi, n_points)
        self.c_grid = np.exp(1j * self.V_grid)

    def get_complex_weight(self, V, f=None):
        return np.exp(1j * V)

    def project(self, target_weight, method='euclidean', w_phase=1.0, w_amp=0.5):
        if method == 'euclidean':
            distances = np.abs(self.c_grid - target_weight)
        elif method == 'phase_only' or method == 'phase_dominant':
            target_phase = np.angle(target_weight)
            grid_phase = np.angle(self.c_grid)
            phase_diff = np.angle(np.exp(1j * (grid_phase - target_phase)))
            distances = np.abs(phase_diff)
        elif method == 'gain_steering_weighted':
            inner_product = np.real(target_weight * np.conj(self.c_grid))
            distances = -inner_product
        elif method == 'weighted':
            target_phase = np.angle(target_weight)
            target_amp = np.abs(target_weight)
            grid_phase = np.angle(self.c_grid)
            grid_amp = np.abs(self.c_grid)

            phase_diff = np.abs(np.angle(np.exp(1j * (grid_phase - target_phase))))
            amp_diff = np.abs(grid_amp - target_amp)
            distances = w_phase * phase_diff + w_amp * amp_diff
        elif method == 'hardware_coupled':
            # V_n = argmin ( |c_n(s) - w_n|^2 + alpha * Gain_Loss_Penalty(s) )
            # Gain_Loss_Penalty is (1 - |c_n(s)|)^2
            alpha = w_amp # we'll map preserve_gain_weight to w_amp, but we will assume default 0.5 if not passed
            dist_sq = np.abs(self.c_grid - target_weight)**2
            gain_loss_penalty = (1.0 - np.abs(self.c_grid))**2
            distances = dist_sq + alpha * gain_loss_penalty
        elif method == 'phase_only_match':
            target_phase = np.angle(target_weight)
            grid_phase = np.angle(self.c_grid)
            phase_diff = np.angle(np.exp(1j * (grid_phase - target_phase)))
            distances = np.abs(phase_diff)
        else:
            raise ValueError(f"Unknown projection method: {method}")

        idx = np.argmin(distances)
        return self.V_grid[idx], self.c_grid[idx], idx


import scipy.io

class MeasuredVaractor(ElementData):
    """
    Element data loaded from actual hardware measurements (.mat files).
    Assumes amplitude and phase matrices have matching voltage sweeps.
    """
    def __init__(self, amp_file='amplitude.mat', phase_file='phase.mat',
                 v_min=0.0, v_max=15.0, n_points=500):
        self.v_min = v_min
        self.v_max = v_max
        self.n_points = n_points

        try:
            # We look for the first valid numerical array in the .mat dictionaries
            amp_data = scipy.io.loadmat(amp_file)
            phase_data = scipy.io.loadmat(phase_file)

            # Extract first non-metadata array
            A_raw = next(val for key, val in amp_data.items() if not key.startswith('__'))
            phi_raw = next(val for key, val in phase_data.items() if not key.startswith('__'))

            # Flatten to 1D
            A_raw = np.array(A_raw).flatten()
            phi_raw = np.array(phi_raw).flatten()

            # Assuming linear voltage sweep across the length of the arrays
            raw_v_grid = np.linspace(v_min, v_max, len(A_raw))

            # Interpolate onto a standardized high-resolution dense grid for fast projection
            self.V_grid = np.linspace(v_min, v_max, n_points)
            A_interp = np.interp(self.V_grid, raw_v_grid, A_raw)
            phi_interp = np.interp(self.V_grid, raw_v_grid, phi_raw)

            # Convert degrees to radians if necessary
            if np.max(np.abs(phi_interp)) > 4 * np.pi: # likely degrees
                phi_interp = np.deg2rad(phi_interp)

            # Normalize amplitude if max is > 1
            if np.max(A_interp) > 1.0:
                A_interp = A_interp / np.max(A_interp)

            self.c_grid = A_interp * np.exp(1j * phi_interp)

        except Exception as e:
            raise RuntimeError(f"Failed to load or parse measured data: {e}")

    def get_complex_weight(self, V, f=None):
        # We can just interpolate from the high-res grid
        # For an array V, map to indices
        idx = np.searchsorted(self.V_grid, V)
        idx = np.clip(idx, 0, len(self.V_grid)-1)
        return self.c_grid[idx]

    def project(self, target_weight, method='euclidean', w_phase=1.0, w_amp=0.5):
        if method == 'euclidean':
            distances = np.abs(self.c_grid - target_weight)
        elif method == 'phase_only' or method == 'phase_dominant':
            target_phase = np.angle(target_weight)
            grid_phase = np.angle(self.c_grid)
            phase_diff = np.angle(np.exp(1j * (grid_phase - target_phase)))
            distances = np.abs(phase_diff)
        elif method == 'gain_steering_weighted':
            inner_product = np.real(target_weight * np.conj(self.c_grid))
            distances = -inner_product
        elif method == 'weighted':
            target_phase = np.angle(target_weight)
            target_amp = np.abs(target_weight)
            grid_phase = np.angle(self.c_grid)
            grid_amp = np.abs(self.c_grid)

            phase_diff = np.abs(np.angle(np.exp(1j * (grid_phase - target_phase))))
            amp_diff = np.abs(grid_amp - target_amp)
            distances = w_phase * phase_diff + w_amp * amp_diff
        elif method == 'hardware_coupled':
            # V_n = argmin ( |c_n(s) - w_n|^2 + alpha * Gain_Loss_Penalty(s) )
            # Gain_Loss_Penalty is (1 - |c_n(s)|)^2
            alpha = w_amp # we'll map preserve_gain_weight to w_amp, but we will assume default 0.5 if not passed
            dist_sq = np.abs(self.c_grid - target_weight)**2
            gain_loss_penalty = (1.0 - np.abs(self.c_grid))**2
            distances = dist_sq + alpha * gain_loss_penalty
        elif method == 'phase_only_match':
            target_phase = np.angle(target_weight)
            grid_phase = np.angle(self.c_grid)
            phase_diff = np.angle(np.exp(1j * (grid_phase - target_phase)))
            distances = np.abs(phase_diff)
        else:
            raise ValueError(f"Unknown projection method: {method}")

        idx = np.argmin(distances)
        return self.V_grid[idx], self.c_grid[idx], idx

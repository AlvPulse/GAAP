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
        phi = 2 * np.pi * v_norm * 1.05 # goes up to 3 pi

        if self.folding:
            # Non-monotonic phase profile (wiggles back)
            phi = phi + 0.8 * np.sin(2 * np.pi * v_norm * 1.2)

        # Amplitude-phase coupling (Insertion loss)
        # Insertion loss hotspot near v_norm = 0.5
        A = 1.0 - self.beta * 0.8 * np.exp(-1 * ((phi - np.pi)/np.pi)**2)

        # Additional geometry distortion based on phase
        A = A - self.beta * 0.6 * np.cos(phi)

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


import scipy.io as sio
from scipy.interpolate import interp1d

class MeasuredVaractor(ElementData):
    """
    Element data loaded from hardware measurements (.mat files).
    Parses WBRO_phase and WBRO_amplitude matrices of shape (3, N_bias)
    for specific operating frequencies.
    """

    FREQ_INDEX = {"1.690": 0, "2.080": 1, "2.455": 2}

    def __init__(
        self,
        amp_file='WBRO_amplitude.mat',
        phase_file='WNBRO_phase.mat',
        freq='2.080',
        v_min=0.0,
        v_max=12.0,
        folding=True,
        n_points=500
    ):
        self.v_min = v_min
        self.v_max = v_max
        self.folding = folding
        self.n_points = n_points
        self.freq_str = str(freq)

        if self.freq_str not in self.FREQ_INDEX:
            raise ValueError(f"Unsupported frequency '{freq}'. Choose from {list(self.FREQ_INDEX.keys())}")

        freq_idx = self.FREQ_INDEX[self.freq_str]

        try:
            phase_mat = sio.loadmat(amp_file if 'phase' in amp_file else phase_file)
            amp_mat = sio.loadmat(phase_file if 'amplitude' in phase_file else amp_file)

            phi_raw = phase_mat.get("WBRO_phase", phase_mat.get("WNBRO_phase"))
            A_raw = amp_mat.get("WBRO_amplitude")

            if phi_raw is None or A_raw is None:
                raise KeyError("Could not find 'WBRO_phase' or 'WBRO_amplitude' variables in .mat files.")

            # Extract selected frequency sweep line: shape (N_bias,)
            phi_freq = phi_raw[freq_idx, :]
            A_freq = A_raw[freq_idx, :]

            # Normalize amplitude against value at phase nearest to 360 degrees
            ref_idx = np.argmin(np.abs(phi_freq - 360.0))
            ref_val = A_freq[ref_idx] if A_freq[ref_idx] != 0 else 1.0
            A_norm = A_freq / ref_val

            # Measured bias grid across raw samples
            n_raw = len(A_norm)
            raw_v_grid = np.linspace(v_min, v_max, n_raw)

            # Restrict voltage range to 360 deg if folding is disabled
            if not self.folding:
                unwrapped_phi = np.unwrap(np.deg2rad(phi_freq))
                phase_delta = np.abs(unwrapped_phi - unwrapped_phi[0])

                exceed_indices = np.where(phase_delta >= 2 * np.pi)[0]
                if len(exceed_indices) > 0 and exceed_indices[0] > 0:
                    idx = exceed_indices[0]
                    # Interpolate exact voltage where phase excursion reaches 360 degrees
                    v_cutoff = np.interp(
                        2 * np.pi,
                        [phase_delta[idx - 1], phase_delta[idx]],
                        [raw_v_grid[idx - 1], raw_v_grid[idx]]
                    )
                    self.v_max = float(v_cutoff)

            # Build standardized interpolation grid within valid [v_min, v_max]
            self.V_grid = np.linspace(self.v_min, self.v_max, self.n_points)

            # Interpolate Amplitude and Phase onto dense grid
            A_interp = np.interp(self.V_grid, raw_v_grid, A_norm)
            phi_interp = np.interp(self.V_grid, raw_v_grid, phi_freq)

            # Compute complex response
            self.c_grid = A_interp * np.exp(1j * np.deg2rad(phi_interp))

            # Continuous interpolators for smooth get_complex_weight lookup
            self._interp_real = interp1d(self.V_grid, self.c_grid.real, kind='linear', fill_value='extrapolate')
            self._interp_imag = interp1d(self.V_grid, self.c_grid.imag, kind='linear', fill_value='extrapolate')

        except Exception as e:
            raise RuntimeError(f"Failed to load or parse measured data: {e}") from e

    def get_complex_weight(self, V, f=None):
        """
        Evaluates the complex weight at bias voltage V.
        Supports scalar inputs or NumPy arrays.
        """
        V_arr = np.asarray(V)
        re = self._interp_real(V_arr)
        im = self._interp_imag(V_arr)
        c = re + 1j * im
        return c.item() if np.ndim(V) == 0 else c
    
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

if __name__ == "__main__":
    MeasuredVaractor()

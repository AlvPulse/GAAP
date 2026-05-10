import numpy as np
import scipy.fft as fft

def angle_to_u(theta_rad):
    """Convert physical angle (radians) from broadside to u-space."""
    return np.sin(theta_rad)

def u_to_angle(u):
    """Convert u-space back to physical angle (radians) from broadside."""
    return np.arcsin(np.clip(u, -1.0, 1.0))

def oversampled_fft(weights, oversample_factor=8):
    """
    Compute oversampled FFT of complex weights to obtain pattern F(u).
    Array factor for lambda/2 spacing: F(u) = sum_{n=0}^{N-1} w_n * exp(j * n * pi * u)
    Returns (u_grid, F).
    """
    N = len(weights)
    N_pad = N * oversample_factor

    padded_weights = np.pad(weights, (0, N_pad - N))

    # We want F(u) = sum w_n exp(j * n * pi * u)
    # The FFT computes: X[k] = sum_{n=0}^{N_{pad}-1} x[n] exp(-j * 2pi * n * k / N_{pad})
    # To match, we need to take FFT of complex conjugated weights, then conjugate the result:
    # F(u) = conj(FFT(conj(padded_weights))) is equivalent to sum x[n] exp(+j * 2pi * n * k / N_{pad})
    # Then pi * u = 2pi * k / N_pad => u = 2k / N_pad

    F = np.conj(fft.fft(np.conj(padded_weights)))

    # Shift so that k=0 is in the middle
    F = fft.fftshift(F)
    k = np.arange(N_pad) - N_pad // 2
    u_grid = 2 * k / N_pad

    return u_grid, F

def oversampled_ifft(F, N):
    """
    Inverse of oversampled_fft to get back N weights from pattern F.
    """
    N_pad = len(F)
    F_unshift = fft.ifftshift(F)

    # We invert the operation: F = conj(FFT(conj(w)))
    # conj(F) = FFT(conj(w))
    # conj(w) = IFFT(conj(F))
    # w = conj(IFFT(conj(F)))

    padded_weights = np.conj(fft.ifft(np.conj(F_unshift)))
    return padded_weights[:N]

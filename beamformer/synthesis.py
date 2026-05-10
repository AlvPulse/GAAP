import numpy as np
from .utils import angle_to_u

def get_steering_vector(N, u_target):
    """
    Returns ideal linear phase weights for a given target u.
    w_n = exp(-j * pi * n * u_target)
    """
    n = np.arange(N)
    return np.exp(-1j * np.pi * n * u_target)

def synthesize_pencil(N, u_target):
    """
    Classical uniform linear phase progression for pencil beam.
    """
    return get_steering_vector(N, u_target)

def synthesize_sector(N, u_min, u_max):
    """
    Simple Woodward-Lawson superposition for a sector beam.
    """
    # Sample the u-space sector
    du = 2.0 / N
    u_samples = np.arange(u_min, u_max + du/2, du)

    weights = np.zeros(N, dtype=np.complex128)
    for u in u_samples:
        weights += get_steering_vector(N, u)

    # Normalize to unit magnitude (phase-only requirement)
    # Gerchberg-Saxton style projection
    return np.exp(1j * np.angle(weights))

def synthesize_multitarget(N, u_targets, relative_gains=None):
    """
    Superposition for multiple targets.
    """
    if relative_gains is None:
        relative_gains = np.ones(len(u_targets))

    weights = np.zeros(N, dtype=np.complex128)
    for u, g in zip(u_targets, relative_gains):
        weights += g * get_steering_vector(N, u)

    return np.exp(1j * np.angle(weights))

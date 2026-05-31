import numpy as np
import scipy.signal
from .utils import angle_to_u

def get_steering_vector(N, u_target):
    """
    Returns ideal linear phase weights for a given target u.
    w_n = exp(-j * pi * n * u_target)
    """
    n = np.arange(N)
    return np.exp(-1j * np.pi * n * u_target)

def apply_taper(weights, N, taper_type=None, sll_db=30):
    """
    Applies amplitude tapering to the weights for SLL reduction.
    Note: For phase-only arrays, this initial amplitude taper will be
    partially flattened by the hardware constraints later, but it gives
    the AP algorithm a better starting point.
    """
    if not taper_type or taper_type == 'uniform':
        return weights

    if taper_type == 'chebyshev':
        taper = scipy.signal.windows.chebwin(N, at=sll_db)
    elif taper_type == 'taylor':
        # Simple approximation, scipy doesn't have a direct taylor window
        # We can just use hamming or hanning as alternatives if taylor is requested
        # but not implemented
        taper = scipy.signal.windows.taylor(N, nbar=4, sll=sll_db)
    elif taper_type == 'hamming':
        taper = np.hamming(N)
    elif taper_type == 'hanning':
        taper = np.hanning(N)
    else:
        raise ValueError(f"Unknown taper type: {taper_type}")

    # Normalize taper
    taper = taper / np.max(taper)
    return weights * taper

def synthesize_pencil(N, u_target, taper_type=None, sll_db=30):
    """
    Classical uniform linear phase progression for pencil beam.
    """
    w = get_steering_vector(N, u_target)
    return apply_taper(w, N, taper_type, sll_db)

def synthesize_sector(N, u_min, u_max, taper_type=None, sll_db=30):
    """
    Simple Woodward-Lawson superposition for a sector beam.
    """
    # Sample the u-space sector
    du = 2.0 / N
    u_samples = np.arange(u_min, u_max + du/2, du)

    weights = np.zeros(N, dtype=np.complex128)
    for u in u_samples:
        weights += get_steering_vector(N, u)

    weights = apply_taper(weights, N, taper_type, sll_db)

    # We might not want to strictly force it to unit magnitude here if
    # we want the AP to start with a tapered ideal.
    # Let's keep the amplitude variations from superposition and taper.
    return weights / np.max(np.abs(weights))

def synthesize_multitarget(N, u_targets, relative_gains=None, taper_type=None, sll_db=30):
    """
    Superposition for multiple targets.
    """
    if relative_gains is None:
        relative_gains = np.ones(len(u_targets))

    weights = np.zeros(N, dtype=np.complex128)
    for u, g in zip(u_targets, relative_gains):
        weights += g * get_steering_vector(N, u)

    weights = apply_taper(weights, N, taper_type, sll_db)
    max_val = np.max(np.abs(weights))
    if max_val > 1e-12:
        weights = weights / max_val
    return weights

def synthesize_schelkunoff(N, u_target, null_angles):
    """
    Schelkunoff polynomial root shifting for null synthesis.
    Places roots exactly at the desired null angles on the unit circle.
    """
    # Standard array roots for uniform progression towards u_target
    roots = np.exp(1j * 2 * np.pi * np.arange(1, N) / N)

    # Rotate standard roots to steer main beam to u_target
    roots *= np.exp(1j * np.pi * u_target)

    # Replace closest roots with explicit null roots
    for null_u in null_angles:
        null_root = np.exp(1j * np.pi * null_u)
        # Find closest existing root
        dists = np.abs(roots - null_root)
        idx = np.argmin(dists)
        roots[idx] = null_root

    # Expand roots to polynomial coefficients (which are our weights)
    poly = np.poly(roots)

    # Poly gives weights from right to left, we reverse to match normal indexing
    weights = poly[::-1]

    # Normalize
    return weights / np.max(np.abs(weights))

def synthesize_phase_only_lms(N, u_target, null_angles, iterations=200, mu=0.1):
    """
    Phase-only friendly LMS-style initialization for nulls.
    Starts with a pencil beam and iteratively perturbs phases to dig nulls.
    """
    w = get_steering_vector(N, u_target)
    phases = np.angle(w)

    for _ in range(iterations):
        # Current pattern at nulls
        null_vals = np.zeros(len(null_angles), dtype=np.complex128)
        for i, nu in enumerate(null_angles):
            sv = get_steering_vector(N, nu)
            null_vals[i] = np.sum(np.exp(1j * phases) * np.conj(sv))

        # Update phases to reduce null power
        for i, nu in enumerate(null_angles):
            sv = get_steering_vector(N, nu)
            # Phase gradient to reduce amplitude at this angle
            grad = -np.imag(np.exp(-1j * phases) * sv * null_vals[i])
            phases -= mu * grad

    return np.exp(1j * phases)

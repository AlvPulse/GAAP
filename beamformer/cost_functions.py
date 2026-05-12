import numpy as np
from .utils import oversampled_fft

def evaluate_pattern_cost(c_n, task, oversample_factor=8):
    """
    Evaluates a scalar cost function in the pattern domain.
    Lower cost is better.

    Weights:
    - Main lobe gain loss: 0.5
    - Null depth violation: 0.35
    - SLL violation: 0.15
    """
    u_grid, F = oversampled_fft(c_n, oversample_factor)
    power_db = 20 * np.log10(np.abs(F) + 1e-12)

    cost = 0.0

    # 1. Main lobe gain (we want to maximize this, so lower is better)
    # The max possible power with a unit array is N.
    N = len(c_n)
    max_ideal_power_db = 20 * np.log10(N)

    if task.target_angles:
        # Check power at target angles
        target_power = []
        for angle in task.target_angles:
            idx = np.argmin(np.abs(u_grid - angle))
            target_power.append(power_db[idx])
        avg_target_power = np.mean(target_power)
        gain_loss = max_ideal_power_db - avg_target_power
        cost += 0.5 * max(0, gain_loss) # Penalty for losing gain

    # 2. Null depths
    if hasattr(task, 'null_angles') and task.null_angles:
        null_violation = 0.0
        for angle in task.null_angles:
            idx = np.argmin(np.abs(u_grid - angle))
            # If null is not deep enough (e.g. we want it below -40 dB relative to max)
            depth_violation = power_db[idx] - (max_ideal_power_db - 40)
            if depth_violation > 0:
                null_violation += depth_violation
        cost += 0.35 * null_violation

    # 3. SLL violation
    if hasattr(task, 'sll_ceiling'):
        # Find peaks
        # For simplicity, we just look at the whole region outside main lobe and nulls
        # SLL ceiling is usually relative to main lobe, but let's use absolute here for stability
        sll_db_abs = max_ideal_power_db + 20 * np.log10(task.sll_ceiling)

        # Mask out main lobes
        mask = np.ones_like(u_grid, dtype=bool)
        if task.target_angles:
            for angle in task.target_angles:
                mask &= (np.abs(u_grid - angle) > 0.1) # Exclude main beam width

        if task.type == 'sector':
            mask &= ((u_grid < task.sector_u_min - 0.1) | (u_grid > task.sector_u_max + 0.1))

        if np.any(mask):
            sidelobe_power = power_db[mask]
            max_sll = np.max(sidelobe_power)
            sll_violation = max_sll - sll_db_abs
            if sll_violation > 0:
                cost += 0.15 * sll_violation

    return cost

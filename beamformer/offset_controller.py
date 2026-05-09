import numpy as np

class GeometryAwareOffsetController:
    """
    Controls the global phase offset dynamically during the AP iteration based on
    convergence geometry diagnostics, avoiding direct grid-search optimization.
    """
    def __init__(self, initial_delta=0.0):
        self.delta = initial_delta

        # History for diagnostics
        self.residuals = []
        self.voltages = None
        self.deltas = [initial_delta]

    def update(self, current_residual, new_voltages, iteration):
        """
        State-dependent offset evolution.
        Returns the new delta and suggested damping factors.
        """
        self.residuals.append(current_residual)

        # Baseline recommended damping
        tau = 0.7   # Pattern domain damping
        alpha = 0.7 # Hardware domain damping

        if iteration < 2 or self.voltages is None:
            self.voltages = new_voltages.copy()
            return self.delta, tau, alpha

        # Diagnostics
        res_change = self.residuals[-1] - self.residuals[-2]

        # Branch instability check (voltage jumps)
        voltage_diffs = np.abs(new_voltages - self.voltages)
        num_flips = np.sum(voltage_diffs > 3.0) # threshold for a branch flip

        # Oscillation check (residual bouncing)
        is_oscillating = False
        if len(self.residuals) >= 3:
            prev_change = self.residuals[-2] - self.residuals[-3]
            if res_change * prev_change < 0 and np.abs(res_change) > 1e-3:
                is_oscillating = True

        # Rule 1: Stable convergence
        if res_change < 0 and num_flips == 0 and not is_oscillating:
            u_k = 0.0 # No need to move

        # Rule 2: Oscillation detected
        elif is_oscillating:
            # Gently perturb delta to reshape manifold geometry locally
            u_k = 0.05 * np.sign(np.random.randn())
            tau *= 0.8
            alpha *= 0.8

        # Rule 3: Branch instability
        elif num_flips > 2:
            # High conflict, slow down offset velocity and increase damping
            u_k = 0.01 * np.sign(np.random.randn())
            tau *= 0.5
            alpha *= 0.5

        # Rule 4: Stagnation
        elif np.abs(res_change) < 1e-4 and current_residual > 0.1:
            # Drift towards lower-conflict regions
            u_k = 0.02
        else:
            u_k = 0.0 # Default hold

        # Apply update
        self.delta = (self.delta + u_k) % (2 * np.pi)

        self.voltages = new_voltages.copy()
        self.deltas.append(self.delta)

        return self.delta, tau, alpha

import numpy as np

class GeometryAwareOffsetController:
    """
    Controls the global phase offset dynamically during the AP iteration based on
    convergence geometry diagnostics.
    Now correctly tracks Aperture Potential (E_eff), Branch Flips, and Pattern Cost,
    and uses numerical gradients to find the optimal direction for delta.
    """
    def __init__(self, initial_delta=0.0, use_pattern_cost=False):
        self.delta = initial_delta
        self.use_pattern_cost = use_pattern_cost

        # History for diagnostics
        self.history = {
            'costs': [],
            'e_effs': [],
            'flips': [],
            'deltas': [initial_delta]
        }

        # State for gradient computation
        self._perturb_mode = False
        self._base_delta = initial_delta
        self._base_cost = None
        self._base_e_eff = None
        self._perturb_direction = 1
        self._perturb_step = 0.05

    def update(self, current_cost, e_eff, num_flips, iteration):
        """
        State-dependent offset evolution using explicit geometry gradients.
        """
        self.history['costs'].append(current_cost)
        self.history['e_effs'].append(e_eff)
        self.history['flips'].append(num_flips)

        tau = 0.7   # Pattern domain damping
        alpha = 0.7 # Hardware domain damping

        if iteration < 2:
            return self.delta, tau, alpha

        # 1. Check if we are in perturb mode to evaluate gradient
        if self._perturb_mode:
            # We stepped delta by self._perturb_step * self._perturb_direction.
            # Did the geometry improve?

            # Improvement is a combination of lowering cost and raising E_eff
            cost_improved = current_cost < self._base_cost
            e_eff_improved = e_eff > self._base_e_eff

            # We require pattern cost to improve or stay roughly the same while E_eff improves
            if cost_improved or (abs(current_cost - self._base_cost) < 1e-3 and e_eff_improved):
                # Gradient confirmed! Move in this direction.
                step = 0.05 * self._perturb_direction
            else:
                # Gradient failed. Reverse direction.
                self._perturb_direction *= -1
                step = 0.05 * self._perturb_direction

            self._perturb_mode = False
            self.delta = (self._base_delta + step) % (2 * np.pi)

        else:
            # 2. We are in standard tracking mode. Check if geometry is stagnating or unstable.
            cost_change = self.history['costs'][-1] - self.history['costs'][-2]

            is_oscillating = False
            if len(self.history['costs']) >= 3:
                prev_change = self.history['costs'][-2] - self.history['costs'][-3]
                if cost_change * prev_change < 0 and np.abs(cost_change) > 1e-3:
                    is_oscillating = True

            if is_oscillating or (np.abs(cost_change) < 1e-4 and current_cost > 0.1):
                # Stagnation or oscillation detected. Initiate a gradient perturb to find a better basin.
                self._perturb_mode = True
                self._base_delta = self.delta
                self._base_cost = current_cost
                self._base_e_eff = e_eff

                # Take a tiny test step
                self.delta = (self.delta + self._perturb_step * self._perturb_direction) % (2 * np.pi)
                tau *= 0.8
                alpha *= 0.8
            elif num_flips > 2:
                # High conflict, slow down
                tau *= 0.5
                alpha *= 0.5
                # We don't move delta here, just let the AP settle

        self.history['deltas'].append(self.delta)
        return self.delta, tau, alpha

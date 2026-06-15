import numpy as np
from .sa_families import BaseController

class StrainGuidedController(BaseController):
    """
    Strain-Guided Gauge Controller.
    This outer loop meta-controller unifies the 3-layer pipeline by using
    lower-level feedback (strain) from the Local Refinement (LR) stage.

    Strain is defined as a combination of:
    1. Effort: How many active set elements LR had to perturb (acc_corr).
    2. Degradation: How much peak gain was sacrificed to achieve the nulls.

    If strain is high, the current macro-gauge (delta) is highly strained/pathological,
    triggering a larger phase jump and a reheat of the inner memory.
    If strain is low, LR only needed surgical tweaks, so we lock into the basin.
    """
    def __init__(self, N, target_gain_db, max_corrections_expected=None, T0=10.0, alpha=0.9, **kwargs):
        super().__init__(**kwargs)
        self.N = N
        self.target_gain_db = target_gain_db # Usually 20*log10(N)
        self.max_corrections = max_corrections_expected if max_corrections_expected else 0.1 * N

        self.T = T0
        self.T0 = T0
        self.alpha = alpha

        # Internal memory
        self.lr_step = kwargs.get('lr_initial', 0.05)
        self.lr_initial = kwargs.get('lr_initial', 0.05)
        self.lr_floor = kwargs.get('lr_min', 0.001)

        self.current_strain = 1.0 # start fully strained to explore

    def propose(self, current_delta, **kwargs):
        # Proposal width scales directly with current strain
        # High strain -> large jump (exploration). Low strain -> micro jitter (exploitation).
        jump_std = self.current_strain * (np.pi / 2)
        return (current_delta + np.random.normal(0, jump_std)) % (2 * np.pi)

    def accept(self, current_cost, new_cost):
        if new_cost < current_cost:
            return True
        if self.T > 1e-6:
            prob = np.exp(-(new_cost - current_cost) / self.T)
            return np.random.rand() < prob
        return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)

        if accepted:
            # Extract lower-level feedback
            gain = kwargs.get('gain', self.target_gain_db)
            acc_corr = kwargs.get('acc_corr', 0)

            # Calculate Gain Degradation Penalty (0 to 1)
            # If gain drops by 3dB or more, degradation is high.
            gain_loss = max(0, self.target_gain_db - gain)
            degradation = np.clip(gain_loss / 3.0, 0.0, 1.0)

            # Calculate LR Effort Penalty (0 to 1)
            effort = np.clip(acc_corr / max(self.max_corrections, 1), 0.0, 1.0)

            # Combined Strain Metric
            # Weight gain degradation highly because the user's novel LR protects gain.
            # If gain falls, the offset gauge is extremely bad.
            self.current_strain = 0.7 * degradation + 0.3 * effort

            # Soft Carryover of Inner Memory (LR step size) based on strain
            # Low strain preserves memory (close to lr_floor). High strain wipes memory (lr_initial).
            self.lr_step = (1 - self.current_strain) * self.lr_step + self.current_strain * self.lr_initial
            self.lr_step = np.clip(self.lr_step, self.lr_floor, self.lr_initial)

        self.T *= self.alpha

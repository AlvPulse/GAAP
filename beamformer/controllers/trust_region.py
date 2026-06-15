import numpy as np
from .sa_families import BaseController

class TR_HC(BaseController):
    def __init__(self, delta_init=np.pi/4, delta_min=0.01, delta_max=np.pi, expand=1.2, shrink=0.5, smart_reset=False, jump_threshold=np.pi/4, **kwargs):
        super().__init__(**kwargs)
        self.TR = delta_init
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.expand = expand
        self.shrink = shrink
        self.smart_reset = smart_reset
        self.jump_threshold = jump_threshold
        self.lr_step = kwargs.get('lr_initial', 0.05)
        self.lr_initial = kwargs.get('lr_initial', 0.05)

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.uniform(-self.TR, self.TR)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        return new_cost < current_cost

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)

        current_lr = kwargs.get('lr_step', self.lr_step)

        if accepted:
            self.TR = min(self.TR * self.expand, self.delta_max)

            # Manage inner LR step memory based on jump size
            current_delta = kwargs.get('current_delta', 0.0)
            cand_delta = kwargs.get('cand_delta', 0.0)
            diff = np.abs(cand_delta - current_delta)
            circular_diff = min(diff, 2*np.pi - diff)

            if not self.smart_reset:
                # Legacy hard reset on every jump
                self.lr_step = self.lr_initial
            else:
                # Smart reset: only wipe memory on large jumps
                if circular_diff > self.jump_threshold:
                    self.lr_step = self.lr_initial
                else:
                    self.lr_step = current_lr
        else:
            self.TR = max(self.TR * self.shrink, self.delta_min)
            self.lr_step = current_lr

class TR_SA(BaseController):
    def __init__(self, T0=10.0, alpha=0.9, delta_init=np.pi/4, delta_min=0.01, delta_max=np.pi, expand=1.2, shrink=0.5, smart_reset=False, jump_threshold=np.pi/4, **kwargs):
        super().__init__(**kwargs)
        self.T = T0
        self.alpha = alpha
        self.TR = delta_init
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.expand = expand
        self.shrink = shrink
        self.smart_reset = smart_reset
        self.jump_threshold = jump_threshold
        self.lr_step = kwargs.get('lr_initial', 0.05)
        self.lr_initial = kwargs.get('lr_initial', 0.05)

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.uniform(-self.TR, self.TR)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        if new_cost < current_cost:
            return True
        if self.T > 1e-6:
            prob = np.exp(-(new_cost - current_cost) / self.T)
            return np.random.rand() < prob
        return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)

        current_lr = kwargs.get('lr_step', self.lr_step)

        if accepted:
            self.TR = min(self.TR * self.expand, self.delta_max)

            # Manage inner LR step memory based on jump size
            current_delta = kwargs.get('current_delta', 0.0)
            cand_delta = kwargs.get('cand_delta', 0.0)
            diff = np.abs(cand_delta - current_delta)
            circular_diff = min(diff, 2*np.pi - diff)

            if not self.smart_reset:
                # Legacy hard reset on every jump
                self.lr_step = self.lr_initial
            else:
                # Smart reset: only wipe memory on large jumps
                if circular_diff > self.jump_threshold:
                    self.lr_step = self.lr_initial
                else:
                    self.lr_step = current_lr
        else:
            self.TR = max(self.TR * self.shrink, self.delta_min)
            self.lr_step = current_lr

        self.T *= self.alpha

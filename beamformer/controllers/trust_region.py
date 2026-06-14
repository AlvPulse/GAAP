import numpy as np
from .sa_families import BaseController

class TR_HC(BaseController):
    def __init__(self, delta_init=np.pi/4, delta_min=0.01, delta_max=np.pi, expand=1.2, shrink=0.5, **kwargs):
        super().__init__(**kwargs)
        self.TR = delta_init
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.expand = expand
        self.shrink = shrink

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.uniform(-self.TR, self.TR)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        return new_cost < current_cost

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)
        if accepted:
            self.TR = min(self.TR * self.expand, self.delta_max)
        else:
            self.TR = max(self.TR * self.shrink, self.delta_min)

class TR_SA(BaseController):
    def __init__(self, T0=10.0, alpha=0.9, delta_init=np.pi/4, delta_min=0.01, delta_max=np.pi, expand=1.2, shrink=0.5, **kwargs):
        super().__init__(**kwargs)
        self.T = T0
        self.alpha = alpha
        self.TR = delta_init
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.expand = expand
        self.shrink = shrink

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
        if accepted:
            self.TR = min(self.TR * self.expand, self.delta_max)
        else:
            self.TR = max(self.TR * self.shrink, self.delta_min)

        self.T *= self.alpha

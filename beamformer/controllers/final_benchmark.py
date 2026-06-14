import numpy as np

class BaseController:
    def __init__(self, **kwargs):
        self.accepted_moves = 0
        self.total_proposals = 0

    def propose(self, current_delta, **kwargs):
        raise NotImplementedError

    def accept(self, current_cost, new_cost):
        raise NotImplementedError

    def update_state(self, accepted, **kwargs):
        self.total_proposals += 1
        if accepted:
            self.accepted_moves += 1

class FixedOffset(BaseController):
    def propose(self, current_delta, **kwargs):
        return current_delta

    def accept(self, current_cost, new_cost):
        # Always evaluate inner AP+LR
        return True

class RandomRestart(BaseController):
    def __init__(self, restart_interval=10, **kwargs):
        super().__init__(**kwargs)
        self.restart_interval = restart_interval
        self.counter = 0

    def propose(self, current_delta, **kwargs):
        self.counter += 1
        if self.counter % self.restart_interval == 0:
            return np.random.uniform(0, 2*np.pi)
        return current_delta

    def accept(self, current_cost, new_cost):
        return True # Always evaluate

class HillClimbing(BaseController):
    def __init__(self, step_size=np.pi/18, **kwargs): # 10 degrees default
        super().__init__(**kwargs)
        self.step_size = step_size

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.normal(0, self.step_size)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        return new_cost < current_cost

class SA(BaseController):
    def __init__(self, T0=10.0, alpha=0.9, sigma=np.pi/12, **kwargs):
        super().__init__(**kwargs)
        self.T = T0
        self.alpha = alpha
        self.sigma = sigma

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.normal(0, self.sigma)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        if new_cost < current_cost:
            return True
        if self.T > 1e-6:
            prob = np.exp(-(new_cost - current_cost) / self.T)
            return np.random.rand() < prob
        return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)
        self.T *= self.alpha

class CROA(BaseController):
    """
    Continuation-Reheated Offset Annealing (CROA).
    Implements distance-aware reheating.
    """
    def __init__(self, T0=10.0, alpha=0.9, sigma=np.pi/12, W=5, eta=0.5, ablation_mode='soft_carryover', **kwargs):
        super().__init__(**kwargs)
        self.T0 = T0
        self.T = T0
        self.alpha = alpha
        self.TR = sigma # Trust region / proposal std
        self.TR0 = sigma
        self.W = W
        self.eta = eta

        self.ablation_mode = ablation_mode # 'soft_carryover', 'hard_reset', 'no_reheat', 'always_reheat'

        # State memory
        self.lr_step = kwargs.get('lr_initial', 0.05)
        self.lr_initial = kwargs.get('lr_initial', 0.05)
        self.lr_floor = kwargs.get('lr_min', 0.001)

    def propose(self, current_delta, **kwargs):
        # Heavy-tail jumps occasionally to allow escaping deep basins
        if np.random.rand() < 0.1:
            J_t = np.random.normal(0, np.pi/2)
        else:
            J_t = 0

        return (current_delta + np.random.normal(0, self.TR) + J_t) % (2*np.pi)

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
            current_delta = kwargs.get('current_delta', 0.0)
            cand_delta = kwargs.get('cand_delta', 0.0)

            diff = np.abs(cand_delta - current_delta)
            circular_diff = min(diff, 2*np.pi - diff)
            rho = circular_diff / np.pi # [0, 1]

            if self.ablation_mode == 'soft_carryover':
                # Distance-aware reheating
                self.lr_step = self.lr_floor + (self.lr_initial - self.lr_floor) * rho
                self.TR = self.TR0 * rho + self.TR * (1 - rho)
            elif self.ablation_mode == 'hard_reset':
                self.lr_step = self.lr_initial
                self.TR = self.TR0
            elif self.ablation_mode == 'no_reheat':
                pass # keep lr_step and TR as is
            elif self.ablation_mode == 'always_reheat':
                self.lr_step = self.lr_initial
                self.TR = self.TR0

        # Temperature Reheat on stagnation (Optional extra defense)
        history_r = kwargs.get('history_r', [])
        reheated = False
        if len(history_r) >= self.W:
            r_t = history_r[-1]
            r_t_W = history_r[-self.W]

            if r_t_W > 1e-12:
                gamma = (r_t / r_t_W)**(1/self.W)
                if gamma > 0.98 and self.lr_step <= self.lr_floor * 1.01:
                    if self.ablation_mode != 'no_reheat':
                        self.T = self.eta * self.T0
                        reheated = True

        if not reheated:
            self.T *= self.alpha

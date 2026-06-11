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
        return False # Never move

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
        # We force acceptance on restart intervals
        if self.counter % self.restart_interval == 0:
            return True
        return False

class HillClimbing(BaseController):
    def __init__(self, step_size=np.pi/18, **kwargs): # 10 degrees default
        super().__init__(**kwargs)
        self.step_size = step_size

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.normal(0, self.step_size)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        return new_cost < current_cost # Strictly greedy

class BasinHopping(BaseController):
    def propose(self, current_delta, **kwargs):
        # Large random jump
        return np.random.uniform(0, 2*np.pi)

    def accept(self, current_cost, new_cost):
        # Accept if better or equal
        return new_cost <= current_cost

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

class VFSA(BaseController):
    def __init__(self, T0=10.0, c=1.0, **kwargs):
        super().__init__(**kwargs)
        self.T0 = T0
        self.c = c
        self.k = 1
        self.T = T0

    def propose(self, current_delta, **kwargs):
        # Textbook VFSA 1D generating distribution
        u = np.random.uniform(0, 1)
        y = np.sign(u - 0.5) * self.T * ((1 + 1/self.T)**np.abs(2*u - 1) - 1)
        # Assuming domain size is 2pi, scale y by pi (max move is domain bounds)
        new_delta = (current_delta + y * np.pi) % (2*np.pi)
        return new_delta

    def accept(self, current_cost, new_cost):
        if new_cost < current_cost:
            return True
        if self.T > 1e-6:
            prob = np.exp(-(new_cost - current_cost) / self.T)
            return np.random.rand() < prob
        return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)
        self.k += 1
        self.T = self.T0 * np.exp(-self.c * self.k) # D=1

class ASA(BaseController):
    def __init__(self, T0=10.0, **kwargs):
        super().__init__(**kwargs)
        self.T0 = T0
        self.T_cost = T0
        self.k = 1
        self.sigma = np.pi/4 # Start with large proposal width

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.normal(0, self.sigma)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        if new_cost < current_cost:
            return True
        if self.T_cost > 1e-6:
            prob = np.exp(-(new_cost - current_cost) / self.T_cost)
            return np.random.rand() < prob
        return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)
        self.k += 1
        # Adaptive components:
        # If we accept too much, increase sigma to explore wider
        # If we reject too much, decrease sigma to exploit
        recent_accept_rate = self.accepted_moves / self.total_proposals
        if recent_accept_rate > 0.4:
            self.sigma *= 1.05
        elif recent_accept_rate < 0.2:
            self.sigma *= 0.95

        self.sigma = np.clip(self.sigma, 0.01, np.pi)
        self.T_cost = self.T0 * np.exp(-0.1 * self.k**(1/1))

class HT_ROA(BaseController):
    def __init__(self, T0=10.0, alpha=0.9, sigma=np.pi/12, W=5, eta=0.5, **kwargs):
        super().__init__(**kwargs)
        self.T0 = T0
        self.T = T0
        self.alpha = alpha
        self.sigma = sigma
        self.W = W
        self.eta = eta

    def propose(self, current_delta, **kwargs):
        # Heavy-tailed proposal
        if np.random.rand() < 0.1:
            J_t = np.random.normal(0, np.pi/2)
        else:
            J_t = 0

        new_delta = (current_delta + np.random.normal(0, self.sigma) + J_t) % (2*np.pi)
        return new_delta

    def accept(self, current_cost, new_cost):
        if new_cost < current_cost:
            return True
        if self.T > 1e-6:
            prob = np.exp(-(new_cost - current_cost) / self.T)
            return np.random.rand() < prob
        return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)

        # Check reheat trigger
        history_r = kwargs.get('history_r', [])
        lr_step = kwargs.get('lr_step', 0.1)
        lr_min = kwargs.get('lr_min', 0.001)

        reheated = False
        if len(history_r) >= self.W:
            r_t = history_r[-1]
            r_t_W = history_r[-self.W]

            if r_t_W > 1e-12:
                gamma = (r_t / r_t_W)**(1/self.W)
                # Reheat if stagnant and LR step size is floored
                if gamma > 0.98 and lr_step <= lr_min * 1.01:
                    self.T = self.eta * self.T0
                    reheated = True

        if not reheated:
            self.T *= self.alpha

class CEM(BaseController):
    def __init__(self, pop_size=10, elite_frac=0.2, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.elite_size = max(1, int(pop_size * elite_frac))
        self.mu = 0.0 # updated when first proposed
        self.sigma = np.pi # wide initial search

        self.pop_deltas = []
        self.pop_costs = []
        self.curr_idx = 0
        self.initialized = False

    def propose(self, current_delta, **kwargs):
        if not self.initialized:
            self.mu = current_delta
            self.initialized = True

        if self.curr_idx == 0:
            # Generate new population
            self.pop_deltas = (np.random.normal(self.mu, self.sigma, self.pop_size)) % (2*np.pi)
            self.pop_costs = []

        return self.pop_deltas[self.curr_idx]

    def accept(self, current_cost, new_cost):
        # In CEM, we always evaluate (so "accept" the evaluation)
        # but we don't necessarily update the base state immediately.
        # We will track costs and update mu/sigma at end of generation.
        # We return True to ensure the outer loop sets the new cost for tracking,
        # but for true CEM, the "best" elite becomes the center.
        return True

    def update_state(self, accepted, new_cost=float('inf'), **kwargs):
        super().update_state(accepted, **kwargs)
        self.pop_costs.append(new_cost)
        self.curr_idx += 1

        if self.curr_idx >= self.pop_size:
            # End of generation, update distribution
            elite_indices = np.argsort(self.pop_costs)[:self.elite_size]
            elites = self.pop_deltas[elite_indices]

            # Simple circular mean/std
            mean_sin = np.mean(np.sin(elites))
            mean_cos = np.mean(np.cos(elites))
            self.mu = np.arctan2(mean_sin, mean_cos) % (2*np.pi)

            # Approximate circular std
            R = np.sqrt(mean_sin**2 + mean_cos**2)
            self.sigma = np.sqrt(-2 * np.log(R + 1e-12))

            self.curr_idx = 0

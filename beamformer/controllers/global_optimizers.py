import numpy as np
from .sa_families import BaseController

class GA_1D(BaseController):
    """
    1D Genetic Algorithm tailored for offset delta in [0, 2pi].
    """
    def __init__(self, pop_size=10, mutation_rate=0.2, mutation_std=np.pi/4, **kwargs):
        super().__init__(**kwargs)
        self.pop_size = pop_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std

        self.population = np.random.uniform(0, 2*np.pi, self.pop_size)
        self.fitness = np.zeros(self.pop_size)
        self.curr_idx = 0
        self.initialized = False

    def propose(self, current_delta, **kwargs):
        if not self.initialized:
            # Force current_delta into population zero for fair start
            self.population[0] = current_delta
            self.initialized = True

        return self.population[self.curr_idx]

    def accept(self, current_cost, new_cost):
        return True # Always accept to allow state progression

    def update_state(self, accepted, new_cost=float('inf'), **kwargs):
        super().update_state(accepted, **kwargs)
        self.fitness[self.curr_idx] = new_cost
        self.curr_idx += 1

        if self.curr_idx >= self.pop_size:
            # Selection (Tournament)
            new_pop = np.zeros(self.pop_size)
            for i in range(self.pop_size):
                idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
                winner = idx1 if self.fitness[idx1] < self.fitness[idx2] else idx2
                new_pop[i] = self.population[winner]

            # Crossover (Arithmetic crossover mod 2pi)
            for i in range(0, self.pop_size-1, 2):
                if np.random.rand() < 0.7:
                    d1, d2 = new_pop[i], new_pop[i+1]
                    # Calculate circular mean
                    mean_sin = (np.sin(d1) + np.sin(d2)) / 2.0
                    mean_cos = (np.cos(d1) + np.cos(d2)) / 2.0
                    child = np.arctan2(mean_sin, mean_cos) % (2*np.pi)
                    new_pop[i] = child
                    new_pop[i+1] = (child + np.pi) % (2*np.pi) # Antenodal child for diversity

            # Mutation
            for i in range(self.pop_size):
                if np.random.rand() < self.mutation_rate:
                    new_pop[i] = (new_pop[i] + np.random.normal(0, self.mutation_std)) % (2*np.pi)

            self.population = new_pop
            self.fitness = np.zeros(self.pop_size)
            self.curr_idx = 0


class PSO_1D(BaseController):
    """
    1D Particle Swarm Optimization tailored for offset delta.
    """
    def __init__(self, swarm_size=10, w=0.5, c1=1.5, c2=1.5, **kwargs):
        super().__init__(**kwargs)
        self.swarm_size = swarm_size
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.positions = np.random.uniform(0, 2*np.pi, self.swarm_size)
        self.velocities = np.random.uniform(-np.pi/4, np.pi/4, self.swarm_size)

        self.pbest_pos = self.positions.copy()
        self.pbest_fit = np.ones(self.swarm_size) * float('inf')

        self.gbest_pos = 0.0
        self.gbest_fit = float('inf')

        self.curr_idx = 0
        self.initialized = False

    def propose(self, current_delta, **kwargs):
        if not self.initialized:
            self.positions[0] = current_delta
            self.initialized = True

        return self.positions[self.curr_idx]

    def accept(self, current_cost, new_cost):
        return True

    def update_state(self, accepted, new_cost=float('inf'), **kwargs):
        super().update_state(accepted, **kwargs)

        # Update Personal Best
        if new_cost < self.pbest_fit[self.curr_idx]:
            self.pbest_fit[self.curr_idx] = new_cost
            self.pbest_pos[self.curr_idx] = self.positions[self.curr_idx]

        # Update Global Best
        if new_cost < self.gbest_fit:
            self.gbest_fit = new_cost
            self.gbest_pos = self.positions[self.curr_idx]

        self.curr_idx += 1

        if self.curr_idx >= self.swarm_size:
            # End of swarm evaluation, update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()

                # Circular distance for personal best
                diff_p = (self.pbest_pos[i] - self.positions[i] + np.pi) % (2*np.pi) - np.pi
                # Circular distance for global best
                diff_g = (self.gbest_pos - self.positions[i] + np.pi) % (2*np.pi) - np.pi

                self.velocities[i] = self.w * self.velocities[i] + self.c1 * r1 * diff_p + self.c2 * r2 * diff_g
                self.velocities[i] = np.clip(self.velocities[i], -np.pi, np.pi) # Vmax bounds

                self.positions[i] = (self.positions[i] + self.velocities[i]) % (2*np.pi)

            self.curr_idx = 0

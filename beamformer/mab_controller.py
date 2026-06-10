import numpy as np

class UCBPhaseController:
    """
    Multi-Armed Bandit (UCB1) Phase Sector Controller.
    Discretizes the [0, 2pi] space into M sectors.
    Balances Exploration (trying unvisited sectors) and Exploitation (focusing on low-residual sectors).
    """
    def __init__(self, n_arms=16, exploration_weight=2.0):
        self.n_arms = n_arms
        self.exploration_weight = exploration_weight
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms) # Will store average REWARD (we want to maximize this, so reward = 1/cost)
        self.total_pulls = 0

    def select_arm(self):
        # Try each arm at least once
        if self.total_pulls < self.n_arms:
            return self.total_pulls

        # UCB1 algorithm
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                ucb_values[arm] = float('inf')
            else:
                exploitation = self.values[arm]
                exploration = self.exploration_weight * np.sqrt(np.log(self.total_pulls) / self.counts[arm])
                ucb_values[arm] = exploitation + exploration

        return np.argmax(ucb_values)

    def update(self, arm, cost):
        # Convert cost to reward (prevent divide by zero)
        reward = 1.0 / (cost + 1e-6)

        self.counts[arm] += 1
        n = self.counts[arm]
        # Running average
        self.values[arm] = ((n - 1) * self.values[arm] + reward) / n
        self.total_pulls += 1

    def get_delta_for_arm(self, arm):
        sector_size = 2 * np.pi / self.n_arms
        # Center of the sector
        return arm * sector_size + sector_size / 2.0

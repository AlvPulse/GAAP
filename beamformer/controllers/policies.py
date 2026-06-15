import numpy as np
from typing import List, Dict

class ControllerPolicy:
    def __init__(self, N: int):
        self.N = N

    def step(self, t: int, delta: float, metrics: Dict, history: List[Dict], w: np.ndarray, b: np.ndarray, backbone) -> float:
        """
        Given the current state, returns the next delta.
        """
        raise NotImplementedError

class PolicyA_FixedBaseline(ControllerPolicy):
    def step(self, t, delta, metrics, history, w, b, backbone):
        return delta

class PolicyB_RandomBasinHopping(ControllerPolicy):
    def step(self, t, delta, metrics, history, w, b, backbone):
        if np.random.rand() < 0.2:
            return np.random.uniform(0, 2*np.pi)
        return delta

class PolicyC_TrajectoryScan(ControllerPolicy):
    def __init__(self, N: int, M: int = 10, scan_angles=[-30, -15, 0, 15, 30]):
        super().__init__(N)
        self.M = M
        self.scan_angles = scan_angles
        self.scan_idx = 0
        self.delta_gain = 0.0 # Expected to be set globally or initialized

    def step(self, t, delta, metrics, history, w, b, backbone):
        if len(history) == 0:
            self.delta_gain = delta # Assuming initial delta is optimal for gain
            return delta

        if t % self.M == 0 and len(history) >= self.M:
            # Look back M steps
            window = history[-self.M:]
            mean_null = np.mean([h['null_depth'] for h in window])
            mean_gain = np.mean([h['gain'] for h in window])

            def_null = mean_null - (-40.0)
            max_gain = 20 * np.log10(self.N)
            def_gain = max_gain - mean_gain

            if def_null > def_gain:
                # Null bottleneck -> local scan
                angle_deg = self.scan_angles[self.scan_idx % len(self.scan_angles)]
                self.scan_idx += 1
                return delta + (angle_deg * np.pi / 180)
            else:
                # Gain bottleneck -> force to macro optimum
                return self.delta_gain
        return delta

class PolicyD_StagnationEscapement(ControllerPolicy):
    def __init__(self, N: int, W: int = 5):
        super().__init__(N)
        self.W = W
        self.b_history = []
        self.w_history = []

    def step(self, t, delta, metrics, history, w, b, backbone):
        self.b_history.append(b.copy())
        self.w_history.append(w.copy())

        if len(history) < self.W:
            return delta

        window = history[-self.W:]
        sigma_null = np.std([h['null_depth'] for h in window])
        sigma_gain = np.std([h['gain'] for h in window])

        # Calculate instantaneous branch flips over window
        # b_history has length at least W+1 at this point because we just appended
        flips_total = 0
        for tau in range(len(self.b_history) - self.W, len(self.b_history)):
            flips = np.sum(self.b_history[tau] != self.b_history[tau-1])
            flips_total += flips
        avg_flips = flips_total / self.W

        if sigma_null < 0.5 and sigma_gain < 0.2 and avg_flips <= 1:
            # Stagnation! Calculate trajectory momentum
            w_old = self.w_history[-self.W-1]
            # Mean phase change
            phase_change = np.angle(np.sum(w * np.conj(w_old)))
            return delta + phase_change + (np.pi / 2)

        return delta

class PolicyE_ProbeDiversity(PolicyD_StagnationEscapement):
    def step(self, t, delta, metrics, history, w, b, backbone):
        self.b_history.append(b.copy())
        self.w_history.append(w.copy())

        if len(history) < self.W:
            return delta

        window = history[-self.W:]
        sigma_null = np.std([h['null_depth'] for h in window])
        sigma_gain = np.std([h['gain'] for h in window])

        flips_total = 0
        for tau in range(len(self.b_history) - self.W, len(self.b_history)):
            flips = np.sum(self.b_history[tau] != self.b_history[tau-1])
            flips_total += flips
        avg_flips = flips_total / self.W

        if sigma_null < 0.5 and sigma_gain < 0.2 and avg_flips <= 1:
            # Trigger 12 point probe
            candidates = np.linspace(0, 2*np.pi, 12, endpoint=False)
            best_cand = None
            best_null = float('inf')

            for cand in candidates:
                w_probe, b_probe, null_res = backbone.run_probe(w, cand)
                D = np.sum(b_probe != b)

                if 0.1 * self.N <= D <= 0.4 * self.N:
                    if null_res < best_null:
                        best_null = null_res
                        best_cand = cand

            if best_cand is not None:
                return best_cand
            # If no candidate fits criteria, fall back to D's logic or stay?
            # Fall back to stay.
            return delta

        return delta

class PolicyF_SimulatedAnnealing(ControllerPolicy):
    def __init__(self, N: int):
        super().__init__(N)
        # To accept 3 dB deterioration with p=0.5 at t=1:
        # exp(-3 / T0) = 0.5 => -3 / T0 = ln(0.5) => T0 = -3 / ln(0.5) = 4.32
        self.T0 = 4.32
        self.last_E = None

    def step(self, t, delta, metrics, history, w, b, backbone):
        # We don't propose a new delta immediately, we wait until the next step to decide.
        # But wait, SA typically proposes BEFORE taking the step.
        # If we propose here, the outer loop will call backbone(w, proposed_delta), which executes it!
        # Then next time step() is called, we can see if it improved. If not, we might need to REVERT.
        # This breaks the simple outer loop abstraction unless the outer loop can revert states.
        # Let's handle SA by doing the lookahead inside step() using backbone run_step on a COPY.
        # BUT the backbone is stateful (last_delta). We need to be careful.

        current_E = -metrics['ptnr'] # E = -PTNR

        T_t = self.T0 * (0.85**t)

        # Propose
        cand_delta = delta + np.random.normal(0, (15 * np.pi / 180))

        # We need to evaluate cand_delta. We do this by asking backbone to run a step on a copy of w.
        w_new, b_new, cand_metrics = backbone.run_step(w.copy(), cand_delta)
        cand_E = -cand_metrics['ptnr']

        # Revert the backbone's last_delta state because we just used it for a test
        backbone.last_delta = delta

        if cand_E < current_E:
            return cand_delta
        else:
            if np.random.rand() < np.exp(-(cand_E - current_E) / max(T_t, 1e-6)):
                return cand_delta

        return delta

class PolicyG_TrustRegionAnnealing(ControllerPolicy):
    def __init__(self, N: int, W: int = 5):
        super().__init__(N)
        self.W = W
        self.sigma = 0.1 # start small
        self.last_E = float('inf')

    def step(self, t, delta, metrics, history, w, b, backbone):
        current_E = -metrics['ptnr']

        if current_E < self.last_E:
            # Success! Shrink sigma
            self.sigma *= 0.8
        self.last_E = current_E

        # Stagnation check
        if len(history) >= self.W:
            window = history[-self.W:]
            sigma_null = np.std([h['null_depth'] for h in window])
            if sigma_null < 0.05: # Stalled contraction
                self.sigma *= 4.0
                if np.random.rand() < 0.1:
                    cand_delta = delta + np.random.normal(0, np.pi/2)
                    return cand_delta

        cand_delta = delta + np.random.normal(0, self.sigma)

        # Test candidate
        w_new, b_new, cand_metrics = backbone.run_step(w.copy(), cand_delta)
        cand_E = -cand_metrics['ptnr']
        backbone.last_delta = delta # revert test side-effect

        if cand_E < current_E:
            return cand_delta

        return delta

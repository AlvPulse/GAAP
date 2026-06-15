import numpy as np
from typing import Tuple, List, Dict
from beamformer.element_model import ElementData
from beamformer.pattern_projection import Task
from beamformer.iterative import optimize_beam_ap

class OptimizationBackbone:
    """
    A strictly encapsulated wrapper for the AP + LR optimization process.
    The outer controller supplies a weight vector and a global offset delta.

    This backbone enforces the 3 critical phase jump steps:
      1) State Rotation
      2) Manifold Realignment
      3) Step-size reset

    Then it strictly executes exactly 1 AP step + 20 LR microsteps,
    returning the new state and metrics.
    """

    def __init__(self, task: Task, element: ElementData, N: int):
        self.task = task
        self.element = element
        self.N = N
        self.last_delta = None
        # We need to maintain the step_size for LR so it can adapt
        self.current_step_size = 0.05

    def run_step(self, w: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Executes exactly 1 macro iteration (1 AP + 20 LR steps).

        Args:
            w: Current complex weights.
            delta: Current proposed global phase offset.

        Returns:
            w_new: Updated complex weights after 1 AP + 20 LR.
            b_new: Branch assignment vector.
            metrics: Dict containing 'null_depth', 'gain', 'ptnr'.
        """

        # 1. State Rotation & Manifold Realignment & Step Size Reset if delta changed
        if self.last_delta is not None and np.abs(delta - self.last_delta) > 1e-6:
            # Step 1: State Rotation
            w_rot = w * np.exp(1j * (delta - self.last_delta))

            # Step 2: Manifold Realignment
            w_realigned = np.zeros_like(w_rot)
            for n in range(self.N):
                _, w_realigned[n], _ = self.element.project(w_rot[n], method='euclidean')

            w = w_realigned

            # Step 3: Step-Size Reset
            self.current_step_size = 0.05

        self.last_delta = delta

        # We now call the optimize_beam_ap, but we must strictly limit it to 1 iteration!
        # optimize_beam_ap handles both AP and LR, but it has its own loop.
        # We set K_max=1 to ensure it only runs 1 step.
        # We also pass the current_step_size.

        V_n, phi_0, c_n, history = optimize_beam_ap(
            task=self.task,
            element=self.element,
            N=self.N,
            initial_weights=w,
            initial_delta=delta,
            K_max=1,  # Strictly 1 iteration
            enable_ap=True,
            enable_local_refinement=True,
            refinement_steps_inner=20, # The requested 20 microsteps
            final_cleanup=False, # Don't run extra cleanup
            step_size=self.current_step_size
        )

        # Determine branches
        b_new = np.zeros(self.N, dtype=int)
        for n in range(self.N):
            _, _, b_new[n] = self.element.project(c_n[n], method='euclidean')

        # Get metrics
        from beamformer.controllers.metrics import get_metrics
        null_db, gain_db, ptnr_db = get_metrics(c_n, self.task, self.element)

        metrics = {
            'null_depth': null_db,
            'gain': gain_db,
            'ptnr': ptnr_db
        }

        return c_n, b_new, metrics

    def run_probe(self, w: np.ndarray, delta: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Executes exactly 2 AP steps with NO Local Refinement.
        This is for cheap exploration probes (Policy E).

        Returns:
            w_new: The complex weights after 2 AP steps.
            b_new: The resulting branch assignment vector.
            null_residual: The resulting null depth (or sparse residual).
        """

        # Rotate state for the probe candidate
        # We don't update self.last_delta because this is a temporary probe.
        current_delta = self.last_delta if self.last_delta is not None else 0.0
        w_rot = w * np.exp(1j * (delta - current_delta))

        w_realigned = np.zeros_like(w_rot)
        for n in range(self.N):
            _, w_realigned[n], _ = self.element.project(w_rot[n], method='euclidean')

        # Run 2 AP steps (K_max=2), disable LR
        V_n, phi_0, c_n, history = optimize_beam_ap(
            task=self.task,
            element=self.element,
            N=self.N,
            initial_weights=w_realigned,
            initial_delta=delta,
            K_max=2,
            enable_ap=True,
            enable_local_refinement=False,
            final_cleanup=False
        )

        b_new = np.zeros(self.N, dtype=int)
        for n in range(self.N):
            _, _, b_new[n] = self.element.project(c_n[n], method='euclidean')

        from beamformer.controllers.metrics import get_metrics
        null_db, gain_db, ptnr_db = get_metrics(c_n, self.task, self.element)

        return c_n, b_new, null_db

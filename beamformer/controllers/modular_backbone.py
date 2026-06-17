import numpy as np
from typing import Tuple, List, Dict
from beamformer.element_model import ElementData
from beamformer.pattern_projection import Task
from beamformer.iterative import optimize_beam_ap

class BackboneConfig:
    def __init__(self,
                 projection_type="hardware_coupled",
                 lr_call_frequency=1,
                 lr_active_set_size=8,
                 lr_step_size_init=1e-3,
                 preserve_gain_weight=0.5):
        self.projection_type = projection_type
        self.lr_call_frequency = lr_call_frequency
        self.lr_active_set_size = lr_active_set_size
        self.lr_step_size_init = lr_step_size_init
        self.preserve_gain_weight = preserve_gain_weight

class OptimizationBackbone:
    """
    A strictly encapsulated wrapper for the AP + LR optimization process.
    The outer controller supplies a weight vector and a global offset delta.

    This backbone enforces the 3 critical phase jump steps:
      1) State Rotation
      2) Manifold Realignment
      3) Step-size reset

    Then it strictly executes exactly 1 AP step + `lr_max_steps` LR microsteps,
    returning the new state and metrics.
    """

    def __init__(self, task: Task, element: ElementData, N: int, config: BackboneConfig = None):
        self.task = task
        self.element = element
        self.N = N
        self.config = config if config else BackboneConfig()
        self.last_delta = None
        # We need to maintain the step_size for LR so it can adapt
        self.current_step_size = self.config.lr_step_size_init

    def run_step(self, w: np.ndarray, delta: float, t: int = 1, autopsy_callback=None) -> Tuple[np.ndarray, np.ndarray, Dict]:
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
                _, w_realigned[n], _ = self.element.project(w_rot[n], method=self.config.projection_type)

            w = w_realigned

            # Step 3: Step-Size Reset
            self.current_step_size = self.config.lr_step_size_init

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
            enable_local_refinement=(t % self.config.lr_call_frequency == 0),
            refinement_steps_inner=20, # LR runs exactly 20 microsteps when called
            final_cleanup=False, # Don't run extra cleanup
            step_size=self.current_step_size,
            projection_method=self.config.projection_type,
            lr_objective="ptnr_constrained", # Enforce ptnr_constrained based on prompt
            k_active=self.config.lr_active_set_size,
            autopsy_callback=autopsy_callback,
            preserve_gain_weight=self.config.preserve_gain_weight # Pass through gain weight
        )

        # Determine branches
        b_new = np.zeros(self.N, dtype=int)
        for n in range(self.N):
            _, _, b_new[n] = self.element.project(c_n[n], method=self.config.projection_type)

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
            final_cleanup=False,
            projection_method=self.config.projection_type
        )

        b_new = np.zeros(self.N, dtype=int)
        for n in range(self.N):
            _, _, b_new[n] = self.element.project(c_n[n], method=self.config.projection_type)

        from beamformer.controllers.metrics import get_metrics
        null_db, gain_db, ptnr_db = get_metrics(c_n, self.task, self.element)

        return c_n, b_new, null_db

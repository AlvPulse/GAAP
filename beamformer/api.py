import numpy as np

from .pattern_projection import Task
from .element_model import ElementData, SyntheticVaractor
from .synthesis import synthesize_pencil, synthesize_sector, synthesize_multitarget
from .iterative import optimize_beam_ap
from .geometry import scan_offset_feasibility

def beamform(
    task: Task,
    element: ElementData,
    N: int,
    initial_delta: float = 0.5,
    K_max: int = 250,
    debug_dir: str = None,
    **kwargs
):
    """
    Top-level API for geometry-aware phase-only beamforming.
    """

    # 1. Warm start
    if task.type in ['pencil', 'nulled']:
        if len(task.target_angles) == 0:
            raise ValueError("Pencil beam requires at least one target_angle")
        initial_weights = synthesize_pencil(N, task.target_angles[0])
    elif task.type == 'sector':
        initial_weights = synthesize_sector(N, task.sector_u_min, task.sector_u_max)
    elif task.type == 'multitarget':
        initial_weights = synthesize_multitarget(N, task.target_angles, task.target_gains)
    else:
        raise ValueError(f"Unknown task type: {task.type}")

    # 2. Iterate
    V_n, final_delta, c_n, history = optimize_beam_ap(
        task=task,
        element=element,
        N=N,
        initial_weights=initial_weights,
        initial_delta=initial_delta,
        K_max=K_max,
        debug_dir=debug_dir,
        **kwargs
    )

    return {
        'voltages': V_n,
        'final_delta': final_delta,
        'weights': c_n,
        'history': history,
        'initial_weights': initial_weights
    }

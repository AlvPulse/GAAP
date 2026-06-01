import numpy as np

from .pattern_projection import Task
from .element_model import ElementData, SyntheticVaractor
from .synthesis import synthesize_pencil, synthesize_sector, synthesize_multitarget
from .iterative import optimize_beam_ap
from .geometry import scan_offset_feasibility
from .baseline_optimization import optimize_offset_only

def beamform(
    task: Task,
    element: ElementData,
    N: int,
    initial_delta: float = 0.0,
    K_max: int = 50,
    debug_dir: str = None,
    taper: str = None,
    sll_db: float = 30,
    projection_method: str = 'euclidean',
    w_phase: float = 1.0,
    w_amp: float = 0.5,
    enable_ap: bool = True,
    enable_offset: bool = True,
    enable_local_refinement: bool = False,
    **kwargs
):
    """
    Top-level API for geometry-aware phase-only beamforming.
    """

    # 1. Warm start
    if task.type == 'pencil':
        if len(task.target_angles) == 0:
            raise ValueError("Pencil beam requires at least one target_angle")
        initial_weights = synthesize_pencil(N, task.target_angles[0], taper_type=taper, sll_db=sll_db)
    elif task.type == 'nulled':
        if len(task.target_angles) == 0:
            raise ValueError("Nulled beam requires at least one target_angle")

        null_init_method = kwargs.get('null_init_method', 'schelkunoff')

        from .synthesis import synthesize_schelkunoff, synthesize_phase_only_lms

        if null_init_method == 'schelkunoff':
            initial_weights = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)
        elif null_init_method == 'lms':
            initial_weights = synthesize_phase_only_lms(N, task.target_angles[0], task.null_angles)
        else:
            initial_weights = synthesize_pencil(N, task.target_angles[0], taper_type=taper, sll_db=sll_db)

    elif task.type == 'sector':
        initial_weights = synthesize_sector(N, task.sector_u_min, task.sector_u_max, taper_type=taper, sll_db=sll_db)
    elif task.type == 'multitarget':
        initial_weights = synthesize_multitarget(N, task.target_angles, task.target_gains, taper_type=taper, sll_db=sll_db)
    else:
        raise ValueError(f"Unknown task type: {task.type}")

    # Stage 0: Geometry Selection
    if enable_offset:
        from .geometry_selection import select_optimal_geometry
        selected_delta = select_optimal_geometry(
            initial_weights, element, N,
            projection_method=projection_method
        )
    else:
        selected_delta = initial_delta

    # SOTA Baselines Intercept
    solver = kwargs.get('solver', 'ap')

    if solver == 'pgd':
        from .sota_solvers import solve_pgd
        V_n, c_n = solve_pgd(task, element, N, initial_weights)
        return {'voltages': V_n, 'weights': c_n, 'initial_weights': initial_weights, 'history': {'residuals': []}}

    elif solver == 'omp':
        from .sota_solvers import solve_omp_inspired
        V_n, c_n = solve_omp_inspired(task, element, N, initial_weights)
        return {'voltages': V_n, 'weights': c_n, 'initial_weights': initial_weights, 'history': {'residuals': []}}

    # Calculate optimal non-projection baseline for reference
    baseline_delta, baseline_c_n, baseline_V_n = optimize_offset_only(
        initial_weights, element,
        projection_method=projection_method, w_phase=w_phase, w_amp=w_amp
    )

    # Stage 1 & 2: Iterate AP Feasibility and Local Refinement
    V_n, final_delta, c_n, history = optimize_beam_ap(
        task=task,
        element=element,
        N=N,
        initial_weights=initial_weights,
        initial_delta=selected_delta, # Now we use the fixed selected delta
        K_max=K_max,
        debug_dir=debug_dir,
        projection_method=projection_method,
        w_phase=w_phase,
        w_amp=w_amp,
        enable_ap=enable_ap,
        enable_local_refinement=enable_local_refinement,
        baseline_weights=baseline_c_n,
        **kwargs
    )

    return {
        'voltages': V_n,
        'final_delta': final_delta,
        'weights': c_n,
        'history': history,
        'initial_weights': initial_weights,
        'baseline_weights': baseline_c_n,
        'baseline_voltages': baseline_V_n,
        'baseline_delta': baseline_delta
    }

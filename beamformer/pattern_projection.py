import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Callable

@dataclass
class Task:
    """Unified specification for beamforming tasks."""
    type: str  # 'pencil', 'sector', 'multitarget', 'nulled'
    target_angles: List[float] = field(default_factory=list) # in u-space
    target_gains: List[float] = field(default_factory=list)  # linear magnitude
    null_angles: List[float] = field(default_factory=list)   # in u-space
    null_depth: float = 0.00316  # -50 dB default
    sll_ceiling: float = 0.1     # -20 dB default
    beamwidth: float = 0.1       # u-space half-width of main lobe

    sector_u_min: Optional[float] = None
    sector_u_max: Optional[float] = None

def pattern_project(F, u_grid, task: Task):
    """
    Projects the current pattern F onto the task constraints.
    Critically: only modifies magnitude where necessary, preserving phase.
    """
    F_new = F.copy()
    mag_F = np.abs(F)

    # Avoid div by zero for phase
    phase_F = F / (mag_F + 1e-12)

    # Pre-calculate regions
    # Main lobe mask for pencil beams
    is_main_lobe = np.zeros_like(u_grid, dtype=bool)
    if task.type in ['pencil', 'nulled']:
        for u_target in task.target_angles:
            is_main_lobe |= (np.abs(u_grid - u_target) <= task.beamwidth)

    elif task.type == 'sector':
        is_main_lobe = (u_grid >= task.sector_u_min) & (u_grid <= task.sector_u_max)

    elif task.type == 'multitarget':
        for u_target in task.target_angles:
            is_main_lobe |= (np.abs(u_grid - u_target) <= task.beamwidth)

    # 1. Enforce SLL ceiling in sidelobe regions
    sidelobe_region = ~is_main_lobe
    violates_sll = sidelobe_region & (mag_F > task.sll_ceiling)
    F_new[violates_sll] = task.sll_ceiling * phase_F[violates_sll]

    # 2. Enforce target gains in main lobe (if specified)
    if task.target_gains:
        for u_t, g in zip(task.target_angles, task.target_gains):
            # Find closest grid point
            idx = np.argmin(np.abs(u_grid - u_t))
            if mag_F[idx] < g:
                F_new[idx] = g * phase_F[idx]

    # 3. Enforce Nulls
    for u_null in task.null_angles:
        # Find closest grid point
        idx = np.argmin(np.abs(u_grid - u_null))
        if mag_F[idx] > task.null_depth:
            # Soft null with magnitude ceiling
            F_new[idx] = task.null_depth * phase_F[idx]

    return F_new

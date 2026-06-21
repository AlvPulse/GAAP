"""
Benchmark: MR-LCMV (three refine modes) vs the legacy AP backbone and SOTA solvers.

Demonstrates the core claim: the AP backbone sacrifices main-beam gain (it minimizes
a pattern residual, not gain), while MR-LCMV stays at the hardware gain ceiling and
GLCP digs deep nulls there. Run:

    ./.venv/Scripts/python.exe experiments/benchmark_mrlcmv.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer.api import beamform
from beamformer.controllers.metrics import get_metrics, get_sll


def gain_ceiling(element, N, u_t):
    """Best achievable coherent gain (dB) with no nulls -- the hardware limit."""
    n = np.arange(N)
    s_t = np.exp(-1j * np.pi * n * u_t)
    cg = element.c_grid
    best = -np.inf
    for psi in np.linspace(0, 2 * np.pi, 361):
        tgt = np.exp(1j * psi) * s_t
        c = cg[np.argmax(np.real(np.conj(tgt)[:, None] * cg[None, :]), axis=1)]
        g = 20 * np.log10(abs(np.sum(c * np.exp(1j * np.pi * n * u_t))) / N + 1e-12)
        best = max(best, g)
    return best


def run_case(name, N, task, element):
    u_t = task.target_angles[0]
    ceil = gain_ceiling(element, N, u_t)
    print(f"\n=== {name}   (N={N}, target u={u_t}, nulls={list(task.null_angles)}) ===")
    print(f"    hardware gain ceiling (no nulls): {ceil:6.2f} dB")
    header = f"{'method':28s} {'gain dB':>8s} {'null dB':>9s} {'PTNR dB':>8s} {'SLL/pk':>8s} {'ms':>6s}"
    print(header)
    print("-" * len(header))

    configs = [
        ("AP euclidean",            dict(solver="ap", projection_method="euclidean")),
        ("AP weighted",             dict(solver="ap", projection_method="weighted")),
        ("AP weighted + LR",        dict(solver="ap", projection_method="weighted",
                                          enable_local_refinement=True)),
        ("PGD (sota)",              dict(solver="pgd")),
        ("OMP (sota)",              dict(solver="omp")),
        ("MR-LCMV (none)",          dict(solver="mrlcmv", refine="none")),
        ("MR-LCMV -> AP+LR warm",   dict(solver="mrlcmv", refine="ap_lr")),
        ("MR-LCMV + GLCP  *NEW*",   dict(solver="mrlcmv", refine="glcp")),
    ]

    peak_abs_ref = None
    for label, cfg in configs:
        try:
            t = time.perf_counter()
            res = beamform(task, element, N=N, K_max=80,
                           null_init_method="schelkunoff", use_pattern_cost=True,
                           w_phase=1.0, w_amp=0.5, **cfg)
            dt = (time.perf_counter() - t) * 1e3
            c = res["weights"]
            nd, ng, ptnr = get_metrics(c, task, element)
            sll_abs = get_sll(c, task)
            peak_abs = ng + 20 * np.log10(N)
            sll = sll_abs - peak_abs  # SLL relative to main-beam peak
            print(f"{label:28s} {ng:8.2f} {nd:9.2f} {ptnr:8.2f} {sll:8.1f} {dt:6.0f}")
        except Exception as e:
            print(f"{label:28s}  ERROR: {repr(e)[:60]}")


def main():
    element = SyntheticVaractor(beta=0.8, folding=True)  # severe amplitude-phase coupling

    cases = [
        ("Canonical null", 64, Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])),
        ("Three nulls",    64, Task("nulled", target_angles=[0.0], null_angles=[-0.5, -0.25, 0.4])),
        ("Steered, small", 32, Task("nulled", target_angles=[-0.3], null_angles=[0.1, 0.5])),
    ]
    for name, N, task in cases:
        run_case(name, N, task, element)

    print("\nKey: gain dB = normalized main-beam gain (0 dB = ideal coherent N).")
    print("     MR-LCMV+GLCP should sit at the gain ceiling with the deepest nulls.")


if __name__ == "__main__":
    main()

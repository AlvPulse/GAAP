"""
MR-LCMV nullforming example: full-height main beam with deep nulls, contrasted
against the legacy AP backbone that collapses gain.

    ./.venv/Scripts/python.exe examples/mrlcmv_nullforming.py
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer.api import beamform
from beamformer.utils import oversampled_fft
from beamformer.controllers.metrics import get_metrics


def pattern_db(c_n, oversample=16):
    u, F = oversampled_fft(c_n, oversample)
    # absolute dB relative to ideal coherent N so curves are comparable in *gain*
    return u, 20 * np.log10(np.abs(F) + 1e-12) - 20 * np.log10(len(c_n))


def main():
    N = 64
    task = Task("nulled", target_angles=[0.2], null_angles=[-0.4, 0.5])
    element = SyntheticVaractor(beta=0.8, folding=True)

    runs = {
        "AP (weighted)":        dict(solver="ap", projection_method="weighted"),
        "AP weighted + LR":     dict(solver="ap", projection_method="weighted",
                                     enable_local_refinement=True),
        "MR-LCMV + GLCP (new)": dict(solver="mrlcmv", refine="glcp"),
    }
    colors = {"AP (weighted)": "tab:orange", "AP weighted + LR": "tab:red",
              "MR-LCMV + GLCP (new)": "tab:green"}

    plt.figure(figsize=(11, 6))
    for label, cfg in runs.items():
        res = beamform(task, element, N=N, K_max=80, null_init_method="schelkunoff",
                       use_pattern_cost=True, w_phase=1.0, w_amp=0.5, **cfg)
        u, pdb = pattern_db(res["weights"])
        nd, ng, ptnr = get_metrics(res["weights"], task, element)
        plt.plot(u, pdb, color=colors[label], lw=1.8,
                 label=f"{label}: gain={ng:.1f} dB, null={nd:.0f} dB")

    plt.axvline(task.target_angles[0], color="k", ls=":", lw=1, label="target")
    for nu in task.null_angles:
        plt.axvline(nu, color="gray", ls="-.", lw=1,
                    label="null" if nu == task.null_angles[0] else None)
    plt.axhline(0, color="k", lw=0.6)
    plt.ylim(-70, 3)
    plt.xlim(-1, 1)
    plt.xlabel("u = sin(theta)")
    plt.ylabel("Gain relative to ideal coherent N (dB)")
    plt.title(f"MR-LCMV vs AP  (N={N}, lossy varactor manifold)")
    plt.legend(loc="lower center", fontsize=9)
    plt.grid(True, alpha=0.3)
    out = os.path.join(os.path.dirname(__file__), "mrlcmv_nullforming_result.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Fig. 6 — Complex-plane locus of the over-range WBRO RTPS at 2.455 GHz,
         with convex hull and the ideal unit circle.

Purpose
-------
The convex-hull perimeter ratio  P / P_ideal  (where P_ideal = 2*pi for the
unit circle) is the geometric figure of merit the paper uses to bound
nullforming reachability.  This script computes the ratio directly from the
measured (amplitude, phase) data so the number quoted in the paper can be
audited.

Data source
-----------
Raw files:
    VNA Results (1)/WNBRO_phase.mat       (variable WBRO_phase, shape (3, 61))
    VNA Results (1)/WBRO_amplitude.mat    (variable WBRO_amplitude, shape (3, 61))

Row 2 of each matrix corresponds to 2.455 GHz.  The complex weight at bias V_k
is reconstructed as:
    w_k = a_k * exp(j * phi_k * pi/180)
where a_k = WBRO_amplitude[2, k]  and  phi_k = WBRO_phase[2, k].

The convex hull is computed by scipy.spatial.ConvexHull.  The perimeter is
the sum of edge lengths.  The ratio P / P_ideal is reported on stdout.

Usage
-----
    python plot_fig6_locus_convex_hull.py [--phase PATH] [--amp PATH] [--freq 2.455] [--out PATH]

Outputs
-------
    ../figures/fig6_locus_convex_hull_2p455GHz.png
    ../figures/fig6_locus_convex_hull_2p455GHz.pdf
    ../figures/fig6_locus_convex_hull_2p080GHz.png   (if --freq 2.080)
    ../figures/fig6_locus_convex_hull_1p690GHz.png   (if --freq 1.690)

Verification
------------
Prints:
  - amplitude range
  - phase range
  - hull perimeter  P
  - ideal perimeter 2*pi
  - ratio P / (2*pi)
The paper claims P/P_ideal = 0.97 at 2.45 GHz.  The number printed by this
script is the auditable value.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.spatial import ConvexHull

HERE = Path(__file__).resolve().parent
DEFAULT_PHASE = "WNBRO_phase.mat"
DEFAULT_AMP = "WBRO_amplitude.mat"
DEFAULT_OUT_DIR =  "figures"

FREQ_INDEX = {"1.690": 0, "2.080": 1, "2.455": 2}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--phase", type=Path, default=DEFAULT_PHASE)
    p.add_argument("--amp",   type=Path, default=DEFAULT_AMP)
    p.add_argument("--freq", choices=list(FREQ_INDEX.keys()), default="2.080",
                   help="frequency index to plot (default 2.455 GHz)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    if not args.phase.exists() or not args.amp.exists():
        print(f"ERROR: .mat file(s) not found:\n  {args.phase}\n  {args.amp}",
              file=sys.stderr)
        return 1

    phase = sio.loadmat(str(args.phase))["WBRO_phase"]
    amp_in = sio.loadmat(str(args.amp))["WBRO_amplitude"]
    # Don't normalize globally - normalize only the selected frequency row
    idx = FREQ_INDEX[args.freq]
    a = amp_in[idx]
    #a = a / np.max(a)  # this frequency's max becomes exactly 1.0
    # print(phase[idx])
    a= a/a[np.argmin(np.abs(phase[idx]-360))]
    # if phase.shape != amp.shape or phase.shape != (3, 61):
    #     print(f"ERROR: unexpected shapes phase={phase.shape} amp={amp.shape}",
    #           file=sys.stderr)
    #     return 1

    phi = phase[idx]
    w = a * np.exp(1j * phi * np.pi / 180.0)

    # Convex hull
    pts = np.column_stack([w.real, w.imag])
    hull = ConvexHull(pts)
    hull_pts = pts[hull.vertices]
    # Perimeter = sum of edge lengths
    edges = np.diff(np.vstack([hull_pts, hull_pts[:1]]), axis=0)
    P = float(np.sum(np.hypot(edges[:, 0], edges[:, 1])))
    P_ideal = 2.0 * np.pi
    ratio = P / P_ideal

    # Verification printout
    print(f"\n--- Verification @ {args.freq} GHz ---")
    print(f"  amplitude range: {a.min():.4f} - {a.max():.4f}  "
          f"({20*np.log10(a.min()):.2f} to {20*np.log10(a.max()):.2f} dB)")
    print(f"  phase range:     {phi.min():.2f} - {phi.max():.2f} deg  "
          f"(excursion {phi.max()-phi.min():.2f} deg)")
    print(f"  convex hull perimeter P = {P:.4f}")
    print(f"  ideal unit-circle perimeter 2*pi = {P_ideal:.4f}")
    print(f"  ratio P / P_ideal = {ratio:.4f}")
    print(f"  Paper claim: P/P_ideal = 0.97 at 2.45 GHz.  "
          f"Match: {'YES' if abs(ratio - 0.97) < 0.03 else 'NO'}")

    # ----------------------- plot ----------------------- #
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.linewidth": 0.8, "figure.dpi": 110,
    })
    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    theta = np.linspace(0, 2 * np.pi, 401)
    ax.plot(np.cos(theta), np.sin(theta), "--", color="gray",
            linewidth=1.0, label="Ideal unit circle")

    ax.plot(w.real, w.imag, "-", color="#1f4e79", linewidth=1.6,
            label="Measured locus")
    ax.scatter(w.real, w.imag, c=range(len(w)), cmap="viridis",
               s=18, zorder=4, label="Bias sweep")

    # Convex hull (closed polygon)
    ax.fill(hull_pts[:, 0], hull_pts[:, 1], facecolor="#fbb6a0",
            edgecolor="#c0392b", linewidth=1.4, alpha=0.35,
            label=f"Convex hull  ")

    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Re($w$)")
    ax.set_ylabel("Im($w$)")
    ax.set_title(f"Over-range RTPS measured locus")
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    ax.grid(True, alpha=0.3, linewidth=0.5)

    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = {"1.690": "1p690", "2.080": "2p080", "2.455": "2p455"}[args.freq]
    png = args.out / f"fig6_locus_convex_hull_{suffix}GHz.png"
    pdf = args.out / f"fig6_locus_convex_hull_{suffix}GHz.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")
    print(f"  wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Compute convex hull at multiple frequencies and across the full band
to address the supervisor's concern: '6 dB loss for some voltages across
the bandwidth'. Also generate the Smith-chart source data.
"""
import os, glob, re, json
import numpy as np
from scipy.spatial import ConvexHull
from vna_analyze import parse_s2p, voltage_from_name, VNA

OUT = "/home/z/my-project/download/vna_figs"

# Load directory 1 (best prototype)
files = sorted(glob.glob(os.path.join(VNA, "1", "*.S2P")))
print(f"Loading {len(files)} S2P files from directory 1...")

# Frequency grid: 1.0 to 4.0 GHz, 201 points
FREQS_HZ = np.linspace(1.0e9, 4.0e9, 201)
FREQS_GHZ = FREQS_HZ / 1e9

# For each bias voltage, load S-parameters and interpolate to FREQS_HZ
records = []  # list of (voltage, S11(freqs), S21(freqs))
for fp in files:
    v = voltage_from_name(os.path.basename(fp))
    if v is None:
        continue
    freqs_file, s11_file, s21_file, s12_file, s22_file = parse_s2p(fp)
    if len(freqs_file) == 0:
        continue
    # interpolate to FREQS_HZ
    s11_grid = np.array([complex(np.interp(f, freqs_file, s11_file.real),
                                  np.interp(f, freqs_file, s11_file.imag)) for f in FREQS_HZ])
    s21_grid = np.array([complex(np.interp(f, freqs_file, s21_file.real),
                                  np.interp(f, freqs_file, s21_file.imag)) for f in FREQS_HZ])
    records.append((v, s11_grid, s21_grid))

# sort by voltage
records.sort(key=lambda r: r[0])
voltages = np.array([r[0] for r in records])
S11_grid = np.stack([r[1] for r in records])  # shape (V, F)
S21_grid = np.stack([r[2] for r in records])  # shape (V, F)
print(f"Loaded: {len(voltages)} bias points x {len(FREQS_HZ)} freq points")
print(f"Voltage range: {voltages.min()} - {voltages.max()} V")

# === Hull perimeter as a function of frequency ===
hull_perim_freq = np.zeros(len(FREQS_HZ))
hull_area_freq = np.zeros(len(FREQS_HZ))
phase_range_freq = np.zeros(len(FREQS_HZ))
il_min_freq = np.zeros(len(FREQS_HZ))
il_max_freq = np.zeros(len(FREQS_HZ))
il_avg_freq = np.zeros(len(FREQS_HZ))

for fi, f in enumerate(FREQS_HZ):
    pts = S21_grid[:, fi]  # complex array, one per voltage
    mags = np.abs(pts)
    phases = np.unwrap(np.angle(pts))
    phase_range_freq[fi] = np.degrees(phases[-1] - phases[0])
    il_min_freq[fi] = 20 * np.log10(mags.min() + 1e-12)
    il_max_freq[fi] = 20 * np.log10(mags.max() + 1e-12)
    il_avg_freq[fi] = 20 * np.log10(np.mean(mags) + 1e-12)
    if len(pts) >= 3 and mags.min() > 1e-6:
        try:
            hull = ConvexHull(np.column_stack([pts.real, pts.imag]))
            hull_perim_freq[fi] = hull.area
            hull_area_freq[fi] = hull.volume
        except Exception:
            pass

hull_ratio_freq = hull_perim_freq / (2 * np.pi)

# Print key frequencies
key_freqs = [2.0, 2.45, 2.9]  # start/mid/end of operating band
print("\n" + "=" * 80)
print("CONVEX HULL vs FREQUENCY (S21 locus, directory 1)")
print("=" * 80)
print(f"{'Freq (GHz)':<12}{'Phase range':<14}{'IL min':<10}{'IL max':<10}{'IL avg':<10}{'Hull perim':<12}{'Hull ratio':<10}")
for fg in [1.0, 1.5, 2.0, 2.45, 2.9, 3.0, 3.5, 4.0]:
    fi = int(round((fg - 1.0) / 3.0 * 200))
    print(f"{fg:<12.2f}{phase_range_freq[fi]:<14.1f}{il_min_freq[fi]:<10.2f}{il_max_freq[fi]:<10.2f}{il_avg_freq[fi]:<10.2f}{hull_perim_freq[fi]:<12.3f}{hull_ratio_freq[fi]:<10.3f}")

# Specifically check the operating band 2.0-2.9 GHz (37% BW around 2.45)
band_mask = (FREQS_GHZ >= 2.0) & (FREQS_GHZ <= 2.9)
print(f"\nOperating band 2.0-2.9 GHz:")
print(f"  Hull ratio: min={hull_ratio_freq[band_mask].min():.3f}  max={hull_ratio_freq[band_mask].max():.3f}  mean={hull_ratio_freq[band_mask].mean():.3f}")
print(f"  IL avg:     min={il_avg_freq[band_mask].min():.2f}  max={il_avg_freq[band_mask].max():.2f}  mean={il_avg_freq[band_mask].mean():.2f} dB")
print(f"  IL worst (min across band+voltages): {il_min_freq[band_mask].min():.2f} dB")
print(f"  Phase range: min={phase_range_freq[band_mask].min():.1f}  max={phase_range_freq[band_mask].max():.1f}")

# === Per-voltage IL across the band (supervisor's '6 dB loss' comment) ===
print("\n" + "=" * 80)
print("PER-VOLTAGE IL across 2.0-2.9 GHz band (supervisor's '6 dB loss' check)")
print("=" * 80)
print(f"{'Voltage (V)':<14}{'IL min':<10}{'IL max':<10}{'IL swing':<10}{'IL avg':<10}")
for vi, v in enumerate(voltages):
    if v % 1.0 < 0.05 or abs(v - round(v)) < 0.05:  # print integer voltages
        mags_band = np.abs(S21_grid[vi, band_mask])
        il_band = 20 * np.log10(mags_band + 1e-12)
        print(f"{v:<14.1f}{il_band.min():<10.2f}{il_band.max():<10.2f}{(il_band.max()-il_band.min()):<10.2f}{il_band.mean():<10.2f}")

# === Save the data ===
np.savez(f"{OUT}/vna_full.npz",
         voltages=voltages,
         freqs_hz=FREQS_HZ,
         freqs_ghz=FREQS_GHZ,
         S11_grid=S11_grid,
         S21_grid=S21_grid,
         hull_perim_freq=hull_perim_freq,
         hull_area_freq=hull_area_freq,
         hull_ratio_freq=hull_ratio_freq,
         phase_range_freq=phase_range_freq,
         il_min_freq=il_min_freq,
         il_max_freq=il_max_freq,
         il_avg_freq=il_avg_freq)

# Save the key numbers for the paper
summary = {
    "prototype_dir": "1",
    "voltage_count": int(len(voltages)),
    "voltage_min_V": float(voltages.min()),
    "voltage_max_V": float(voltages.max()),
    "freq_grid_GHz_min": 1.0,
    "freq_grid_GHz_max": 4.0,
    "freq_grid_points": 201,
    "operating_band_GHz": [2.0, 2.9],
    "operating_band_frac_BW": 0.37,
    "center_freq_GHz": 2.45,
    "at_2.45GHz": {
        "phase_range_deg": float(phase_range_freq[int(round((2.45-1.0)/3.0*200))]),
        "il_min_dB": float(il_min_freq[int(round((2.45-1.0)/3.0*200))]),
        "il_max_dB": float(il_max_freq[int(round((2.45-1.0)/3.0*200))]),
        "il_avg_dB": float(il_avg_freq[int(round((2.45-1.0)/3.0*200))]),
        "hull_perim": float(hull_perim_freq[int(round((2.45-1.0)/3.0*200))]),
        "hull_ratio": float(hull_ratio_freq[int(round((2.45-1.0)/3.0*200))]),
    },
    "band_2.0_2.9_stats": {
        "hull_ratio_min": float(hull_ratio_freq[band_mask].min()),
        "hull_ratio_max": float(hull_ratio_freq[band_mask].max()),
        "hull_ratio_mean": float(hull_ratio_freq[band_mask].mean()),
        "il_avg_min_dB": float(il_avg_freq[band_mask].min()),
        "il_avg_max_dB": float(il_avg_freq[band_mask].max()),
        "il_avg_mean_dB": float(il_avg_freq[band_mask].mean()),
        "il_worst_dB": float(il_min_freq[band_mask].min()),
        "phase_range_min_deg": float(phase_range_freq[band_mask].min()),
        "phase_range_max_deg": float(phase_range_freq[band_mask].max()),
    },
}
with open(f"{OUT}/vna_paper_numbers.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved full NPZ and paper-number JSON to {OUT}/")
print(json.dumps(summary, indent=2))
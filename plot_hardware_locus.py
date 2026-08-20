import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import os
import sys

sys.path.append('.')
from beamformer.element_model import MeasuredVaractor

os.makedirs("experiments/figures_measured", exist_ok=True)

# Load the hardware model
el_folded = MeasuredVaractor(folding=True)
el_360 = MeasuredVaractor(folding=False)

c_folded = np.asarray(el_folded.c_grid)
c_360 = np.asarray(el_360.c_grid)

points_folded = np.column_stack((c_folded.real, c_folded.imag))
hull = ConvexHull(points_folded)

# IEEE Formatting
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'font.family': 'serif',
    'pdf.fonttype': 42
})

fig, ax = plt.subplots(figsize=(3.5, 3.5))

# Plot ideal circle
theta = np.linspace(0, 2*np.pi, 200)
ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, label='Ideal Circle ($|c|=1$)')

# Plot the 360 slice
ax.plot(c_360.real, c_360.imag, color='blue', linewidth=2, label=r'Measured $360^{\circ}$ locus')

# Plot the folded over-range extension
# Assuming the rest of the array beyond 360 length is the extension
ext_len = len(c_folded) - len(c_360)
if ext_len > 0:
    c_ext = c_folded[len(c_360):]
    ax.plot(c_ext.real, c_ext.imag, color='red', linewidth=1.5, linestyle='-.', label=r'Over-range locus ($>360^{\circ}$)')

# Plot Convex Hull
for simplex in hull.simplices:
    ax.plot(points_folded[simplex, 0], points_folded[simplex, 1], 'g-', linewidth=1, alpha=0.6)
# Custom legend entry for hull
ax.plot([], [], 'g-', linewidth=1, alpha=0.6, label='Convex Hull ($c_W$)')

ax.set_aspect('equal', 'box')
ax.set_xlim([-1.1, 1.1])
ax.set_ylim([-1.1, 1.1])
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.grid(True, linestyle='--', alpha=0.5)

plt.legend(loc='lower left', framealpha=0.9, edgecolor='0.8')
plt.tight_layout(pad=0.5)
plt.savefig("experiments/figures_measured/fig1_measured_hardware_locus_IEEE.pdf", format='pdf', bbox_inches='tight')

print("Generated Hardware VNA Locus plot: experiments/figures_measured/fig1_measured_hardware_locus_IEEE.pdf")

import pandas as pd
import matplotlib.pyplot as plt
import os

os.makedirs("experiments/figures_measured", exist_ok=True)

df = pd.read_csv("experiments/results/convergence_scaling.csv")

# IEEE Double-Column Formatting Standards
# Column width is typically ~3.5 inches
fig_width = 3.5
fig_height = 2.8

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'lines.linewidth': 1.5,
    'lines.markersize': 5,
    'font.family': 'serif',
    'pdf.fonttype': 42  # TrueType fonts for IEEE compliance
})

# Plot 2: Final worst-null depth vs N
plt.figure(figsize=(fig_width, fig_height))
for mode in df['Mode'].unique():
    d = df[df['Mode'] == mode]
    plt.plot(d['N'], d['Null'], marker='o', label=mode)
plt.xlabel("Array Size ($N$)")
plt.ylabel("Worst Null Depth (dB)")
plt.xscale('log', base=2)
plt.xticks(df['N'].unique(), labels=[str(int(n)) for n in df['N'].unique()])
plt.legend(loc='best', framealpha=0.9, edgecolor='0.8')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=0.5)
plt.savefig("experiments/figures_measured/plot2_null_vs_n_IEEE.pdf", format='pdf', bbox_inches='tight')
plt.close()

# Plot 3: Iterations-to-convergence vs N
plt.figure(figsize=(fig_width, fig_height))
cert = df[df['Mode'] == 'Certified']
adapt = df[df['Mode'] == 'Adaptive']
if not cert.empty and not adapt.empty:
    plt.plot(cert['N'], cert['K'], marker='^', label='Certified Limit', color='#2ca02c')
    plt.plot(adapt['N'], adapt['K'], marker='s', label='Adaptive Practical', color='#ff7f0e')
    plt.xlabel("Array Size ($N$)")
    plt.ylabel("Dual Iterations ($K$)")
    plt.xscale('log', base=2)
    plt.xticks(df['N'].unique(), labels=[str(int(n)) for n in df['N'].unique()])
    plt.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(pad=0.5)
    plt.savefig("experiments/figures_measured/plot3_iterations_vs_n_IEEE.pdf", format='pdf', bbox_inches='tight')
plt.close()

# Plot 4: Worst-null depth vs full AF evaluations
plt.figure(figsize=(fig_width, fig_height))
markers = ['o', '^', 's', 'D', 'v', 'p']
for idx, mode in enumerate(df['Mode'].unique()):
    d = df[df['Mode'] == mode]
    plt.scatter(d['Evals'], d['Null'], label=mode, marker=markers[idx % len(markers)], alpha=0.8)
plt.xlabel("Full AF Evaluations ($N_{AF}$)")
plt.ylabel("Worst Null Depth (dB)")
plt.xscale('log')
plt.legend(loc='lower right', framealpha=0.9, edgecolor='0.8')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=0.5)
plt.savefig("experiments/figures_measured/plot4_null_vs_evals_IEEE.pdf", format='pdf', bbox_inches='tight')
plt.close()

# Plot 5: Equivalent cost / runtime scaling vs N
plt.figure(figsize=(fig_width, fig_height))
for idx, mode in enumerate(df['Mode'].unique()):
    d = df[df['Mode'] == mode]
    plt.plot(d['N'], d['Time'], marker=markers[idx % len(markers)], label=mode)
plt.xlabel("Array Size ($N$)")
plt.ylabel("Runtime (ms)")
plt.xscale('log', base=2)
plt.xticks(df['N'].unique(), labels=[str(int(n)) for n in df['N'].unique()])
plt.yscale('log')
plt.legend(loc='upper left', framealpha=0.9, edgecolor='0.8')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout(pad=0.5)
plt.savefig("experiments/figures_measured/plot5_runtime_vs_n_IEEE.pdf", format='pdf', bbox_inches='tight')
plt.close()

print("IEEE-styled PDFs successfully generated in experiments/figures_measured/")

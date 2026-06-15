import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    df_main = pd.read_csv('experiments/data/main_benchmark_policies.csv')
    try:
        df_scale = pd.read_csv('experiments/data/scaling_benchmark_policies.csv')
    except:
        df_scale = None
    return df_main, df_scale

def plot_metrics(df_main, df_scale):
    fig, axs = plt.subplots(3, 2, figsize=(16, 18))
    axs = axs.flatten()

    policies = df_main['Policy'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(policies)))

    # Mappings
    metrics = [
        ('NullDepth', 'Null Depth (dB)', 'Lower is better', axs[0], -40),
        ('Gain', 'Normalized Main-Beam Gain (dB)', 'Higher is better (Closer to 0)', axs[1], 0),
        ('PTNR', 'Peak-to-Null Ratio (dB)', 'Higher is better', axs[2], None),
        ('Residual', 'Manifold Projection Residual', 'Lower is better', axs[3], None),
        ('CumulativeFlips', 'Cumulative Branch Flips', 'Stagnation check', axs[4], None)
    ]

    iterations = np.sort(df_main['Iteration'].unique())

    for metric, ylabel, note, ax, ref_line in metrics:
        for i, policy in enumerate(policies):
            df_p = df_main[df_main['Policy'] == policy]
            means = []
            lower = []
            upper = []
            for t in iterations:
                vals = df_p[df_p['Iteration'] == t][metric]
                means.append(vals.mean())
                lower.append(vals.quantile(0.25))
                upper.append(vals.quantile(0.75))

            ax.plot(iterations, means, label=policy, color=colors[i], linewidth=2)
            ax.fill_between(iterations, lower, upper, color=colors[i], alpha=0.2)

        if ref_line is not None:
            ax.axhline(ref_line, color='k', linestyle='--', linewidth=1.5, label='Target')

        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{ylabel} vs. Iteration')
        ax.grid(True, linestyle='--', alpha=0.6)
        if metric == 'NullDepth':
            ax.legend(loc='upper right', fontsize=9)

    # Plot 6: Scaling
    ax = axs[5]
    if df_scale is not None:
        scales = np.sort(df_scale['N'].unique())
        scale_policies = df_scale['Policy'].unique()

        for i, policy in enumerate(scale_policies):
            df_p = df_scale[df_scale['Policy'] == policy]
            means = []
            stds = []
            for n in scales:
                vals = df_p[df_p['N'] == n]['WallClock_s']
                means.append(vals.mean())
                stds.append(vals.std())

            # Find matching color
            c_idx = list(policies).index(policy)
            ax.errorbar(scales, means, yerr=stds, label=policy, color=colors[c_idx], marker='o', capsize=5, linewidth=2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(scales)
        ax.set_xticklabels([str(n) for n in scales])
        ax.set_xlabel('Array Scale N')
        ax.set_ylabel('Wall-Clock Time (s)')
        ax.set_title('Processing Time vs. Array Scale')
        ax.grid(True, which="both", ls="-", alpha=0.5)
        ax.legend(loc='upper left', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Scaling Data Not Available', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig('experiments/Final_6Panel_Benchmark.png', dpi=300)
    print("Saved Final_6Panel_Benchmark.png")

if __name__ == '__main__':
    df_main, df_scale = load_data()
    plot_metrics(df_main, df_scale)

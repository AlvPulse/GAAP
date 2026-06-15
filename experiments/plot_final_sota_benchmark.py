import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    df = pd.read_csv('experiments/data/sota_benchmark_final.csv')
    return df

def plot_performance_metrics(df):
    algorithms = df['method'].unique()
    scales = sorted(df['N'].unique())

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    metrics = [
        ('PTNR', 'Peak-to-Null Ratio (dB)', 'Higher is better'),
        ('null_depth_dB', 'Null Depth (dB)', 'Lower is better'),
        ('gain_loss_dB', 'Mainlobe Gain Loss (dB)', 'Lower is better'),
        ('wall_clock_s', 'Convergence Time (s)', 'Lower is better')
    ]

    for i, (metric, ylabel, note) in enumerate(metrics):
        for algo in algorithms:
            df_algo = df[df['method'] == algo]
            means = []
            stds = []
            for n in scales:
                data = df_algo[df_algo['N'] == n][metric]
                means.append(data.mean())
                stds.append(data.std())

            axs[i].errorbar(scales, means, yerr=stds, label=algo, marker='o', capsize=5, linewidth=2)

        axs[i].set_xscale('log')
        axs[i].set_xticks(scales)
        axs[i].set_xticklabels([str(n) for n in scales])
        axs[i].set_xlabel('Array Size $N$')
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(f'{ylabel} vs N ({note})')
        axs[i].grid(True, which="both", ls="-", alpha=0.5)

        # Only log scale time
        if metric == 'wall_clock_s':
            axs[i].set_yscale('log')

        if i == 0:
            axs[i].legend(fontsize='small')

    plt.tight_layout()
    plt.savefig('experiments/SOTA_Performance_Scaling.png')
    print("Saved SOTA Performance Scaling plot.")

def plot_success_profiles(df):
    algorithms = df['method'].unique()

    # We will just plot success probability averaged across all N and seeds for different thresholds
    thresholds = ['success_20dB', 'success_30dB', 'success_40dB', 'success_50dB']
    labels = ['-20dB', '-30dB', '-40dB', '-50dB']

    x = np.arange(len(thresholds))
    width = 0.8 / len(algorithms)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, algo in enumerate(algorithms):
        df_algo = df[df['method'] == algo]
        probs = [df_algo[thresh].mean() * 100.0 for thresh in thresholds]

        offset = (i - len(algorithms)/2) * width + width/2
        ax.bar(x + offset, probs, width, label=algo)

    ax.set_ylabel('Success Probability (%)')
    ax.set_title('Null Synthesis Success Profiles')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('experiments/SOTA_Success_Profiles.png')
    print("Saved SOTA Success Profiles plot.")

def plot_pareto(df):
    # Plot Gain Loss vs Null Depth for N=64
    df_64 = df[df['N'] == 64]
    algorithms = df_64['method'].unique()

    plt.figure(figsize=(10, 8))

    for algo in algorithms:
        df_algo = df_64[df_64['method'] == algo]
        plt.scatter(df_algo['null_depth_dB'], df_algo['gain_loss_dB'], label=algo, alpha=0.7, s=100)

    plt.axvline(-40, color='r', linestyle='--', label='Target Null (-40dB)')
    plt.axhline(3, color='k', linestyle='--', label='Max Acceptable Gain Loss (3dB)')

    # Fill the "Ideal Region" (Deep nulls, low gain loss)
    plt.fill_betweenx([0, 3], -100, -40, color='green', alpha=0.1, label='Ideal Operating Region')

    plt.xlim(0, -60) # Reverse x axis
    plt.ylim(-1, 15)

    plt.xlabel('Final Null Depth (dB)')
    plt.ylabel('Mainlobe Gain Loss (dB)')
    plt.title('Gain vs Null Depth Pareto Frontier (N=64)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('experiments/SOTA_Pareto_Frontier.png')
    print("Saved SOTA Pareto Frontier plot.")

if __name__ == '__main__':
    df = load_data()
    plot_performance_metrics(df)
    plot_success_profiles(df)
    plot_pareto(df)

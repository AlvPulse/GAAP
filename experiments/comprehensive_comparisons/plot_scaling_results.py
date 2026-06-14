import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    df_iter = pd.read_csv('experiments/comprehensive_comparisons/data/scaling_iterations.csv')
    df_trials = pd.read_csv('experiments/comprehensive_comparisons/data/scaling_trials.csv')
    return df_iter, df_trials

def plot_metrics_vs_scaling(df_trials):
    scales = sorted(df_trials['N'].unique())
    algorithms = df_trials['algorithm'].unique()

    metrics = {
        'final_PTNR': 'Peak-to-Null Ratio (dB)',
        'final_gain': 'Peak Gain (dB)',
        'final_null': 'Null Depth (dB)',
        'final_SLL': 'Side Lobe Level (dB)'
    }

    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs = axs.flatten()

    for i, (metric, label) in enumerate(metrics.items()):
        ax = axs[i]
        for algo in algorithms:
            df_algo = df_trials[df_trials['algorithm'] == algo]
            means = []
            stds = []
            for n in scales:
                data = df_algo[df_algo['N'] == n]
                means.append(np.mean(data[metric]))
                stds.append(np.std(data[metric]))

            ax.errorbar(scales, means, yerr=stds, label=algo, marker='o', capsize=5, linewidth=2)

        ax.set_xscale('log')
        ax.set_xticks(scales)
        ax.set_xticklabels([str(n) for n in scales])
        ax.set_xlabel('Array Size $N$')
        ax.set_ylabel(label)
        ax.set_title(f'Scaling Performance: {label}')
        ax.grid(True, which="both", ls="-", alpha=0.5)

        if i == 0: # Only put legend on first plot to save space
            ax.legend()

    plt.tight_layout()
    plt.savefig('experiments/comprehensive_comparisons/Scaling_Metrics.png')

def plot_time_to_null_thresholds(df_trials):
    scales = sorted(df_trials['N'].unique())
    algorithms = df_trials['algorithm'].unique()

    thresholds = ['TTT_30dB', 'TTT_40dB', 'TTT_50dB']
    labels = ['-30dB PTNR', '-40dB PTNR', '-50dB PTNR']

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    for i, (thresh, label) in enumerate(zip(thresholds, labels)):
        ax = axs[i]
        for algo in algorithms:
            df_algo = df_trials[df_trials['algorithm'] == algo]
            means = []
            stds = []
            for n in scales:
                data = df_algo[df_algo['N'] == n]

                # Estimate time = runtime_per_iter * TTT
                runtime_per_iter = data['runtime'].values / 50.0 # B_total
                ttt_runtime = runtime_per_iter * data[thresh].values

                means.append(np.mean(ttt_runtime))
                stds.append(np.std(ttt_runtime))

            ax.errorbar(scales, means, yerr=stds, label=algo, marker='o', capsize=4, linewidth=2)

        # Add deep learning zero-inference theoretical baseline annotation
        ax.axhline(y=0.01, color='r', linestyle='--', label='Theoretical Deep Learning Inference')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks(scales)
        ax.set_xticklabels([str(n) for n in scales])
        ax.set_xlabel('Array Size $N$')
        ax.set_ylabel(f'Time to reach {label} (s)')
        ax.set_title(f'Computational Time to {label}')
        ax.grid(True, which="both", ls="-", alpha=0.5)

    axs[0].legend(fontsize='small')
    plt.tight_layout()
    plt.savefig('experiments/comprehensive_comparisons/Scaling_Time_to_Targets.png')

def plot_convergence_history(df_iter):
    # Plot convergence for N=64
    df_64 = df_iter[df_iter['N'] == 64]
    algorithms = df_64['algorithm'].unique()

    metrics = {
        'PTNR': 'PTNR Convergence',
        'gain': 'Peak Gain Preservation',
        'null_depth': 'Null Synthesis Convergence'
    }

    fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    for i, (metric, title) in enumerate(metrics.items()):
        ax = axs[i]
        for algo in algorithms:
            df_algo = df_64[df_64['algorithm'] == algo]
            iterations = sorted(df_algo['iteration'].unique())

            means = []
            for it in iterations:
                subset = df_algo[df_algo['iteration'] == it]
                means.append(np.mean(subset[metric]))

            ax.plot(iterations, means, label=algo, linewidth=2)

        ax.set_ylabel(metric.replace('_', ' ').title() + ' (dB)')
        ax.set_title(f'{title} (N=64)')
        ax.grid(True)
        if i == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    axs[-1].set_xlabel('Iterations')
    plt.tight_layout()
    plt.savefig('experiments/comprehensive_comparisons/Convergence_History.png')

if __name__ == '__main__':
    df_iter, df_trials = load_data()
    plot_metrics_vs_scaling(df_trials)
    plot_time_to_null_thresholds(df_trials)
    plot_convergence_history(df_iter)
    print("Generated comprehensive comparison plots successfully.")

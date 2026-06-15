import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    df_iter = pd.read_csv('experiments/data/sota_scaling_iterations.csv')
    df_trials = pd.read_csv('experiments/data/sota_scaling_trials.csv')
    return df_iter, df_trials

def plot_time_to_target(df_trials):
    plt.figure(figsize=(10, 6))

    algorithms = df_trials['algorithm'].unique()
    scales = sorted(df_trials['N'].unique())

    for algo in algorithms:
        df_algo = df_trials[df_trials['algorithm'] == algo]

        means = []
        stds = []
        for n in scales:
            data = df_algo[df_algo['N'] == n]

            # Estimate runtime to hit target PTNR
            # Runtime per iteration * TTT
            runtime_per_iter = data['runtime'].values / 100.0
            ttt_runtime = runtime_per_iter * data['TTT_30dB'].values

            means.append(np.mean(ttt_runtime))
            stds.append(np.std(ttt_runtime))

        plt.errorbar(scales, means, yerr=stds, label=algo, marker='o', capsize=5, linewidth=2)

    # Deep Learning (Zero-Inference) annotation line
    plt.axhline(y=0.01, color='r', linestyle='--', label='Deep Learning Inference (Theoretical)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(scales, [str(n) for n in scales])
    plt.xlabel('Array Size $N$')
    plt.ylabel('Wall-clock Time to 30dB PTNR (s)')
    plt.title('Pipeline Computational Scaling vs. Monolithic Methods')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig('experiments/SOTA_Time_Scaling.png')

def plot_gain_null_separation(df_iter):
    # Filter to just N=64 for iteration trajectories
    df_64 = df_iter[df_iter['N'] == 64]

    fig, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    algorithms = ['Strain-Guided Pipeline', 'AP-Only', 'PGD-Baseline (Monolithic)']
    colors = ['tab:blue', 'tab:orange', 'tab:red']

    for i, algo in enumerate(algorithms):
        df_algo = df_64[df_64['algorithm'] == algo]
        iterations = sorted(df_algo['iteration'].unique())

        mean_gain, mean_null = [], []
        for it in iterations:
            subset = df_algo[df_algo['iteration'] == it]
            mean_gain.append(np.mean(subset['gain']))
            mean_null.append(np.mean(subset['null_depth']))

        axs[0].plot(iterations, mean_gain, label=algo, color=colors[i], linewidth=2)
        axs[1].plot(iterations, mean_null, label=algo, color=colors[i], linewidth=2)

    # Theoretical max gain for N=64 is 20*log10(64) ~ 36dB
    target_gain = 20 * np.log10(64)
    axs[0].axhline(target_gain, color='k', linestyle='--', alpha=0.5, label='Theoretical Max Gain')
    axs[0].set_ylabel('Peak Gain (dB)')
    axs[0].set_title('Gain Preservation Dynamics')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].axhline(-40, color='k', linestyle='--', alpha=0.5, label='Target Null Depth')
    axs[1].set_ylabel('Null Depth (dB)')
    axs[1].set_xlabel('Evaluations')
    axs[1].set_title('Null Synthesis Dynamics')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig('experiments/SOTA_Gain_Null_Separation.png')

def plot_pareto_trajectory(df_iter):
    df_64 = df_iter[df_iter['N'] == 64]

    plt.figure(figsize=(10, 8))

    algorithms = ['Strain-Guided Pipeline', 'AP-Only', 'LR-Only', 'PGD-Baseline (Monolithic)']

    for algo in algorithms:
        df_algo = df_64[df_64['algorithm'] == algo]

        # We plot the trajectory of (Null, Gain) over iterations to show if gain drops as nulls deepen
        # Group by iteration to get mean trajectory
        mean_gains = df_algo.groupby('iteration')['gain'].mean()
        mean_nulls = df_algo.groupby('iteration')['null_depth'].mean()

        plt.plot(mean_nulls, mean_gains, label=algo, marker='o', markersize=3, alpha=0.7)
        # Mark start and end points
        plt.scatter(mean_nulls.iloc[0], mean_gains.iloc[0], marker='s', s=100, edgecolors='black')
        plt.scatter(mean_nulls.iloc[-1], mean_gains.iloc[-1], marker='*', s=200, edgecolors='black')

    target_gain = 20 * np.log10(64)
    plt.axhline(target_gain, color='k', linestyle='--', alpha=0.5, label='Theoretical Max Gain')
    plt.axvline(-40, color='r', linestyle='--', alpha=0.5, label='Target Null Depth')

    plt.xlabel('Null Depth (dB) [Lower is better]')
    plt.ylabel('Peak Gain (dB) [Higher is better]')
    plt.title('Gain vs. Null Depth Pareto Trajectories')
    # We want top left corner (high gain, low null)
    plt.xlim(0, -60) # Reverse X axis to show deeper nulls moving right-to-left

    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/SOTA_Pareto_Trajectory.png')

if __name__ == '__main__':
    df_iter, df_trials = load_data()
    plot_time_to_target(df_trials)
    plot_gain_null_separation(df_iter)
    plot_pareto_trajectory(df_iter)
    print("Generated SOTA Comparison Figures successfully.")

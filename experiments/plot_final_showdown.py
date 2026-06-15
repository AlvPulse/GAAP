import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    df_iter = pd.read_csv('experiments/data/showdown_iterations.csv')
    df_trials = pd.read_csv('experiments/data/showdown_trials.csv')
    return df_iter, df_trials

def plot_showdown(df_iter, df_trials):
    algorithms = [
        'Fixed (Baseline)',
        'SA (Hard Reset)',
        'Hill Climbing',
        'Trust-Region HC',
        'Basin Hopping'
    ]

    fig, axs = plt.subplots(3, 1, figsize=(10, 16))

    # 1. PTNR Convergence over Iterations
    iterations = sorted(df_iter['iteration'].unique())
    for algo in algorithms:
        df_algo = df_iter[df_iter['algorithm'] == algo]
        means = df_algo.groupby('iteration')['PTNR'].mean()
        stds = df_algo.groupby('iteration')['PTNR'].std()

        axs[0].plot(iterations, means, label=algo, linewidth=2)
        axs[0].fill_between(iterations, means - 0.5*stds, means + 0.5*stds, alpha=0.1)

    axs[0].set_ylabel('PTNR (dB)')
    axs[0].set_xlabel('Iterations')
    axs[0].set_title('PTNR Convergence: Final Controller Showdown')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Boxplot of Final PTNR
    final_data = []
    for algo in algorithms:
        final_ptnrs = df_trials[df_trials['algorithm'] == algo]['final_PTNR'].values
        final_data.append(final_ptnrs)

    axs[1].boxplot(final_data, tick_labels=algorithms)
    axs[1].set_ylabel('Final PTNR (dB)')
    axs[1].set_title('Statistical Distribution of Final PTNR')
    axs[1].grid(True, axis='y')

    # 3. Bar Chart for Time-To-Targets
    x = np.arange(len(algorithms))
    width = 0.25

    ttt_30_means = [df_trials[df_trials['algorithm'] == algo]['TTT_30dB'].mean() for algo in algorithms]
    ttt_40_means = [df_trials[df_trials['algorithm'] == algo]['TTT_40dB'].mean() for algo in algorithms]
    ttt_50_means = [df_trials[df_trials['algorithm'] == algo]['TTT_50dB'].mean() for algo in algorithms]

    rects1 = axs[2].bar(x - width, ttt_30_means, width, label='-30dB Target', capsize=5)
    rects2 = axs[2].bar(x, ttt_40_means, width, label='-40dB Target', capsize=5)
    rects3 = axs[2].bar(x + width, ttt_50_means, width, label='-50dB Target', capsize=5)

    axs[2].set_ylabel('Iterations to Target')
    axs[2].set_title('Time-to-Target Performance')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(algorithms, rotation=45, ha='right')
    axs[2].legend()
    axs[2].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('experiments/Final_Showdown.png')
    print("Saved Final Showdown plots to experiments/Final_Showdown.png")


if __name__ == '__main__':
    df_iter, df_trials = load_data()
    plot_showdown(df_iter, df_trials)
    plot_detailed_metrics(df_iter)

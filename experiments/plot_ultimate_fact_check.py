import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    df_iter = pd.read_csv('experiments/ultimate_fact_check_iterations.csv')
    df_trials = pd.read_csv('experiments/ultimate_fact_check_trials.csv')
    return df_iter, df_trials

def plot_fact_check(df_iter, df_trials):
    algorithms = [
        'Fixed (Baseline)',
        'One-Shot SA + Freeze',
        'Cont SA (Hard Reset)',
        'Cont SA (No Reset)',
        'Cont SA (Smart Reset)',
        'CROA'
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
    axs[0].set_title('PTNR Convergence vs Reset Mechanism')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Boxplot of Final PTNR
    final_data = []
    for algo in algorithms:
        final_ptnrs = df_trials[df_trials['algorithm'] == algo]['final_PTNR'].values
        final_data.append(final_ptnrs)

    axs[1].boxplot(final_data)
    axs[1].set_xticklabels(algorithms, rotation=45, ha='right')
    axs[1].set_ylabel('Final PTNR (dB)')
    axs[1].set_title('Statistical Distribution of Final PTNR')
    axs[1].grid(True, axis='y')

    # 3. Bar Chart for Time-To-Targets
    x = np.arange(len(algorithms))
    width = 0.35

    ttt_30_means = [df_trials[df_trials['algorithm'] == algo]['TTT_30dB'].mean() for algo in algorithms]
    ttt_30_stds = [df_trials[df_trials['algorithm'] == algo]['TTT_30dB'].std() for algo in algorithms]

    ttt_40_means = [df_trials[df_trials['algorithm'] == algo]['TTT_40dB'].mean() for algo in algorithms]
    ttt_40_stds = [df_trials[df_trials['algorithm'] == algo]['TTT_40dB'].std() for algo in algorithms]

    rects1 = axs[2].bar(x - width/2, ttt_30_means, width, yerr=ttt_30_stds, label='-30dB Target', capsize=5)
    rects2 = axs[2].bar(x + width/2, ttt_40_means, width, yerr=ttt_40_stds, label='-40dB Target', capsize=5)

    axs[2].set_ylabel('Iterations to Target')
    axs[2].set_title('Time-to-Target Performance')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(algorithms, rotation=45, ha='right')
    axs[2].legend()
    axs[2].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('experiments/Ultimate_Fact_Check.png')
    print("Saved Ultimate Fact Check plots to experiments/Ultimate_Fact_Check.png")

if __name__ == '__main__':
    df_iter, df_trials = load_data()
    plot_fact_check(df_iter, df_trials)

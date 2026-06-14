import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def load_data():
    df_iter = pd.read_csv('experiments/data/final_benchmark_iterations.csv')
    df_trials = pd.read_csv('experiments/data/final_benchmark_trials.csv')
    return df_iter, df_trials

def plot_fig2_convergence(df_iter):
    # Figure 2: PTNR convergence
    plt.figure(figsize=(10, 6))

    algorithms = df_iter['algorithm'].unique()
    iterations = df_iter['iteration'].unique()

    for algo in algorithms:
        data = df_iter[df_iter['algorithm'] == algo].pivot(index='seed', columns='iteration', values='PTNR').values
        mean_val = np.mean(data, axis=0)
        q25 = np.percentile(data, 25, axis=0)
        q75 = np.percentile(data, 75, axis=0)

        plt.plot(iterations, mean_val, label=algo, linewidth=2)
        plt.fill_between(iterations, q25, q75, alpha=0.1)

    plt.xlabel('Evaluations (AP+LR executions)')
    plt.ylabel('Peak-to-Null Ratio (PTNR) [dB]')
    plt.title('Figure 2: Pipeline PTNR Convergence')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/Fig2_PTNR_Convergence.png')

def plot_fig3_trajectories(df_iter):
    # Figure 3: Offset trajectories for a single seed to show behavior
    plt.figure(figsize=(10, 6))

    seed_to_plot = df_iter['seed'].unique()[0]
    df_seed = df_iter[df_iter['seed'] == seed_to_plot]

    target_algos = ['SA', 'HillClimbing', 'CROA']

    for algo in target_algos:
        df_algo = df_seed[df_seed['algorithm'] == algo]
        plt.plot(df_algo['iteration'], df_algo['offset'], label=algo, marker='.', alpha=0.7)

    plt.ylabel('Offset $\\\\delta(k)$ (rad)')
    plt.xlabel('Iteration $k$')
    plt.title('Figure 3: Offset Trajectories')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/Fig3_Offset_Trajectories.png')

def plot_fig4_lr_step(df_iter):
    # Figure 4: LR step size evolution
    plt.figure(figsize=(10, 6))

    seed_to_plot = df_iter['seed'].unique()[0]
    df_seed = df_iter[df_iter['seed'] == seed_to_plot]

    target_algos = ['SA', 'CROA']

    for algo in target_algos:
        df_algo = df_seed[df_seed['algorithm'] == algo]
        plt.plot(df_algo['iteration'], df_algo['step_size'], label=algo, alpha=0.8)

    plt.ylabel('LR Step Size $\\\\alpha(k)$')
    plt.xlabel('Iteration $k$')
    plt.title('Figure 4: Inner-Loop Step Size Evolution (Soft Carryover)')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/Fig4_LR_Step_Size.png')

def plot_fig5_carryover(df_iter):
    # Figure 5: Carryover ratio after offset updates (PTNR_after / PTNR_before in linear power terms)
    # PTNR is in dB. PTNR_after - PTNR_before is the ratio in dB.
    plt.figure(figsize=(10, 6))

    target_algos = ['SA', 'HillClimbing', 'CROA']

    for algo in target_algos:
        df_algo = df_iter[(df_iter['algorithm'] == algo) & (df_iter['offset_jump'] > 1e-6)]

        # Calculate ratio in linear scale: 10^((PTNR_after - PTNR_before)/10)
        # Ratio of 1.0 means perfect momentum preservation. < 1.0 means destructive jump.
        db_diff = df_algo['post_update_PTNR'] - df_algo['pre_update_PTNR']
        linear_carryover = 10**(db_diff / 10.0)

        plt.hist(linear_carryover, bins=30, alpha=0.5, label=algo, density=True)

    plt.axvline(1.0, color='r', linestyle='--', label='Perfect Carryover')
    plt.xlabel('Carryover Ratio (PTNR_after / PTNR_before)')
    plt.ylabel('Density')
    plt.title('Figure 5: Carryover Ratio After Offset Jumps')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 2) # Focus on the drop
    plt.savefig('experiments/Fig5_Carryover_Ratio.png')

def plot_fig6_jump_sizes(df_iter):
    # Figure 6: Offset jump size distribution
    plt.figure(figsize=(10, 6))

    target_algos = ['SA', 'HillClimbing', 'CROA']

    for algo in target_algos:
        df_algo = df_iter[(df_iter['algorithm'] == algo) & (df_iter['offset_jump'] > 1e-6)]

        plt.hist(df_algo['offset_jump'], bins=30, alpha=0.5, label=algo, density=True)

    plt.xlabel('Offset Jump Magnitude $|\\\\Delta\\\\delta|$ (rad)')
    plt.ylabel('Density')
    plt.title('Figure 6: Offset Jump Size Distribution')
    plt.legend()
    plt.grid(True)
    plt.savefig('experiments/Fig6_Jump_Sizes.png')

if __name__ == '__main__':
    df_iter, df_trials = load_data()
    plot_fig2_convergence(df_iter)
    plot_fig3_trajectories(df_iter)
    plot_fig4_lr_step(df_iter)
    plot_fig5_carryover(df_iter)
    plot_fig6_jump_sizes(df_iter)
    print("Generated Figures 2-6 successfully.")

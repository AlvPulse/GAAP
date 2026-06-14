import numpy as np
import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.controllers.final_benchmark import CROA
from experiments.run_final_paper_benchmarks import run_single_trial

def run_ablation():
    os.makedirs('experiments/data', exist_ok=True)

    modes = [
        'hard_reset',     # A
        'soft_carryover', # B (Default proposed method)
        'no_reheat',      # C
        'always_reheat'   # D
    ]

    num_seeds = 10
    N = 64
    B_total = 100

    all_iter_df = []
    all_trial_data = []

    for mode in modes:
        print(f"Running CROA Ablation: {mode}...")
        for seed in range(num_seeds):
            df_iter, trial_dict = run_single_trial(seed, N, "CROA", CROA, B_total, ablation_mode=mode)
            all_iter_df.append(df_iter)
            all_trial_data.append(trial_dict)

    final_iter_df = pd.concat(all_iter_df, ignore_index=True)
    final_trial_df = pd.DataFrame(all_trial_data)

    final_iter_df.to_csv('experiments/data/croa_ablation_iterations.csv', index=False)
    final_trial_df.to_csv('experiments/data/croa_ablation_trials.csv', index=False)
    print("Saved CROA Ablation CSVs to experiments/data/")

if __name__ == '__main__':
    run_ablation()

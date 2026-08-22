import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.controllers.core import step_ap_lr, apply_phase_jump
from beamformer.controllers.metrics import get_metrics
from beamformer.controllers.sa_families import BaseController, FixedOffset, SA, CROA
from experiments.fact_check_inner_resets import SA_ResetVariant

class OneShotSA(BaseController):
    def __init__(self, exploration_iters=10, T0=10.0, alpha=0.8, sigma=np.pi/4, **kwargs):
        super().__init__(**kwargs)
        self.explore_iters = exploration_iters
        self.T = T0
        self.alpha = alpha
        self.sigma = sigma
        self.k = 0
        self.best_delta = 0.0
        self.best_cost = float('inf')

    def propose(self, current_delta, **kwargs):
        self.k += 1
        if self.k <= self.explore_iters:
            return (current_delta + np.random.normal(0, self.sigma)) % (2*np.pi)
        else:
            return self.best_delta # Lock forever

    def accept(self, current_cost, new_cost):
        if self.k <= self.explore_iters:
            # Standard SA acceptance during exploration
            if new_cost < current_cost:
                return True
            if self.T > 1e-6:
                prob = np.exp(-(new_cost - current_cost) / self.T)
                return np.random.rand() < prob
            return False
        else:
            # After exploration, never move again (force FixedOffset behavior)
            return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)
        if self.k <= self.explore_iters:
            if accepted:
                self.best_delta = kwargs.get('cand_delta', 0.0)
                self.best_cost = kwargs.get('new_cost', float('inf'))
            self.T *= self.alpha

def run_ultimate_trial(seed, N, algo_name, B_total=150):
    np.random.seed(seed)

    element = SyntheticVaractor(beta=1.5, folding=True)

    # Dynamically generate random target and nulls to avoid bias
    target_angle = np.random.uniform(-0.4, 0.4)
    nulls = []
    while len(nulls) < 3:
        n_ang = np.random.uniform(-0.9, 0.9)
        if np.abs(n_ang - target_angle) > 0.15: # Ensure nulls aren't inside main beam
            nulls.append(n_ang)

    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    current_delta = 0.0
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n], _= element.project(c_n[n] * np.exp(1j * current_delta))

    if algo_name == 'Fixed (Baseline)':
        controller = FixedOffset()
    elif algo_name == 'One-Shot SA + Freeze':
        controller = OneShotSA(exploration_iters=10)
    elif algo_name == 'Cont SA (Hard Reset)':
        controller = SA_ResetVariant(mode='hard')
    elif algo_name == 'Cont SA (No Reset)':
        controller = SA_ResetVariant(mode='none')
    elif algo_name == 'Cont SA (Smart Reset)':
        controller = SA_ResetVariant(mode='smart', jump_threshold=np.pi/4)
    elif algo_name == 'CROA':
        controller = CROA(ablation_mode='soft_carryover')

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    c_n, V_n, res, _null, _gain, _ptnr, _ = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=lr_step)
    null_depth, gain, current_ptnr = get_metrics(c_n, task, element)
    current_cost = -current_ptnr

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta
    history_res = [res]

    iter_data = []
    offset_changes = 0

    for k in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)
        delta_diff = min(np.abs(cand_delta - current_delta), 2*np.pi - np.abs(cand_delta - current_delta))

        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)
        cand_c, cand_V, cand_res, _ = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=lr_step)

        cand_null, cand_gain, cand_ptnr = get_metrics(cand_c, task, element)
        cand_cost = -cand_ptnr

        accepted = controller.accept(current_cost, cand_cost)
        old_delta = current_delta

        if accepted:
            current_cost = cand_cost
            best_c, best_V, best_delta = cand_c, cand_V, cand_delta
            current_ptnr = cand_ptnr
            gain = cand_gain
            null_depth = cand_null

            if delta_diff > 1e-4:
                offset_changes += 1

        controller.update_state(
            accepted,
            history_r=history_res,
            lr_step=lr_step,
            lr_min=lr_min,
            new_cost=cand_cost,
            current_delta=old_delta,
            cand_delta=cand_delta,
            new_res=cand_res,
            r_target=1e-4,
            r_0=history_res[0] if history_res else 1.0
        )

        if accepted:
            current_delta = cand_delta

        if hasattr(controller, 'lr_step'):
            lr_step = controller.lr_step

        iter_data.append({
            'algorithm': algo_name,
            'seed': seed,
            'iteration': k,
            'PTNR': current_ptnr,
            'gain': gain,
            'null_depth': null_depth,
            'lr_step': lr_step,
            'offset_changes': offset_changes
        })

        history_res.append(cand_res if accepted else history_res[-1])

        if k > 1 and cand_res < history_res[-2]:
            lr_step = min(lr_step * 1.1, lr_max)
        else:
            lr_step = max(lr_step * 0.5, lr_min)

        if hasattr(controller, 'lr_step'):
            controller.lr_step = lr_step

    df_iter = pd.DataFrame(iter_data)

    ptnrs = df_iter['PTNR'].values
    def time_to(target_db):
        idx = np.where(ptnrs >= target_db)[0]
        return idx[0] + 1 if len(idx) > 0 else B_total

    trial_data = {
        'algorithm': algo_name,
        'seed': seed,
        'final_PTNR': ptnrs[-1],
        'final_gain': df_iter['gain'].iloc[-1],
        'final_null': df_iter['null_depth'].iloc[-1],
        'offset_changes': offset_changes,
        'TTT_20dB': time_to(20),
        'TTT_30dB': time_to(30),
        'TTT_40dB': time_to(40),
        'TTT_50dB': time_to(50)
    }

    return df_iter, trial_data

def run_ultimate_fact_check():
    num_seeds = 5
    N = 64
    B_total = 150

    algorithms = [
        'Fixed (Baseline)',
        'One-Shot SA + Freeze',
        'Cont SA (Hard Reset)',
        'Cont SA (No Reset)',
        'Cont SA (Smart Reset)',
        'CROA'
    ]

    all_iter_df = []
    all_trial_data = []

    for algo in algorithms:
        print(f"Running Ultimate Fact Check: {algo}...")
        for seed in range(num_seeds):
            df_iter, trial_dict = run_ultimate_trial(seed, N, algo, B_total)
            all_iter_df.append(df_iter)
            all_trial_data.append(trial_dict)

    final_iter_df = pd.concat(all_iter_df, ignore_index=True)
    final_trial_df = pd.DataFrame(all_trial_data)

    final_iter_df.to_csv('experiments/ultimate_fact_check_iterations.csv', index=False)
    final_trial_df.to_csv('experiments/ultimate_fact_check_trials.csv', index=False)
    print("Fact check data saved.")

if __name__ == '__main__':
    run_ultimate_fact_check()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.controllers.core import step_ap_lr, apply_phase_jump
from beamformer.controllers.metrics import get_metrics, get_sll
from beamformer.controllers.sa_families import BaseController, FixedOffset, SA, HillClimbing, BasinHopping
from beamformer.controllers.trust_region import TR_HC

def run_showdown_trial(seed, N, algo_name, controller_class, controller_kwargs, B_total=150):
    np.random.seed(seed)

    element = SyntheticVaractor(beta=1.5, folding=True)

    # Unbiased Randomized Topology
    target_angle = np.random.uniform(-0.4, 0.4)
    nulls = []
    while len(nulls) < 3:
        n_ang = np.random.uniform(-0.9, 0.9)
        if np.abs(n_ang - target_angle) > 0.15:
            nulls.append(n_ang)

    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    current_delta = 0.0
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n], _= element.project(c_n[n] * np.exp(1j * current_delta))

    controller = controller_class(**controller_kwargs)

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    c_n, V_n, res, _ = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=lr_step)
    null_depth, gain, current_ptnr = get_metrics(c_n, task, element)
    sll = get_sll(c_n, task)
    current_cost = -current_ptnr

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta
    history_res = [res]

    iter_data = []

    start_time = time.time()

    for k in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)
        delta_diff = min(np.abs(cand_delta - current_delta), 2*np.pi - np.abs(cand_delta - current_delta))

        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)
        cand_c, cand_V, cand_res, _ = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=lr_step)

        cand_null, cand_gain, cand_ptnr = get_metrics(cand_c, task, element)
        cand_sll = get_sll(cand_c, task)
        cand_cost = -cand_ptnr

        accepted = controller.accept(current_cost, cand_cost)
        old_delta = current_delta

        if accepted:
            current_cost = cand_cost
            best_c, best_V, best_delta = cand_c, cand_V, cand_delta
            current_ptnr = cand_ptnr
            gain = cand_gain
            null_depth = cand_null
            sll = cand_sll

            # HARD RESET logic (Since it was proven to be the best for exploring the sharp topology)
            if delta_diff > 1e-3:
                lr_step = lr_initial

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

        iter_data.append({
            'algorithm': algo_name,
            'seed': seed,
            'iteration': k,
            'PTNR': current_ptnr,
            'gain': gain,
            'null_depth': null_depth,
            'SLL': sll,
            'offset_jump': delta_diff if accepted else 0.0,
            'accept_ratio': controller.accepted_moves / max(1, controller.total_proposals)
        })

        history_res.append(cand_res if accepted else history_res[-1])

        # Standard LR Adaptation
        if k > 1 and cand_res < history_res[-2]:
            lr_step = min(lr_step * 1.1, lr_max)
        else:
            lr_step = max(lr_step * 0.5, lr_min)

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
        'TTT_30dB': time_to(30),
        'TTT_40dB': time_to(40),
        'TTT_50dB': time_to(50)
    }

    return df_iter, trial_data

def run_showdown():
    os.makedirs('experiments/data', exist_ok=True)

    competitors = {
        'Fixed (Baseline)': (FixedOffset, {}),
        'SA (Hard Reset)': (SA, {'T0': 10.0, 'alpha': 0.9, 'sigma': np.pi/12}),
        'Hill Climbing': (HillClimbing, {'step_size': np.pi/12}),
        'Trust-Region HC': (TR_HC, {'delta_init': np.pi/12, 'smart_reset': False}),
        'Basin Hopping': (BasinHopping, {})
    }

    num_seeds = 10
    N = 64
    B_total = 150

    all_iter_df = []
    all_trial_data = []

    for algo_name, (cls, kwargs) in competitors.items():
        print(f"Running Showdown: {algo_name}...")
        for seed in range(num_seeds):
            df_iter, trial_dict = run_showdown_trial(seed, N, algo_name, cls, kwargs, B_total)
            all_iter_df.append(df_iter)
            all_trial_data.append(trial_dict)

    final_iter_df = pd.concat(all_iter_df, ignore_index=True)
    final_trial_df = pd.DataFrame(all_trial_data)

    final_iter_df.to_csv('experiments/data/showdown_iterations.csv', index=False)
    final_trial_df.to_csv('experiments/data/showdown_trials.csv', index=False)
    print("Showdown data saved.")

if __name__ == '__main__':
    run_showdown()

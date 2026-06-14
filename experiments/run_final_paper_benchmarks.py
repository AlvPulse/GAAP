import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.controllers.core import step_ap_lr, apply_phase_jump
from beamformer.controllers.metrics import get_metrics
from beamformer.controllers.final_benchmark import FixedOffset, RandomRestart, HillClimbing, SA, CROA

def run_single_trial(seed, N, controller_name, controller_class, B_total=100, ablation_mode=None):
    np.random.seed(seed)

    element = SyntheticVaractor(beta=1.5, folding=True)
    target_angle = 0.3
    nulls = [-0.5, 0.1, 0.7]
    task = Task(type='nulled', target_angles=[target_angle], null_angles=nulls, sll_ceiling=0.1)

    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    current_delta = 0.0
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n] = element.project(c_n[n] * np.exp(1j * current_delta))

    if ablation_mode:
        controller = controller_class(ablation_mode=ablation_mode)
    else:
        controller = controller_class()

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    c_n, V_n, res, acc_corr = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=lr_step)
    null_depth, gain, current_ptnr = get_metrics(c_n, task, element)
    current_cost = -current_ptnr

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta
    history_res = [res]

    iter_data = []

    start_time = time.time()

    for k in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)
        delta_diff = min(np.abs(cand_delta - current_delta), 2*np.pi - np.abs(cand_delta - current_delta))

        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)
        cand_c, cand_V, cand_res, acc_corr = step_ap_lr(cand_c, cand_delta, cand_V, task, element, step_size=lr_step)

        cand_null, cand_gain, cand_ptnr = get_metrics(cand_c, task, element)
        cand_cost = -cand_ptnr

        accepted = controller.accept(current_cost, cand_cost)
        old_delta = current_delta

        if accepted:
            current_cost = cand_cost
            best_c, best_V, best_delta = cand_c, cand_V, cand_delta
            null_depth, gain, current_ptnr = cand_null, cand_gain, cand_ptnr

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

        # Extract preserved step size
        if hasattr(controller, 'lr_step'):
            lr_step = controller.lr_step

        # Logging Iteration Data
        iter_data.append({
            'algorithm': controller_name if not ablation_mode else f"{controller_name}_{ablation_mode}",
            'seed': seed,
            'iteration': k,
            'offset': current_delta,
            'offset_jump': delta_diff if accepted else 0.0,
            'step_size': lr_step,
            'gain': gain,
            'worst_null': null_depth,
            'PTNR': current_ptnr,
            'residual': cand_res if accepted else history_res[-1],
            'accepted_corrections': acc_corr if accepted else 0,
            'pre_update_PTNR': -current_cost if k==1 else iter_data[-1]['PTNR'] if iter_data else current_ptnr,
            'post_update_PTNR': current_ptnr
        })

        history_res.append(cand_res if accepted else history_res[-1])

        # Base LR Adaptation (Unless overridden by controller soft carryover next turn)
        if k > 1 and cand_res < history_res[-2]:
            lr_step = min(lr_step * 1.1, lr_max)
        else:
            lr_step = max(lr_step * 0.5, lr_min)

        if hasattr(controller, 'lr_step'):
            controller.lr_step = lr_step

    runtime = time.time() - start_time
    df_iter = pd.DataFrame(iter_data)

    # Calculate Trial Summaries
    ptnrs = df_iter['PTNR'].values
    auc = np.sum(ptnrs)
    offset_updates = np.sum(df_iter['offset_jump'] > 1e-6)
    improvement = ptnrs[-1] - ptnrs[0]
    meta_efficiency = improvement / (offset_updates + 1e-6)

    # Carryover efficiency = Average of (post_update_PTNR / pre_update_PTNR) for iterations where jump > 0
    jump_mask = df_iter['offset_jump'] > 1e-6
    if np.any(jump_mask):
        # Shift pre_update_ptnr to be strictly positive relative to a baseline if possible,
        # or use linear scale. Since PTNR is in dB, PTNR_after - PTNR_before is the ratio in linear scale.
        # So "carryover" in dB differences should be close to 0 (no loss).
        # We will log mean PTNR drop upon jumps.
        mean_jump_ptnr_diff = np.mean(df_iter.loc[jump_mask, 'post_update_PTNR'] - df_iter.loc[jump_mask, 'pre_update_PTNR'])
    else:
        mean_jump_ptnr_diff = 0.0

    # Time to Targets
    def time_to(target_db):
        idx = np.where(ptnrs >= target_db)[0]
        return idx[0] + 1 if len(idx) > 0 else B_total

    trial_data = {
        'algorithm': controller_name if not ablation_mode else f"{controller_name}_{ablation_mode}",
        'seed': seed,
        'final_PTNR': ptnrs[-1],
        'final_worst_null': df_iter['worst_null'].iloc[-1],
        'AUC': auc,
        'runtime': runtime,
        'offset_updates': offset_updates,
        'meta_efficiency': meta_efficiency,
        'mean_jump_ptnr_diff_dB': mean_jump_ptnr_diff,
        'TTT_30dB': time_to(30),
        'TTT_40dB': time_to(40),
        'TTT_50dB': time_to(50),
        'TTT_60dB': time_to(60)
    }

    return df_iter, trial_data

def run_benchmarks():
    os.makedirs('experiments/data', exist_ok=True)

    controllers = {
        'Fixed': FixedOffset,
        'Random': RandomRestart,
        'HillClimbing': HillClimbing,
        'SA': SA,
        'CROA': CROA
    }

    num_seeds = 10
    N = 64
    B_total = 100

    all_iter_df = []
    all_trial_data = []

    for name, cls in controllers.items():
        print(f"Benchmarking {name}...")
        for seed in range(num_seeds):
            df_iter, trial_dict = run_single_trial(seed, N, name, cls, B_total)
            all_iter_df.append(df_iter)
            all_trial_data.append(trial_dict)

    final_iter_df = pd.concat(all_iter_df, ignore_index=True)
    final_trial_df = pd.DataFrame(all_trial_data)

    final_iter_df.to_csv('experiments/data/final_benchmark_iterations.csv', index=False)
    final_trial_df.to_csv('experiments/data/final_benchmark_trials.csv', index=False)
    print("Saved final benchmark CSVs to experiments/data/")

if __name__ == '__main__':
    run_benchmarks()

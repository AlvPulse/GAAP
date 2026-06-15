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
from beamformer.controllers.sa_families import BaseController, FixedOffset

class SA_ResetVariant(BaseController):
    def __init__(self, mode='hard', jump_threshold=np.pi/4, T0=10.0, alpha=0.9, sigma=np.pi/12, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode # 'hard', 'none', 'smart'
        self.jump_threshold = jump_threshold

        self.T = T0
        self.alpha = alpha
        self.sigma = sigma

        self.lr_step = kwargs.get('lr_initial', 0.05)
        self.lr_initial = kwargs.get('lr_initial', 0.05)

    def propose(self, current_delta, **kwargs):
        return (current_delta + np.random.normal(0, self.sigma)) % (2*np.pi)

    def accept(self, current_cost, new_cost):
        if new_cost < current_cost:
            return True
        if self.T > 1e-6:
            prob = np.exp(-(new_cost - current_cost) / self.T)
            return np.random.rand() < prob
        return False

    def update_state(self, accepted, **kwargs):
        super().update_state(accepted, **kwargs)

        current_lr = kwargs.get('lr_step', self.lr_step)

        if accepted:
            current_delta = kwargs.get('current_delta', 0.0)
            cand_delta = kwargs.get('cand_delta', 0.0)
            diff = np.abs(cand_delta - current_delta)
            circular_diff = min(diff, 2*np.pi - diff)

            if self.mode == 'hard':
                self.lr_step = self.lr_initial # Always reset
            elif self.mode == 'none':
                self.lr_step = current_lr # Never reset
            elif self.mode == 'smart':
                if circular_diff > self.jump_threshold:
                    self.lr_step = self.lr_initial # Reset on large jump
                else:
                    self.lr_step = current_lr # Preserve on small jump
        else:
            self.lr_step = current_lr

        self.T *= self.alpha

def run_fact_check_trial(seed, N, algo_name, B_total=150):
    np.random.seed(seed)

    element = SyntheticVaractor(beta=1.5, folding=True)
    task = Task(type='nulled', target_angles=[0.3], null_angles=[-0.5, 0.1, 0.7], sll_ceiling=0.1)

    c_n = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)

    current_delta = 0.0
    V_n = np.zeros(N)
    for n in range(N):
        V_n[n], c_n[n], _= element.project(c_n[n] * np.exp(1j * current_delta))

    if algo_name == 'Fixed Offset':
        controller = FixedOffset()
    elif algo_name == 'SA (Hard Reset)':
        controller = SA_ResetVariant(mode='hard')
    elif algo_name == 'SA (No Reset)':
        controller = SA_ResetVariant(mode='none')
    elif algo_name == 'SA (Smart Reset)':
        controller = SA_ResetVariant(mode='smart')

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    c_n, V_n, res, _ = step_ap_lr(c_n, current_delta, V_n, task, element, step_size=lr_step)
    null_depth, gain, current_ptnr = get_metrics(c_n, task, element)
    current_cost = -current_ptnr

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta
    history_res = [res]
    iter_data = []

    for k in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)

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
            'lr_step': lr_step,
            'offset': current_delta
        })

        history_res.append(cand_res if accepted else history_res[-1])

        # Base LR Adaptation (Unless overridden by controller next turn)
        if k > 1 and cand_res < history_res[-2]:
            lr_step = min(lr_step * 1.1, lr_max)
        else:
            lr_step = max(lr_step * 0.5, lr_min)

        if hasattr(controller, 'lr_step'):
            controller.lr_step = lr_step

    return pd.DataFrame(iter_data)

def plot_fact_check(df_iter):
    algorithms = ['Fixed Offset', 'SA (Hard Reset)', 'SA (No Reset)', 'SA (Smart Reset)']
    iterations = sorted(df_iter['iteration'].unique())

    fig, axs = plt.subplots(3, 1, figsize=(10, 14))

    # 1. PTNR Convergence
    for algo in algorithms:
        df_algo = df_iter[df_iter['algorithm'] == algo]
        means = df_algo.groupby('iteration')['PTNR'].mean()
        stds = df_algo.groupby('iteration')['PTNR'].std()

        axs[0].plot(iterations, means, label=algo, linewidth=2)
        axs[0].fill_between(iterations, means - 0.5*stds, means + 0.5*stds, alpha=0.1)

    axs[0].set_ylabel('PTNR (dB)')
    axs[0].set_title('PTNR Convergence: Impact of Inner Optimizer Resets')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Inner LR Step Trace (Single Seed)
    seed_to_plot = df_iter['seed'].unique()[0]
    df_seed = df_iter[df_iter['seed'] == seed_to_plot]

    for algo in algorithms:
        if 'Fixed' in algo: continue # Fixed offset LR step is trivial
        df_algo = df_seed[df_seed['algorithm'] == algo]
        axs[1].plot(df_algo['iteration'], df_algo['lr_step'], label=algo, alpha=0.8)

    axs[1].set_ylabel('Inner LR Step Size $\\\\alpha$')
    axs[1].set_title(f'Inner Learning Rate Tracking (Seed {seed_to_plot})')
    axs[1].legend()
    axs[1].grid(True)

    # 3. Final PTNR Boxplots
    final_data = []
    for algo in algorithms:
        final_ptnrs = df_iter[(df_iter['algorithm'] == algo) & (df_iter['iteration'] == max(iterations))]['PTNR'].values
        final_data.append(final_ptnrs)

    axs[2].boxplot(final_data, tick_labels=algorithms)
    axs[2].set_ylabel('Final PTNR (dB)')
    axs[2].set_title('Statistical Distribution of Final Performance')
    axs[2].grid(True, axis='y')

    plt.tight_layout()
    plt.savefig('experiments/Fact_Check_Inner_Resets.png')
    print("Saved Fact Check plots to experiments/Fact_Check_Inner_Resets.png")

if __name__ == '__main__':
    num_seeds = 15
    N = 64
    B_total = 150

    algorithms = ['Fixed Offset', 'SA (Hard Reset)', 'SA (No Reset)', 'SA (Smart Reset)']
    all_iter_df = []

    for algo in algorithms:
        print(f"Running Fact Check: {algo}...")
        for seed in range(num_seeds):
            df_iter = run_fact_check_trial(seed, N, algo, B_total)
            all_iter_df.append(df_iter)

    final_df = pd.concat(all_iter_df, ignore_index=True)
    plot_fact_check(final_df)

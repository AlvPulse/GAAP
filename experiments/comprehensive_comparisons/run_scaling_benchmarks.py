import numpy as np
import pandas as pd
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from beamformer.element_model import SyntheticVaractor
from beamformer.pattern_projection import Task
from beamformer.synthesis import synthesize_schelkunoff
from beamformer.controllers.core import step_ap_lr, apply_phase_jump
from beamformer.controllers.metrics import get_metrics, get_sll
from beamformer.controllers.unified import StrainGuidedController
from beamformer.controllers.sa_families import SA, FixedOffset

def run_comprehensive_trial(seed, N, algo_name, B_total=200):
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

    target_gain = 20 * np.log10(N)

    # Configure Pipeline
    enable_ap = True
    enable_lr = True

    if algo_name == 'Strain-Guided Pipeline':
        controller = StrainGuidedController(N=N, target_gain_db=target_gain)
        refinement_steps_inner = 20
    elif algo_name == 'SA':
        controller = SA()
        refinement_steps_inner = 20

    elif algo_name == 'AP-Only':
        controller = FixedOffset()
        refinement_steps_inner = 0
    else:
        controller = FixedOffset()
        refinement_steps_inner = 20

    lr_initial = 0.05
    lr_step = lr_initial
    lr_min = 0.001
    lr_max = 0.2

    from beamformer.utils import oversampled_fft, oversampled_ifft
    from beamformer.pattern_projection import pattern_project
    from beamformer.local_refinement import refine_local_active_set
    from beamformer.incremental_af import IncrementalAFCache

    best_c, best_V, best_delta = c_n.copy(), V_n.copy(), current_delta
    best_null, best_gain, current_ptnr = get_metrics(best_c, task, element)
    best_sll = get_sll(best_c, task)
    current_cost = -current_ptnr

    history_res = [float('inf')]
    iter_data = []

    start_time = time.time()

    for k in range(1, B_total + 1):
        cand_delta = controller.propose(current_delta)
        cand_c, cand_V = apply_phase_jump(best_c, best_V, best_delta, cand_delta, element)

        tau, alpha, w_phase, w_amp, projection_method = 0.7, 0.7, 1.0, 0.5, 'euclidean'
        current_residual = 0.0

        # 1. AP Step
        if enable_ap:
            u_grid, F = oversampled_fft(cand_c * np.exp(-1j * cand_delta), 8)
            F_prime = pattern_project(F, u_grid, task, N)
            F_prime_damped = (1 - tau) * F + tau * F_prime
            cand_c_ideal = oversampled_ifft(F_prime_damped, N)
        else:
            cand_c_ideal = cand_c

        rotated_cand = cand_c_ideal * np.exp(1j * cand_delta)
        V_cand_proj = np.zeros(N)
        c_proj = np.zeros(N, dtype=np.complex128)

        for n in range(N):
            V_cand_proj[n], c_proj[n] = element.project(rotated_cand[n], method=projection_method, w_phase=w_phase, w_amp=w_amp)
            current_residual += np.abs(rotated_cand[n] - c_proj[n])**2

        cand_V = (1 - alpha) * cand_V + alpha * V_cand_proj

        # 2. LR Step
        acc_corr = 0
        if enable_lr and refinement_steps_inner > 0:
            V_n_pre_lr = cand_V.copy()
            c_lr_init = np.zeros(N, dtype=np.complex128)
            for n in range(N):
                c_lr_init[n] = element.get_complex_weight(cand_V[n])

            af_cache = IncrementalAFCache(N, task)
            af_cache.initialize(c_lr_init)

            cand_V = refine_local_active_set(
                cand_V, task, element, af_cache=af_cache,
                k_active=8, refinement_steps=refinement_steps_inner, step_size=lr_step
            )
            acc_corr = np.sum(np.abs(cand_V - V_n_pre_lr) > 1e-4)

        for n in range(N):
            cand_c[n] = element.get_complex_weight(cand_V[n])

        cand_res = current_residual

        cand_null, cand_gain, cand_ptnr = get_metrics(cand_c, task, element)
        cand_sll = get_sll(cand_c, task)
        cand_cost = -cand_ptnr

        accepted = controller.accept(current_cost, cand_cost)
        old_delta = current_delta

        if accepted:
            current_cost = cand_cost
            best_c, best_V, best_delta = cand_c, cand_V, cand_delta

            # Update running best metrics
            current_ptnr = cand_ptnr
            best_gain = cand_gain
            best_null = cand_null
            best_sll = cand_sll
            current_delta = cand_delta

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
            r_0=history_res[0] if history_res else 1.0,
            gain=best_gain,
            acc_corr=acc_corr if accepted else 0
        )

        # Population algorithms update their "best" internal states independently,
        # so we track the absolute best found so far for plotting convergence
        if hasattr(controller, 'gbest_fit'): # PSO
            current_cost = controller.gbest_fit
        elif hasattr(controller, 'fitness'): # GA
            best_idx = np.argmin(controller.fitness) if np.any(controller.fitness != 0) else 0
            current_cost = controller.fitness[best_idx] if controller.fitness[best_idx] != 0 else current_cost

        if hasattr(controller, 'lr_step'):
            lr_step = controller.lr_step

        iter_data.append({
            'algorithm': algo_name,
            'N': N,
            'seed': seed,
            'iteration': k,
            'PTNR': -current_cost,
            'gain': best_gain,
            'null_depth': best_null,
            'SLL': best_sll
        })

        history_res.append(cand_res if accepted else history_res[-1])

    runtime = time.time() - start_time
    df_iter = pd.DataFrame(iter_data)

    ptnrs = df_iter['PTNR'].values
    def time_to(target_db):
        idx = np.where(ptnrs >= target_db)[0]
        return idx[0] + 1 if len(idx) > 0 else B_total

    trial_data = {
        'algorithm': algo_name,
        'N': N,
        'seed': seed,
        'final_PTNR': ptnrs[-1],
        'final_gain': best_gain,
        'final_null': best_null,
        'final_SLL': best_sll,
        'runtime': runtime,
        'TTT_30dB': time_to(30),
        'TTT_40dB': time_to(40),
        'TTT_50dB': time_to(50)
    }

    return df_iter, trial_data

def run_gand_baseline(seed, N, B_total=200):
    import time
    import numpy as np
    import pandas as pd
    from beamformer.element_model import SyntheticVaractor
    from beamformer.pattern_projection import Task
    from beamformer.synthesis import synthesize_schelkunoff
    from beamformer.controllers.metrics import get_metrics, get_sll

    np.random.seed(seed)
    element = SyntheticVaractor(beta=1.5, folding=True)
    task = Task(type='nulled', target_angles=[0.3], null_angles=[-0.5, 0.1, 0.7], sll_ceiling=0.1)

    # Init pop around schelkunoff
    init_c = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)
    init_V = np.zeros(N)
    for n in range(N):
        init_V[n], _ = element.project(init_c[n])

    pop_size = 20
    # Population of branch assignments (voltages)
    pop = np.zeros((pop_size, N))
    for i in range(pop_size):
        pop[i] = init_V + np.random.normal(0, 1.0, N)
        pop[i] = np.clip(pop[i], element.v_min, element.v_max)

    iter_data = []
    start_time = time.time()

    for k in range(1, B_total + 1):
        fitness = np.zeros(pop_size)
        c_pops = np.zeros((pop_size, N), dtype=np.complex128)

        # Evaluate
        for i in range(pop_size):
            for n in range(N):
                c_pops[i, n] = element.get_complex_weight(pop[i, n])

            _, _, ptnr = get_metrics(c_pops[i], task, element)
            fitness[i] = ptnr

        # Log best
        best_idx = np.argmax(fitness)
        best_c = c_pops[best_idx]
        null_depth, gain, ptnr = get_metrics(best_c, task, element)
        sll = get_sll(best_c, task)

        iter_data.append({
            'algorithm': 'GA-Monolithic',
            'N': N,
            'seed': seed,
            'iteration': k,
            'PTNR': ptnr,
            'gain': gain,
            'null_depth': null_depth,
            'SLL': sll
        })

        # Tournament Selection
        new_pop = np.zeros((pop_size, N))
        for i in range(pop_size):
            i1, i2 = np.random.choice(pop_size, 2, replace=False)
            winner = i1 if fitness[i1] > fitness[i2] else i2
            new_pop[i] = pop[winner].copy()

        # Single-point crossover
        for i in range(0, pop_size-1, 2):
            if np.random.rand() < 0.8:
                pt = np.random.randint(1, N-1)
                temp = new_pop[i, pt:].copy()
                new_pop[i, pt:] = new_pop[i+1, pt:]
                new_pop[i+1, pt:] = temp

        # Mutation (creep)
        mutation_mask = np.random.rand(pop_size, N) < 0.1
        mutations = np.random.normal(0, 0.5, (pop_size, N))
        new_pop += mutation_mask * mutations
        new_pop = np.clip(new_pop, element.v_min, element.v_max)

        # Elitism
        new_pop[0] = pop[best_idx]
        pop = new_pop

    runtime = time.time() - start_time
    df_iter = pd.DataFrame(iter_data)
    ptnrs = df_iter['PTNR'].values

    def time_to(target_db):
        idx = np.where(ptnrs >= target_db)[0]
        return idx[0] + 1 if len(idx) > 0 else B_total

    trial_data = {
        'algorithm': 'GA-Monolithic',
        'N': N,
        'seed': seed,
        'final_PTNR': ptnrs[-1],
        'final_gain': df_iter['gain'].iloc[-1],
        'final_null': df_iter['null_depth'].iloc[-1],
        'final_SLL': df_iter['SLL'].iloc[-1],
        'runtime': runtime,
        'TTT_30dB': time_to(30),
        'TTT_40dB': time_to(40),
        'TTT_50dB': time_to(50)
    }
    return df_iter, trial_data

def run_pgd_baseline(seed, N, B_total=200):
    import time
    import numpy as np
    import pandas as pd
    from beamformer.element_model import SyntheticVaractor
    from beamformer.pattern_projection import Task
    from beamformer.synthesis import synthesize_schelkunoff, get_steering_vector
    from beamformer.controllers.metrics import get_metrics, get_sll

    np.random.seed(seed)
    element = SyntheticVaractor(beta=1.5, folding=True)
    task = Task(type='nulled', target_angles=[0.3], null_angles=[-0.5, 0.1, 0.7], sll_ceiling=0.1)

    initial_weights = synthesize_schelkunoff(N, task.target_angles[0], task.null_angles)
    phases = np.angle(initial_weights)
    lr = 0.01

    iter_data = []
    start_time = time.time()

    # We step PGD iteration by iteration to log it identically
    for k in range(1, B_total + 1):
        c_n = np.zeros(N, dtype=np.complex128)
        for n in range(N):
            _, c_n[n] = element.project(np.exp(1j * phases[n]), method='phase_only')

        grad = np.zeros(N)
        for nu in task.null_angles:
            sv = get_steering_vector(N, nu)
            af_null = np.sum(c_n * np.conj(sv))
            dAF_dphi = 1j * c_n * np.conj(sv)
            grad += 2 * np.real(af_null * np.conj(dAF_dphi))

        for ta in task.target_angles:
            sv = get_steering_vector(N, ta)
            af_target = np.sum(c_n * np.conj(sv))
            dAF_dphi = 1j * c_n * np.conj(sv)
            grad -= 2 * np.real(af_target * np.conj(dAF_dphi)) * 0.1

        phases -= lr * grad

        # Log state
        null_depth, gain, ptnr = get_metrics(c_n, task, element)
        sll = get_sll(c_n, task)

        iter_data.append({
            'algorithm': 'PGD-Baseline',
            'N': N,
            'seed': seed,
            'iteration': k,
            'PTNR': ptnr,
            'gain': gain,
            'null_depth': null_depth,
            'SLL': sll
        })

    runtime = time.time() - start_time
    df_iter = pd.DataFrame(iter_data)
    ptnrs = df_iter['PTNR'].values

    def time_to(target_db):
        idx = np.where(ptnrs >= target_db)[0]
        return idx[0] + 1 if len(idx) > 0 else B_total

    trial_data = {
        'algorithm': 'PGD-Baseline',
        'N': N,
        'seed': seed,
        'final_PTNR': ptnrs[-1],
        'final_gain': df_iter['gain'].iloc[-1],
        'final_null': df_iter['null_depth'].iloc[-1],
        'final_SLL': df_iter['SLL'].iloc[-1],
        'runtime': runtime,
        'TTT_30dB': time_to(30),
        'TTT_40dB': time_to(40),
        'TTT_50dB': time_to(50)
    }
    return df_iter, trial_data

def run_scaling_benchmark():
    os.makedirs('experiments/comprehensive_comparisons/data', exist_ok=True)

    scales = [16, 64, 256]
    algorithms = ['Strain-Guided Pipeline', 'AP-Only', 'PGD-Baseline', 'GA-Monolithic']
    num_seeds = 2
    B_total = 50

    all_iter_df = []
    all_trial_data = []

    for N in scales:
        for algo in algorithms:
            print(f"Benchmarking {algo} at N={N}...")
            for seed in range(num_seeds):
                if algo == 'PGD-Baseline':
                    df_iter, trial_dict = run_pgd_baseline(seed, N, B_total)
                elif algo == 'GA-Monolithic':
                    df_iter, trial_dict = run_gand_baseline(seed, N, B_total)
                else:
                    df_iter, trial_dict = run_comprehensive_trial(seed, N, algo, B_total)

                all_iter_df.append(df_iter)
                all_trial_data.append(trial_dict)

    final_iter_df = pd.concat(all_iter_df, ignore_index=True)
    final_trial_df = pd.DataFrame(all_trial_data)

    final_iter_df.to_csv('experiments/comprehensive_comparisons/data/scaling_iterations.csv', index=False)
    final_trial_df.to_csv('experiments/comprehensive_comparisons/data/scaling_trials.csv', index=False)
    print("Saved Comprehensive Scaling benchmark data.")

if __name__ == '__main__':
    run_scaling_benchmark()

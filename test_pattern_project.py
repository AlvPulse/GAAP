from beamformer.pattern_projection import Task, pattern_project
import numpy as np

def test():
    N = 64
    task = Task('pencil', target_angles=[0.5], sll_ceiling=0.1)
    # create dummy F
    u = np.linspace(-1, 1, 1000)
    F = np.ones(1000) * 10
    F_new = pattern_project(F, u, task, N)
    print(np.max(np.abs(F_new)))

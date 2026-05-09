import numpy as np
import time
import sys
sys.path.append('..')

from beamformer.pattern_projection import Task
from beamformer.element_model import SyntheticVaractor
from beamformer.api import beamform

def main():
    print("Running Scaling Study (N=64 to 1024)...")
    sizes = [64, 128, 256, 512, 1024]
    element = SyntheticVaractor(beta=0.5, folding=True)

    for N in sizes:
        task = Task('pencil', target_angles=[0.2])

        start_time = time.time()
        # Restrict iterations for benchmarking speed
        res = beamform(task, element, N=N, K_max=20)
        end_time = time.time()

        duration_ms = (end_time - start_time) * 1000
        print(f"N={N:4d} | Time for 20 iterations: {duration_ms:6.1f} ms | Final Res: {res['history']['residuals'][-1]:.4f}")

if __name__ == '__main__':
    main()

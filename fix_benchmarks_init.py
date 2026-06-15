import os
import re

filepath = 'experiments/run_final_benchmarks.py'

with open(filepath, 'r') as f:
    content = f.read()

# Replace Schelkunoff initialization with Pencil Beam and use Euclidean
content = content.replace(
    'from beamformer.synthesis import synthesize_schelkunoff',
    'from beamformer.synthesis import get_steering_vector'
)

content = content.replace(
    '''    # Init
    w = synthesize_schelkunoff(N, target_u, null_u)
    w /= np.max(np.abs(w))''',
    '''    # Init (Pencil Beam for maximum gain preservation)
    w = get_steering_vector(N, target_u)'''
)

content = content.replace("method='phase_only'", "method='euclidean'")

with open(filepath, 'w') as f:
    f.write(content)

print("Updated run_final_benchmarks.py")

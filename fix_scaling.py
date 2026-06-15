import re

with open('experiments/comprehensive_comparisons/run_scaling_benchmarks.py', 'r') as f:
    content = f.read()

# Reduce N sizes and budgets so it doesn't timeout
content = content.replace("scales = [16, 64, 256, 1024]", "scales = [16, 64, 256]")
content = content.replace("B_total = 200", "B_total = 50")
content = content.replace("num_seeds = 3", "num_seeds = 2")

with open('experiments/comprehensive_comparisons/run_scaling_benchmarks.py', 'w') as f:
    f.write(content)

import re
with open('experiments/ultimate_fact_check.py', 'r') as f:
    content = f.read()

content = content.replace("num_seeds = 15", "num_seeds = 5")

with open('experiments/ultimate_fact_check.py', 'w') as f:
    f.write(content)

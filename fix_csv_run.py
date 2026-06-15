import re
with open('experiments/run_final_paper_benchmarks.py', 'r') as f:
    content = f.read()

# Add pandas to install
import subprocess
try:
    import pandas
except ImportError:
    subprocess.run(['pip', 'install', 'pandas'])

import subprocess
import os

os.makedirs("experiments/data", exist_ok=True)

scripts = [
    ("experiments/ablation_mrlcmv.py", "E1: Canonical Ablation (MR-LCMV Components)"),
    ("experiments/robustness_study.py", "E8: Robustness Study"),
]

with open("experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md", "w") as f:
    f.write("# Comprehensive Experimental Data Ledger\n\n")

for script, title in scripts:
    print(f"Running {title}...")
    with open("experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md", "a") as f:
        f.write(f"## {title}\n")
        f.write(f"`Command: python {script}`\n\n```text\n")

    res = subprocess.run(["python", script], capture_output=True, text=True, timeout=120)
    with open("experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md", "a") as f:
        f.write(res.stdout)
        if res.stderr:
            f.write("\n[STDERR]\n" + res.stderr)
        f.write("\n```\n\n")

print("Done generating log.")

import subprocess
import os

os.makedirs("experiments/data", exist_ok=True)

scripts = [
    ("experiments/ablation_mrlcmv.py", "E1: Canonical Ablation (MR-LCMV Components)"),
    ("experiments/sota_families_benchmark.py", "E5: SOTA Families Benchmark (Measured Locus)"),
    ("experiments/robustness_study.py", "E8: Robustness Study"),
    ("experiments/run_real_benchmark.py", "Convergence Scaling vs N"),
]

with open("experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md", "w") as f:
    f.write("# Comprehensive Experimental Data Ledger\n\n")
    f.write("This log contains the exact, unedited standard output from all major evaluation scripts. It serves as the quantitative foundation for the paper's claims.\n\n")

for script, title in scripts:
    print(f"Running {title}...")
    with open("experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md", "a") as f:
        f.write(f"## {title}\n")
        f.write(f"`Command: python {script}`\n\n```text\n")

    try:
        res = subprocess.run(["python", script], capture_output=True, text=True, timeout=300)
        with open("experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md", "a") as f:
            f.write(res.stdout)
            if res.stderr:
                f.write("\n[STDERR]\n" + res.stderr)
            f.write("\n```\n\n")
    except subprocess.TimeoutExpired:
        print(f"  -> {script} timed out.")
        with open("experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md", "a") as f:
            f.write(f"[Process timed out after 300 seconds]\n```\n\n")

print("All scripts executed. Ledger generated at experiments/data/ALL_EXPERIMENTAL_RESULTS_LOG.md")

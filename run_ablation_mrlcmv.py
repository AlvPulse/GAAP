import sys
sys.path.append('.')
import subprocess

with open("VERIFIED_EXPERIMENTAL_RESULTS.md", "w") as out_f:
    out_f.write("# Verified Experimental Ledger\n\n")
    out_f.write("This file contains the raw, captured STDOUT of the central evaluation scripts to strictly anchor the final narrative to measured quantitative evidence.\n\n")

scripts_to_run = [
    ("experiments/ablation_mrlcmv.py", "Ablation of MR-LCMV Components"),
    ("experiments/robustness_study.py", "Robustness & Null Limits"),
    ("experiments/ultimate_fact_check.py", "Ultimate Fact Check (Convergence Trajectories)")
]

for script, title in scripts_to_run:
    print(f"Running {script}...")
    try:
        res = subprocess.run(["python", script], capture_output=True, text=True, timeout=120)
        with open("VERIFIED_EXPERIMENTAL_RESULTS.md", "a") as out_f:
            out_f.write(f"## {title} (`{script}`)\n```text\n")
            out_f.write(res.stdout)
            if res.stderr:
                out_f.write("\nSTDERR:\n" + res.stderr)
            out_f.write("\n```\n\n")
    except subprocess.TimeoutExpired:
        print(f"Timeout on {script}")
        with open("VERIFIED_EXPERIMENTAL_RESULTS.md", "a") as out_f:
            out_f.write(f"## {title} (`{script}`)\n```text\n[Execution Timed Out]\n```\n\n")

print("Finished capturing core empirical scripts.")

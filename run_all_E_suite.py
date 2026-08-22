import subprocess
import os

os.makedirs("experiments/results", exist_ok=True)
os.makedirs("experiments/figures_measured", exist_ok=True)

scripts = [
    ("E1 Ablation", "experiments/run_E1_ablation.py"),
    ("E2 Severity Sweep", "experiments/run_E2_severity_sweep.py"),
    ("E3 Mechanism Factorial", "experiments/run_E3_mechanism_factorial.py"),
    ("E4 Scaling & Benchmark", "experiments/convergence_benchmark.py"),
    ("E5 SOTA Families", "experiments/sota_families_benchmark.py"),
    ("E6 Causal Chain", "experiments/run_E6_causal_chain.py"),
    ("E7 Psi Saturation", "experiments/run_E7_psi_resolution.py"),
    ("E8 Robustness Check", "experiments/robustness_study.py"),
]

with open("experiments/results/MASTER_EXECUTION_LOG.txt", "w") as log_file:
    log_file.write("=== MASTER EXECUTION LOG ===\n\n")

for title, script in scripts:
    print(f"\n======================================")
    print(f"Running {title} ({script})")
    print(f"======================================")

    with open("experiments/results/MASTER_EXECUTION_LOG.txt", "a") as log_file:
        log_file.write(f"\n======================================\n")
        log_file.write(f"Executing: {title}\n")
        log_file.write(f"======================================\n")

    try:
        # Run process and stream stdout to both console and log
        process = subprocess.Popen(["python", script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                with open("experiments/results/MASTER_EXECUTION_LOG.txt", "a") as log_file:
                    log_file.write(output)

        rc = process.poll()
        if rc != 0:
            print(f"[{title}] exited with error code {rc}")

    except Exception as e:
        print(f"Failed to execute {script}: {e}")

print("\nMaster Suite completed. Output saved to experiments/results/MASTER_EXECUTION_LOG.txt")

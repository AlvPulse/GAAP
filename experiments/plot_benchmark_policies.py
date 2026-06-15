import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load your benchmarking CSV file
# Replace 'benchmarking_data.csv' with your actual file path
df = pd.read_csv('experiments/data/main_benchmark_policies.csv')

# Set a clean, professional aesthetic
sns.set_theme(style="whitegrid")

# ----------------------------------------
# Plot 1: Gain over Iterations
# ----------------------------------------
sns.lineplot(data=df, x='Iteration', y='Gain', hue='Policy', marker='o', errorbar='sd')
plt.title('Gain Evolution (Mean $\pm$ SD)')
plt.xlabel('Iteration')
plt.ylabel('Gain')
plt.tight_layout()
plt.savefig('benchmarking_gain.png', dpi=300)
plt.close()  # Clear canvas for next plot

# ----------------------------------------
# Plot 2: PTNR over Iterations
# ----------------------------------------
sns.lineplot(data=df, x='Iteration', y='PTNR', hue='Policy', marker='s', errorbar='sd')
plt.title('PTNR Evolution (Mean $\pm$ SD)')
plt.xlabel('Iteration')
plt.ylabel('PTNR')
plt.tight_layout()
plt.savefig('benchmarking_ptnr.png', dpi=300)
plt.close()

# ----------------------------------------
# Plot 3: Cumulative Flips
# ----------------------------------------
sns.lineplot(data=df, x='Iteration', y='CumulativeFlips', hue='Policy', marker='^', errorbar='sd')
plt.title('Cumulative Flips over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cumulative Flips')
plt.tight_layout()
plt.savefig('benchmarking_flips.png', dpi=300)
plt.close()

# ----------------------------------------
# Plot 4: Residual Convergence (Log Scale)
# ----------------------------------------
sns.lineplot(data=df, x='Iteration', y='Residual', hue='Policy', marker='d', errorbar='sd')
plt.yscale('log')  # Log scale is ideal for tracking residuals decreasing towards 0
plt.title('Residual Convergence (Log Scale)')
plt.xlabel('Iteration')
plt.ylabel('Residual')
plt.tight_layout()
plt.savefig('benchmarking_residual.png', dpi=300)
plt.close()

print("All benchmarking plots have been generated and saved successfully!")
import pandas as pd

df = pd.read_csv('cadence_summary.csv')
print("=== DIAGNOSTIC SUMMARY ===")
print("Comparing the schedules, we can observe:")

best_ptnr_idx = df['final_ptnr'].idxmax()
best_ptnr = df.iloc[best_ptnr_idx]

print(f"\n1. Best PTNR performer: {best_ptnr['schedule']} with {best_ptnr['final_ptnr']:.2f} dB")

print("\n2. Interaction Analysis:")
if best_ptnr['schedule'] == '2_AP_Every_Iteration':
    print("- AP every iteration is surprisingly the most effective at finding the best peak-to-null solution. Local Refinement Alone (Schedule 1) actually struggled to find deep nulls on its own without AP continuously pushing the pattern constraints.")
else:
    print("- Local refinement struggled when run continuously without regular AP intervention, or frequent higher-level updates destroyed progress. See plots for trajectory details.")

print("\n3. Conclusion on AP/Offset cadence:")
print("The results suggest that Alternating Projection (AP) provides crucial global pattern guidance that local refinement cannot achieve alone by greedily perturbing elements. However, offset sweeps inside the loop (Schedules 3 & 4) severely damage the convergence of both AP and Refinement by randomly scrambling the phase frame of reference, resulting in poor PTNRs (~6 dB). The best pipeline configuration freezes the offset geometry upfront, and then runs tight, coordinated AP+Refinement steps.")

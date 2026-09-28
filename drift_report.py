import pandas as pd
import numpy as np

df = pd.read_csv('data/processed_FD001.csv')
feature_cols = [c for c in df.columns if c not in ['unit_number', 'time_cycles', 'max_cycle', 'RUL']]

reference = df[df['time_cycles'] <= 20]
current = df[df['time_cycles'] > 100]

print(f"Reference rows: {len(reference)}, Current rows: {len(current)}\n")

def calculate_psi(ref_values, curr_values, bins=10):
    breakpoints = np.percentile(ref_values, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(ref_values, bins=breakpoints)
    curr_counts, _ = np.histogram(curr_values, bins=breakpoints)

    ref_pct = ref_counts / len(ref_values)
    curr_pct = curr_counts / len(curr_values)

    ref_pct = np.where(ref_pct == 0, 0.0001, ref_pct)
    curr_pct = np.where(curr_pct == 0, 0.0001, curr_pct)

    psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
    return psi

results = []
for col in feature_cols:
    psi = calculate_psi(reference[col].values, current[col].values)
    results.append((col, psi))

results_df = pd.DataFrame(results, columns=['feature', 'psi'])
results_df = results_df.sort_values('psi', ascending=False)

def flag(psi):
    if psi < 0.1:
        return "stable"
    elif psi < 0.25:
        return "moderate drift"
    else:
        return "significant drift"

results_df['status'] = results_df['psi'].apply(flag)

print(results_df.to_string(index=False))
print(f"\n{(results_df['psi'] >= 0.25).sum()} of {len(results_df)} features show significant drift (PSI >= 0.25)")

results_df.to_csv('drift_report.csv', index=False)
print("Saved drift_report.csv")
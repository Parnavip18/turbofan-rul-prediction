import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

df = pd.read_csv('data/processed_FD001.csv')
print(df.shape)

unit_ids = df['unit_number'].unique()
train_units, test_units = train_test_split(unit_ids, test_size=0.2, random_state=42)

train_df = df[df['unit_number'].isin(train_units)]
test_df = df[df['unit_number'].isin(test_units)].copy()

feature_cols = [c for c in df.columns if c not in ['unit_number', 'time_cycles', 'max_cycle', 'RUL']]
print(len(feature_cols), "features")

# "Healthy" reference: early cycles of training engines, before real
# degradation has had time to develop. Shortest engine in this dataset
# runs 128 cycles, so cycle <= 20 is safely early-life for every engine.
healthy_df = train_df[train_df['time_cycles'] <= 20]
print("Healthy reference rows:", healthy_df.shape[0])

iso_forest = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
iso_forest.fit(healthy_df[feature_cols])

test_df['anomaly_score'] = iso_forest.decision_function(test_df[feature_cols])
test_df['is_anomaly'] = iso_forest.predict(test_df[feature_cols])  # -1 = anomaly, 1 = normal

# Real validation: does anomaly rate rise as engines approach failure?
bins = [0, 25, 50, 75, 100, 150, 400]
labels = ['0-25', '25-50', '50-75', '75-100', '100-150', '150+']
test_df['RUL_bin'] = pd.cut(test_df['RUL'], bins=bins, labels=labels)

anomaly_rate_by_rul = test_df.groupby('RUL_bin', observed=True)['is_anomaly'].apply(lambda x: (x == -1).mean())
print(anomaly_rate_by_rul)

plt.figure(figsize=(8, 5))
anomaly_rate_by_rul.plot(kind='bar')
plt.xlabel('RUL bin (higher = healthier)')
plt.ylabel('Fraction flagged anomalous')
plt.title('Anomaly rate vs. Remaining Useful Life')
plt.tight_layout()
plt.savefig('anomaly_vs_rul.png')
print("Saved anomaly_vs_rul.png")

joblib.dump(iso_forest, 'isolation_forest_model.pkl')
print("Saved isolation_forest_model.pkl")
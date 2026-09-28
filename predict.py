import joblib
import pandas as pd

model = joblib.load('rf_model.pkl')
print("Model loaded:", type(model))

df = pd.read_csv('data/processed_FD001.csv')
feature_cols = [c for c in df.columns if c not in ['unit_number', 'time_cycles', 'max_cycle', 'RUL']]

print("Number of features:", len(feature_cols))
print(feature_cols)

sample_row = df[feature_cols].iloc[[0]]
prediction = model.predict(sample_row)
print("Predicted RUL:", prediction[0])
print("Actual RUL:", df['RUL'].iloc[0])

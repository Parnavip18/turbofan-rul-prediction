import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Turbofan RUL Dashboard", layout="wide")
st.title("Turbofan Engine — Predictive Maintenance Dashboard")

df = pd.read_csv('data/processed_FD001.csv')
model = joblib.load('rf_model.pkl')
feature_cols = [c for c in df.columns if c not in ['unit_number', 'time_cycles', 'max_cycle', 'RUL']]

engine_ids = sorted(df['unit_number'].unique())
selected_engine = st.selectbox("Select engine", engine_ids)

engine_df = df[df['unit_number'] == selected_engine].sort_values('time_cycles')

predictions = model.predict(engine_df[feature_cols])
engine_df = engine_df.copy()
engine_df['predicted_RUL'] = predictions

col1, col2, col3 = st.columns(3)
col1.metric("Current cycle", int(engine_df['time_cycles'].iloc[-1]))
col2.metric("Actual RUL", int(engine_df['RUL'].iloc[-1]))
col3.metric("Predicted RUL", round(float(engine_df['predicted_RUL'].iloc[-1]), 1))

st.subheader("Predicted vs Actual RUL over time")
chart_data = engine_df[['time_cycles', 'RUL', 'predicted_RUL']].set_index('time_cycles')
st.line_chart(chart_data)

st.subheader("Sensor readings over time")
sensor_choice = st.selectbox("Select sensor", [c for c in feature_cols if not c.endswith('rollmean') and not c.endswith('rollstd')])
st.line_chart(engine_df.set_index('time_cycles')[sensor_choice])

st.subheader("Drift monitoring summary")
drift_df = pd.read_csv('drift_report.csv')
st.dataframe(drift_df, use_container_width=True)
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Turbofan RUL Prediction API")

model = joblib.load('rf_model.pkl')

feature_cols = [
    'sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 'sensor_9',
    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 'sensor_17',
    'sensor_20', 'sensor_21', 'sensor_2_rollmean', 'sensor_2_rollstd',
    'sensor_3_rollmean', 'sensor_3_rollstd', 'sensor_4_rollmean', 'sensor_4_rollstd',
    'sensor_7_rollmean', 'sensor_7_rollstd', 'sensor_8_rollmean', 'sensor_8_rollstd',
    'sensor_9_rollmean', 'sensor_9_rollstd', 'sensor_11_rollmean', 'sensor_11_rollstd',
    'sensor_12_rollmean', 'sensor_12_rollstd', 'sensor_13_rollmean', 'sensor_13_rollstd',
    'sensor_14_rollmean', 'sensor_14_rollstd', 'sensor_15_rollmean', 'sensor_15_rollstd',
    'sensor_17_rollmean', 'sensor_17_rollstd', 'sensor_20_rollmean', 'sensor_20_rollstd',
    'sensor_21_rollmean', 'sensor_21_rollstd'
]


class EngineReading(BaseModel):
    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float
    sensor_2_rollmean: float
    sensor_2_rollstd: float
    sensor_3_rollmean: float
    sensor_3_rollstd: float
    sensor_4_rollmean: float
    sensor_4_rollstd: float
    sensor_7_rollmean: float
    sensor_7_rollstd: float
    sensor_8_rollmean: float
    sensor_8_rollstd: float
    sensor_9_rollmean: float
    sensor_9_rollstd: float
    sensor_11_rollmean: float
    sensor_11_rollstd: float
    sensor_12_rollmean: float
    sensor_12_rollstd: float
    sensor_13_rollmean: float
    sensor_13_rollstd: float
    sensor_14_rollmean: float
    sensor_14_rollstd: float
    sensor_15_rollmean: float
    sensor_15_rollstd: float
    sensor_17_rollmean: float
    sensor_17_rollstd: float
    sensor_20_rollmean: float
    sensor_20_rollstd: float
    sensor_21_rollmean: float
    sensor_21_rollstd: float


@app.get("/")
def root():
    return {"message": "Turbofan RUL Prediction API is running"}


@app.post("/predict")
def predict(reading: EngineReading):
    row = pd.DataFrame([reading.dict()])[feature_cols]
    predicted_rul = model.predict(row)[0]
    return {"predicted_RUL": float(predicted_rul)}
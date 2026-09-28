import pandas as pd
import numpy as np
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

df = pd.read_csv('data/processed_FD001.csv')

unit_ids = df['unit_number'].unique()
train_units, test_units = train_test_split(unit_ids, test_size=0.2, random_state=42)

train_df = df[df['unit_number'].isin(train_units)]
test_df = df[df['unit_number'].isin(test_units)]

feature_cols = [c for c in df.columns if c not in ['unit_number', 'time_cycles', 'max_cycle', 'RUL']]

X_train, y_train = train_df[feature_cols], train_df['RUL']
X_test, y_test = test_df[feature_cols], test_df['RUL']

mlflow.set_experiment("turbofan_RUL_prediction")

# --- Run 1: Linear Regression baseline ---
with mlflow.start_run(run_name="linear_regression"):
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)

    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.sklearn.log_model(model, name="model", serialization_format="pickle")
    print(f"Linear Regression -- RMSE: {rmse:.2f}, MAE: {mae:.2f}")

# --- Run 2: Random Forest, raw RUL ---
with mlflow.start_run(run_name="random_forest_raw"):
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)

    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("rul_clipped", False)
    mlflow.log_metric("rmse", rf_rmse)
    mlflow.log_metric("mae", rf_mae)
    mlflow.sklearn.log_model(rf_model, name="model", serialization_format="pickle")
    print(f"Random Forest (raw RUL) -- RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")

# --- Run 3: Random Forest, clipped RUL (your best model) ---
RUL_CAP = 125
with mlflow.start_run(run_name="random_forest_clipped"):
    y_train_clipped = y_train.clip(upper=RUL_CAP)
    y_test_clipped = y_test.clip(upper=RUL_CAP)

    rf_clipped_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_clipped_model.fit(X_train, y_train_clipped)
    rf_clipped_preds = rf_clipped_model.predict(X_test)

    rf_clipped_rmse = np.sqrt(mean_squared_error(y_test_clipped, rf_clipped_preds))
    rf_clipped_mae = mean_absolute_error(y_test_clipped, rf_clipped_preds)

    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("rul_clipped", True)
    mlflow.log_param("rul_cap", RUL_CAP)
    mlflow.log_metric("rmse", rf_clipped_rmse)
    mlflow.log_metric("mae", rf_clipped_mae)
    mlflow.sklearn.log_model(rf_clipped_model, name="model", serialization_format="pickle")
    print(f"Random Forest (clipped RUL) -- RMSE: {rf_clipped_rmse:.2f}, MAE: {rf_clipped_mae:.2f}")

joblib.dump(rf_clipped_model, 'rf_model.pkl')
print("Best model also saved to rf_model.pkl")
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

print("[1/3] Loading engineered dataset...")
df = pd.read_pickle("engineered_aqi_features.pkl")

# Define model feature schema
feature_cols = [
    "latitude", "longitude", "is_weekend",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "temperature", "humidity", "wind_speed", "visibility",
    "pm25", "pm10", "no2", "so2", "co", "o3", "aqi",
    "pm25_lag_1", "pm10_lag_1", "aqi_lag_1", "wind_speed_lag_1", "visibility_lag_1",
    "pm25_lag_2", "pm10_lag_2", "aqi_lag_2", "wind_speed_lag_2", "visibility_lag_2",
    "pm25_lag_4", "pm10_lag_4", "aqi_lag_4", "wind_speed_lag_4", "visibility_lag_4",
    "pm25_roll_mean_4", "pm25_roll_max_4", "pm10_roll_mean_4", "aqi_roll_mean_4"
]

X_train = df[df["year"] < 2025][feature_cols]
y_train = df[df["year"] < 2025]["target_aqi_next"]

print(f"[2/3] Fitting final model on {len(X_train):,} rows...")
model = HistGradientBoostingRegressor(
    max_iter=300,
    learning_rate=0.05,
    max_leaf_nodes=63,
    random_state=42
)
model.fit(X_train, y_train)

print("[3/3] Exporting model bundle...")
artifacts = {
    "model": model,
    "feature_names": feature_cols
}
joblib.dump(artifacts, "aqi_forecast_model.pkl")
print("Saved production model as 'aqi_forecast_model.pkl'.")
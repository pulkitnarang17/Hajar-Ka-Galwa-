import joblib
import numpy as np
import pandas as pd

class AQIPredictor:
    def __init__(self, model_path="aqi_forecast_model.pkl"):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_names = bundle["feature_names"]

    def _get_category(self, aqi: float) -> str:
        if aqi <= 50: return "Good"
        if aqi <= 100: return "Satisfactory"
        if aqi <= 200: return "Moderate"
        if aqi <= 300: return "Poor"
        if aqi <= 400: return "Very Poor"
        return "Severe"

    def predict_next(self, current_data: dict, history_df: pd.DataFrame) -> dict:
        dt = pd.to_datetime(current_data["datetime"])
        
        # Periodic temporal features
        hour = dt.hour
        doy = dt.day_of_year
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        doy_sin = np.sin(2 * np.pi * doy / 365.25)
        doy_cos = np.cos(2 * np.pi * doy / 365.25)
        is_weekend = 1 if dt.weekday() >= 5 else 0

        # Lags from history (offset -1 = 6h ago, -2 = 12h ago, -4 = 24h ago)
        lags = {}
        for lag_idx, offset in [(1, -1), (2, -2), (4, -4)]:
            past_row = history_df.iloc[offset]
            for col in ["pm25", "pm10", "aqi", "wind_speed", "visibility"]:
                lags[f"{col}_lag_{lag_idx}"] = past_row[col]

        # 24-hour rolling metrics
        recent_pm25 = np.append(history_df["pm25"].values[-3:], current_data["pm25"])
        recent_pm10 = np.append(history_df["pm10"].values[-3:], current_data["pm10"])
        recent_aqi  = np.append(history_df["aqi"].values[-3:],  current_data["aqi"])

        features_dict = {
            "latitude": current_data["latitude"],
            "longitude": current_data["longitude"],
            "is_weekend": is_weekend,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "doy_sin": doy_sin,
            "doy_cos": doy_cos,
            "temperature": current_data["temperature"],
            "humidity": current_data["humidity"],
            "wind_speed": current_data["wind_speed"],
            "visibility": current_data["visibility"],
            "pm25": current_data["pm25"],
            "pm10": current_data["pm10"],
            "no2": current_data["no2"],
            "so2": current_data["so2"],
            "co": current_data["co"],
            "o3": current_data["o3"],
            "aqi": current_data["aqi"],
            **lags,
            "pm25_roll_mean_4": float(np.mean(recent_pm25)),
            "pm25_roll_max_4": float(np.max(recent_pm25)),
            "pm10_roll_mean_4": float(np.mean(recent_pm10)),
            "aqi_roll_mean_4": float(np.mean(recent_aqi)),
        }

        feature_vector = pd.DataFrame([features_dict])[self.feature_names]
        predicted_aqi = float(self.model.predict(feature_vector)[0])
        predicted_aqi = max(0.0, min(500.0, round(predicted_aqi, 1)))

        return {
            "predicted_aqi_next": predicted_aqi,
            "predicted_category": self._get_category(predicted_aqi),
            "current_aqi": current_data["aqi"],
            "expected_change": round(predicted_aqi - current_data["aqi"], 1)
        }
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
from inference_engine import AQIPredictor

app = FastAPI(title="VayuDrishti AQI Predictor API")
predictor = AQIPredictor()

class SensorPayload(BaseModel):
    current: Dict[str, Any]
    history: List[Dict[str, Any]]

@app.get("/")
def root():
    return {"status": "online", "model": "HistGradientBoosting AQI Predictor"}

@app.post("/predict")
def predict(payload: SensorPayload):
    if len(payload.history) < 4:
        raise HTTPException(status_code=400, detail="Provide at least 4 past readings for lag calculation.")
    
    history_df = pd.DataFrame(payload.history)
    return predictor.predict_next(payload.current, history_df)
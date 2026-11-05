import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="Food Delivery Time Predictor")

BASE = Path(__file__).parent
model = joblib.load(BASE / "delivery_time_model.pkl")
with open(BASE / "model_metadata.json") as f:
    metadata = json.load(f)

experience_fallback = metadata["experience_level_mode_fallback"]


class OrderInput(BaseModel):
    Distance_km: float
    Preparation_Time_min: float
    Courier_Experience_yrs: float
    Weather: str
    Traffic_Level: str
    Time_of_Day: str
    Vehicle_Type: str


def categorize_distance(d):
    if d <= 6:
        return "Low distance"
    elif d <= 14:
        return "Medium distance"
    return "High distance"


def categorize_experience(y, fallback):
    if y <= 2:
        return "Junior"
    elif y <= 6:
        return "Mind-Level"
    elif y <= 9:
        return "Senior"
    return fallback


# The home page is now the HTML file
@app.get("/")
def home():
    return FileResponse(BASE / "index.html")


@app.get("/options")
def options():
    return metadata["categorical_options"]


@app.post("/predict")
def predict(data: OrderInput):
    row = pd.DataFrame([{
        "Distance_km": data.Distance_km,
        "Preparation_Time_min": data.Preparation_Time_min,
        "Courier_Experience_yrs": data.Courier_Experience_yrs,
        "Weather": data.Weather,
        "Traffic_Level": data.Traffic_Level,
        "Time_of_Day": data.Time_of_Day,
        "Vehicle_Type": data.Vehicle_Type,
        "Distance": categorize_distance(data.Distance_km),
        "Experience_level": categorize_experience(data.Courier_Experience_yrs, experience_fallback),
    }])
    prediction = model.predict(row)[0]
    return {"estimated_delivery_time_min": round(float(prediction))}

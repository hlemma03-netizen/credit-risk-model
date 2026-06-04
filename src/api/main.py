from fastapi import FastAPI
from src.api.pydantic_models import (
    CustomerData,
    PredictionResponse
)

import pandas as pd
import joblib

app = FastAPI()

model = joblib.load("best_model.pkl")


@app.get("/")
def home():
    return {"message": "Credit Risk API Running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerData):

    df = pd.DataFrame([data.dict()])

    probability = model.predict_proba(df)[0][1]

    return PredictionResponse(
        risk_probability=float(probability)
    )

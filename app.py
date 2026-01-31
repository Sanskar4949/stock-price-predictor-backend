from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Load model and scaler
# -----------------------------
MODEL_PATH = "saved_model/lstm_stock_model.h5"
SCALER_PATH = "saved_model/price_scaler.pkl"
SEQUENCE_LENGTH = 60

model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Stock Price Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -----------------------------
# Request schema
# -----------------------------
class PriceInput(BaseModel):
    prices: list

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict_price(data: PriceInput):

    if len(data.prices) < 60:
        return {"error": "Need at least 60 prices"}

    # ✅ Always take the last 60 prices
    prices = data.prices[-60:]

    prices = np.array(prices).reshape(1, 60, 1)
    prices = scaler.transform(prices.reshape(-1, 1)).reshape(1, 60, 1)

    prediction = model.predict(prices)
    predicted_price = scaler.inverse_transform(prediction)[0][0]

    return {"predicted_price": float(predicted_price)}


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def root():
    return {"message": "API is running"}

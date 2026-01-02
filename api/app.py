from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path

from src.inference import load_model, predict_single
from src.utils import resolve_repo_path

import torch
from src.dl.inference import load_preprocessor, load_dl_bundle, predict_single_dl


# --- App ---
app = FastAPI(title="Housing Price Prediction API")

# --- Load model on startup ---
MODEL_PATH = resolve_repo_path("models/best_model.joblib")
DL_MODEL_PATH = resolve_repo_path("models/dl/mlp.pt")
DL_PREPROCESSOR_PATH = resolve_repo_path("models/dl/preprocessor.joblib")

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    model = None
    load_error = str(e)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dl_model = None
dl_preprocessor = None
dl_load_error = None

try:
    dl_preprocessor = load_preprocessor(DL_PREPROCESSOR_PATH)
    dl_model, _ = load_dl_bundle(DL_MODEL_PATH, device=device)
except Exception as e:
    dl_load_error = str(e)


# --- Input Schema ---
class HouseInput(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    date_year: int
    date_month: int
    date_day: int


# --- Health check ---
@app.get("/health")
def health():
    return {"status": "ok"}


# --- Prediction endpoint ---
@app.post("/predict")
def predict(data: HouseInput):
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error}")

    try:
        prediction = predict_single(model, data.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict_dl")
def predict_dl(data: HouseInput):
    if dl_model is None or dl_preprocessor is None:
        raise HTTPException(status_code=500, detail=f"DL model not loaded: {dl_load_error}")

    try:
        pred = predict_single_dl(dl_model, dl_preprocessor, data.dict(), device=device)
        return {
            "prediction": pred,
            "device": str(device),
            "model_path": str(DL_MODEL_PATH),
            "preprocessor_path": str(DL_PREPROCESSOR_PATH),
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

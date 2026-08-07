"""
FastAPI serving layer for the trained TFT model.

Thin wrapper around src/forecasting_service.py, which holds the actual
loading and forecasting logic shared with the Streamlit demo.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.forecasting_service import load_serving_artifacts, generate_forecast

app = FastAPI(title="Electricity Load Forecasting API", version="1.0")

_state = {}


class QuantileForecast(BaseModel):
    client_id: str
    horizon_hours: int
    quantiles: list[float]
    forecast: list[list[float]]


@app.on_event("startup")
def startup():
    _state["artifacts"] = load_serving_artifacts("config.yaml")
    print(f"Loaded checkpoint: {_state['artifacts']['checkpoint_path']}")
    print(f"Loaded demo data: {_state['artifacts']['demo_data']['client_id'].nunique()} clients")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "artifacts" in _state}


@app.get("/clients")
def list_clients():
    clients = sorted(_state["artifacts"]["demo_data"]["client_id"].unique().tolist())
    return {"clients": clients}


@app.post("/forecast/{client_id}", response_model=QuantileForecast)
def forecast(client_id: str):
    try:
        result = generate_forecast(client_id, _state["artifacts"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return QuantileForecast(
        client_id=result["client_id"],
        horizon_hours=result["horizon_hours"],
        quantiles=result["quantiles"],
        forecast=result["forecast"],
    )
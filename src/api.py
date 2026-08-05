"""
FastAPI serving layer for the trained TFT model.

Serves forecasts for the small set of demo clients exported by
export_serving_artifacts.py -- NOT the full 314-client dataset, since that
data isn't committed to the repo.

NOTE on future-calendar features: a live forecast needs known-future
calendar values (hour, day_of_week, month, is_weekend) for the 24h horizon.
Since processed data only stores derived calendar columns (not the original
timestamp), these are reconstructed via simple wraparound arithmetic --
accurate for hour/day-of-week continuation, but does NOT handle
month-boundary edge cases correctly. Acceptable for a demo; called out here
as a known simplification.

Usage:
    uvicorn src.api:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer

app = FastAPI(title="Electricity Load Forecasting API", version="1.0")

_state = {}


class QuantileForecast(BaseModel):
    client_id: str
    horizon_hours: int
    quantiles: list[float]
    forecast: list[list[float]]  # shape: [horizon][n_quantiles]


def _generate_future_calendar(last_hour: int, last_dow: int, last_month: int, horizon: int):
    rows = []
    hour, dow = last_hour, last_dow
    for _ in range(horizon):
        hour = (hour + 1) % 24
        if hour == 0:
            dow = (dow + 1) % 7
        rows.append({
            "hour": hour,
            "day_of_week": dow,
            "month": last_month,
            "is_weekend": int(dow >= 5),
        })
    return rows


@app.on_event("startup")
def load_artifacts():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    _state["cfg"] = cfg

    schema_path = Path("models/training_dataset.pkl")
    if not schema_path.exists():
        raise RuntimeError(
            "models/training_dataset.pkl not found -- run "
            "`python -m src.export_serving_artifacts` first."
        )
    _state["training_schema"] = torch.load(schema_path, weights_only=False)

    ckpts = list(Path("models").glob("tft_best*.ckpt"))
    if not ckpts:
        raise RuntimeError("No TFT checkpoint found in models/ -- train the model first.")
    checkpoint_path = str(max(ckpts, key=lambda p: p.stat().st_mtime))

    raw_ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        k: (v.cpu() if isinstance(v, torch.Tensor) else v)
        for k, v in raw_ckpt["state_dict"].items()
    }
    hparams = raw_ckpt.get("hyper_parameters", {})

    model = TemporalFusionTransformer(**hparams)
    model.load_state_dict(state_dict)
    model.eval()

    # torchmetrics.Metric objects (the loss, and anything inside
    # logging_metrics) cache a private _device attribute set at
    # instantiation time -- a plain Python attribute, NOT a tensor, so it
    # survives state_dict remapping untouched and still points at cuda:0
    # from GPU training. This generically resets it on every submodule that
    # has one, rather than guessing which specific metric classes need
    # reconstructing.
    for module in model.modules():
        if hasattr(module, "_device"):
            module._device = torch.device("cpu")

    _state["model"] = model
    print(f"Loaded checkpoint: {checkpoint_path}")

    demo_path = Path("demo/sample_data.parquet")
    if not demo_path.exists():
        raise RuntimeError(
            "demo/sample_data.parquet not found -- run "
            "`python -m src.export_serving_artifacts` first."
        )
    _state["demo_data"] = pd.read_parquet(demo_path)
    print(f"Loaded demo data: {_state['demo_data']['client_id'].nunique()} clients")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/clients")
def list_clients():
    clients = sorted(_state["demo_data"]["client_id"].unique().tolist())
    return {"clients": clients}


@app.post("/forecast/{client_id}", response_model=QuantileForecast)
def forecast(client_id: str):
    demo_data = _state["demo_data"]
    cfg = _state["cfg"]

    client_hist = demo_data[demo_data.client_id == client_id].sort_values("time_idx")
    if client_hist.empty:
        raise HTTPException(status_code=404, detail=f"Unknown client_id: {client_id}")

    lookback = cfg["data"]["lookback_hours"]
    horizon = cfg["data"]["horizon_hours"]
    if len(client_hist) < lookback:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient history for {client_id}: need {lookback}h, have {len(client_hist)}h",
        )

    hist = client_hist.tail(lookback).copy()
    last_row = hist.iloc[-1]
    last_time_idx = int(last_row["time_idx"])

    future_calendar = _generate_future_calendar(
        int(last_row["hour"]), int(last_row["day_of_week"]), int(last_row["month"]), horizon
    )
    future_rows = [
        {
            "client_id": client_id,
            "time_idx": last_time_idx + h,
            "load_kwh": 0.0,
            **future_calendar[h - 1],
        }
        for h in range(1, horizon + 1)
    ]
    infer_df = pd.concat([hist, pd.DataFrame(future_rows)], ignore_index=True)

    infer_dataset = TimeSeriesDataSet.from_dataset(
        _state["training_schema"], infer_df, predict=True, stop_randomization=True
    )
    dataloader = infer_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    preds = _state["model"].predict(dataloader, mode="quantiles", trainer_kwargs={"logger": False})
    forecast_values = preds[0].tolist()

    return QuantileForecast(
        client_id=client_id,
        horizon_hours=horizon,
        quantiles=cfg["tft"]["quantiles"],
        forecast=forecast_values,
    )
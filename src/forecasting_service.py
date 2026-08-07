"""
Shared serving logic used by both src/api.py and src/streamlit_app.py.

Centralizing this here means the checkpoint-loading fix (CPU/GPU device
mismatch -- see comments below) and the future-calendar reconstruction logic
each live in exactly one place, rather than being duplicated and risking
drifting out of sync or reintroducing the same bug in only one of the two
serving surfaces.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import yaml
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer


def _generate_future_calendar(last_hour: int, last_dow: int, last_month: int, horizon: int):
    """Simple wraparound continuation of hour/day_of_week/is_weekend.
    Does NOT handle month-boundary edge cases correctly -- acceptable
    simplification for a demo, not true calendar arithmetic."""
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


def load_serving_artifacts(config_path: str = "config.yaml") -> dict:
    """Loads everything needed to serve forecasts: config, the training
    dataset schema (for reconstructing compatible inference inputs), the
    trained model, and the small demo client sample."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    schema_path = Path("models/training_dataset.pkl")
    if not schema_path.exists():
        raise RuntimeError(
            "models/training_dataset.pkl not found -- run "
            "`python -m src.export_serving_artifacts` first."
        )
    training_schema = torch.load(schema_path, weights_only=False)

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

    for module in model.modules():
        if hasattr(module, "_device"):
            module._device = torch.device("cpu")

    demo_path = Path("demo/sample_data.parquet")
    if not demo_path.exists():
        raise RuntimeError(
            "demo/sample_data.parquet not found -- run "
            "`python -m src.export_serving_artifacts` first."
        )
    demo_data = pd.read_parquet(demo_path)

    return {
        "cfg": cfg,
        "training_schema": training_schema,
        "model": model,
        "demo_data": demo_data,
        "checkpoint_path": checkpoint_path,
    }


def generate_forecast(client_id: str, artifacts: dict) -> dict:
    """Returns a dict with the client's recent history (for plotting) and
    the quantile forecast for the next horizon_hours."""
    demo_data = artifacts["demo_data"]
    cfg = artifacts["cfg"]

    client_hist = demo_data[demo_data.client_id == client_id].sort_values("time_idx")
    if client_hist.empty:
        raise ValueError(f"Unknown client_id: {client_id}")

    lookback = cfg["data"]["lookback_hours"]
    horizon = cfg["data"]["horizon_hours"]
    if len(client_hist) < lookback:
        raise ValueError(
            f"Insufficient history for {client_id}: need {lookback}h, have {len(client_hist)}h"
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
        artifacts["training_schema"], infer_df, predict=True, stop_randomization=True
    )
    dataloader = infer_dataset.to_dataloader(train=False, batch_size=1, num_workers=0)

    preds = artifacts["model"].predict(dataloader, mode="quantiles", trainer_kwargs={"logger": False})
    forecast_values = preds[0].tolist()

    return {
        "client_id": client_id,
        "horizon_hours": horizon,
        "quantiles": cfg["tft"]["quantiles"],
        "forecast": forecast_values,
        "history": hist,
    }
"""
Run this ONCE after TFT training finishes, before starting the API.

Exports two lightweight serving artifacts:
1. models/training_dataset.pkl -- the TimeSeriesDataSet schema (encoder/
   decoder length, feature definitions, per-client GroupNormalizer stats)
   needed to build inference-ready inputs WITHOUT reloading the full
   8M+ row processed dataset at request time.
2. demo/sample_data.parquet -- the most recent `lookback` hours of history
   for a handful of representative clients, small enough to commit directly
   to the repo (unlike the full dataset, which stays gitignored).

Usage:
    python -m src.export_serving_artifacts --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.tft_model import load_full_df, build_training_dataset

N_DEMO_CLIENTS = 8


def export_training_schema(cfg: dict, full_df: pd.DataFrame, train_df: pd.DataFrame):
    training = build_training_dataset(cfg, full_df, train_df)
    Path("models").mkdir(exist_ok=True)
    training.save("models/training_dataset.pkl")
    print("Saved models/training_dataset.pkl")


def export_demo_sample(cfg: dict, full_df: pd.DataFrame, n_clients: int = N_DEMO_CLIENTS):
    lookback = cfg["data"]["lookback_hours"]
    all_clients = sorted(full_df["client_id"].unique())

    step = max(1, len(all_clients) // n_clients)
    chosen = all_clients[::step][:n_clients]

    rows = []
    for client_id in chosen:
        client_hist = full_df[full_df.client_id == client_id].sort_values("time_idx").tail(lookback)
        rows.append(client_hist)

    demo_df = pd.concat(rows, ignore_index=True)
    Path("demo").mkdir(exist_ok=True)
    demo_df.to_parquet("demo/sample_data.parquet", index=False)
    print(f"Saved demo/sample_data.parquet ({len(chosen)} clients, {lookback}h history each)")
    print(f"Demo clients: {chosen}")


def main(cfg: dict):
    full_df, train_df, val_df, test_df = load_full_df(cfg)
    export_training_schema(cfg, full_df, train_df)
    export_demo_sample(cfg, full_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
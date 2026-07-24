"""
Trains and evaluates the seasonal naive and LightGBM baselines on the
processed electricity data, and prints/saves comparison metrics.

Usage:
    python src/evaluate_baselines.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.baselines import SeasonalNaive, LightGBMDirectForecaster, make_lag_features
from src.metrics import summarize


def evaluate_seasonal_naive(test_df: pd.DataFrame, full_df: pd.DataFrame, horizon: int) -> dict:
    """For each client, forecast the last `horizon` hours of the test set using
    the seasonal naive rule, pulling the needed lookback from full_df (since a
    client's test window alone may not contain a full week of history)."""
    model = SeasonalNaive()
    all_preds, all_targets = [], []

    for client_id, client_test in test_df.groupby("client_id"):
        client_full = full_df[full_df["client_id"] == client_id].sort_values("time_idx")
        client_test = client_test.sort_values("time_idx")

        # take the last `horizon` rows of test as the target window
        target_window = client_test.tail(horizon)
        if len(target_window) < horizon:
            continue

        # history = everything in client_full up to (not including) the target window start
        target_start_idx = target_window["time_idx"].iloc[0]
        history = client_full[client_full["time_idx"] < target_start_idx]["load_kwh"]

        try:
            pred = model.predict(history, horizon=horizon)
        except ValueError:
            continue  # not enough history for this client, skip

        all_preds.append(pred)
        all_targets.append(target_window["load_kwh"].values)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return summarize(all_targets, all_preds)


def evaluate_lightgbm(train_df: pd.DataFrame, test_df: pd.DataFrame, full_df: pd.DataFrame,
                       horizon: int, lgb_cfg: dict) -> dict:
    model = LightGBMDirectForecaster(
        horizon=horizon,
        lgb_params={
            "objective": "regression",
            "metric": "mae",
            "num_leaves": lgb_cfg["num_leaves"],
            "learning_rate": lgb_cfg["learning_rate"],
            "feature_fraction": lgb_cfg["feature_fraction"],
            "bagging_fraction": lgb_cfg["bagging_fraction"],
            "bagging_freq": lgb_cfg["bagging_freq"],
            "verbose": -1,
        },
    )
    print("Fitting LightGBM (one model per horizon step, this may take a few minutes)...")
    model.fit(train_df, num_boost_round=lgb_cfg["num_boost_round"])

    # build lag features using full history so test-set rows have valid lags
    full_with_lags = make_lag_features(full_df)
    test_with_lags = full_with_lags[full_with_lags["client_id"].isin(test_df["client_id"].unique())]
    test_with_lags = test_with_lags[test_with_lags["time_idx"].isin(test_df["time_idx"])]

    preds_df = model.predict(test_with_lags)

    # align actual targets: for each row, the actual value h steps ahead
    all_preds, all_targets = [], []
    sorted_full = full_df.sort_values(["client_id", "time_idx"])
    for client_id, client_rows in test_with_lags.groupby("client_id"):
        client_full = sorted_full[sorted_full["client_id"] == client_id].reset_index(drop=True)
        client_full_idx = client_full.set_index("time_idx")["load_kwh"]

        for row_idx, row in client_rows.iterrows():
            t = row["time_idx"]
            target_idxs = [t + h for h in range(1, horizon + 1)]
            if not all(ti in client_full_idx.index for ti in target_idxs):
                continue
            targets = client_full_idx.loc[target_idxs].values
            preds = preds_df.loc[row_idx].values if row_idx in preds_df.index else None
            if preds is None or len(preds) != horizon:
                continue
            all_preds.append(preds)
            all_targets.append(targets)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return summarize(all_targets, all_preds)


def main(cfg: dict):
    data_dir = Path(cfg["data"]["processed_dir"])
    horizon = cfg["data"]["horizon_hours"]

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    print("Evaluating seasonal naive baseline...")
    naive_metrics = evaluate_seasonal_naive(test_df, full_df, horizon)
    print("Seasonal naive:", naive_metrics)

    print("\nEvaluating LightGBM baseline...")
    lgb_metrics = evaluate_lightgbm(train_df, test_df, full_df, horizon, cfg["lightgbm"])
    print("LightGBM:", lgb_metrics)

    results = {"seasonal_naive": naive_metrics, "lightgbm": lgb_metrics}
    Path("reports").mkdir(exist_ok=True)
    with open("reports/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved results to reports/baseline_results.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg)
"""
Trains and evaluates the seasonal naive and LightGBM baselines on the
processed electricity data, and prints/saves comparison metrics.

Usage:
    python -m src.evaluate_baselines --config config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from src.baselines import SeasonalNaive, LightGBMDirectForecaster, make_lag_features
from src.metrics import summarize


def evaluate_seasonal_naive(test_df: pd.DataFrame, full_df: pd.DataFrame, horizon: int) -> dict:
    model = SeasonalNaive()
    all_preds, all_targets = [], []

    client_ids = test_df["client_id"].unique()
    for client_id in tqdm(client_ids, desc="Evaluating seasonal naive"):
        client_test = test_df[test_df["client_id"] == client_id].sort_values("time_idx")
        client_full = full_df[full_df["client_id"] == client_id].sort_values("time_idx")

        target_window = client_test.tail(horizon)
        if len(target_window) < horizon:
            continue

        target_start_idx = target_window["time_idx"].iloc[0]
        history = client_full[client_full["time_idx"] < target_start_idx]["load_kwh"]

        try:
            pred = model.predict(history, horizon=horizon)
        except ValueError:
            continue

        all_preds.append(pred)
        all_targets.append(target_window["load_kwh"].values)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    return summarize(all_targets, all_preds)


def evaluate_lightgbm(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                       full_df: pd.DataFrame, horizon: int, lgb_cfg: dict) -> dict:
    model = LightGBMDirectForecaster(
        horizon=horizon,
        lgb_params={
            "objective": "regression",
            "metric": "mae",
            "num_leaves": lgb_cfg["num_leaves"],
            "min_data_in_leaf": lgb_cfg["min_data_in_leaf"],
            "lambda_l1": lgb_cfg["lambda_l1"],
            "lambda_l2": lgb_cfg["lambda_l2"],
            "learning_rate": lgb_cfg["learning_rate"],
            "feature_fraction": lgb_cfg["feature_fraction"],
            "bagging_fraction": lgb_cfg["bagging_fraction"],
            "bagging_freq": lgb_cfg["bagging_freq"],
            "verbose": -1,
        },
    )
    print("Fitting LightGBM on residuals with early stopping (one model per horizon step)...")
    model.fit(
        train_df,
        val_df,
        num_boost_round=lgb_cfg["num_boost_round"],
        early_stopping_rounds=lgb_cfg["early_stopping_rounds"],
    )

    full_with_lags = make_lag_features(full_df)

    print("Generating predictions (naive baseline + predicted residual)...")
    preds_df = model.predict(full_with_lags)

    test_with_lags = full_with_lags[full_with_lags["client_id"].isin(test_df["client_id"].unique())]
    test_with_lags = test_with_lags[test_with_lags["time_idx"].isin(test_df["time_idx"])]

    all_preds, all_targets = [], []
    sorted_full = full_df.sort_values(["client_id", "time_idx"])

    grouped_test = list(test_with_lags.groupby("client_id"))
    for client_id, client_rows in tqdm(grouped_test, desc="Aligning LightGBM predictions"):
        client_full = sorted_full[sorted_full["client_id"] == client_id].reset_index(drop=True)
        client_full_idx = client_full.set_index("time_idx")["load_kwh"]

        for row_idx, row in client_rows.iterrows():
            t = row["time_idx"]
            target_idxs = [t + h for h in range(1, horizon + 1)]
            if not all(ti in client_full_idx.index for ti in target_idxs):
                continue
            targets = client_full_idx.loc[target_idxs].values
            if row_idx not in preds_df.index:
                continue
            preds = preds_df.loc[row_idx].values
            if len(preds) != horizon or pd.isna(preds).any():
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

    print("\nEvaluating LightGBM baseline (residual-based, regularized + early stopping)...")
    lgb_metrics = evaluate_lightgbm(train_df, val_df, test_df, full_df, horizon, cfg["lightgbm"])
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
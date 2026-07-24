"""
Baseline forecasters, evaluated before the TFT so its lift is quantified
against something a reviewer will actually trust.

1. SeasonalNaive       -- forecast(t+h) = actual(t+h-168)  (same hour, 1 week ago)
2. LightGBMForecaster  -- gradient-boosted trees on engineered lag/calendar
                          features, one-step-ahead recursive OR direct
                          multi-horizon (we use direct: one model per horizon)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb

SEASONAL_LAG_HOURS = 168  # one week, at hourly resolution
LOOKBACK_HOURS = 168
HORIZON_HOURS = 24
LAG_FEATURES = [1, 2, 3, 24, 48, 168]  # hours


class SeasonalNaive:
    """Forecast = value from exactly one week (168h) prior. No training."""

    def predict(self, history: pd.Series, horizon: int = HORIZON_HOURS) -> np.ndarray:
        if len(history) < SEASONAL_LAG_HOURS:
            raise ValueError("history shorter than one seasonal period (168h)")
        return history.values[-SEASONAL_LAG_HOURS: -SEASONAL_LAG_HOURS + horizon]


def make_lag_features(long_df: pd.DataFrame, lags: list[int] = LAG_FEATURES) -> pd.DataFrame:
    """Adds lag_{h} columns per client_id, sorted by time_idx."""
    df = long_df.sort_values(["client_id", "time_idx"]).copy()
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("client_id")["load_kwh"].shift(lag)
    df["rolling_mean_24"] = (
        df.groupby("client_id")["load_kwh"].shift(1).rolling(24).mean().values
    )
    df["rolling_std_168"] = (
        df.groupby("client_id")["load_kwh"].shift(1).rolling(168).std().values
    )
    return df


class LightGBMDirectForecaster:
    """One LightGBM model per horizon step (direct multi-horizon strategy).

    Direct (rather than recursive) forecasting avoids compounding one-step
    errors across the 24h horizon, and is the fairer comparison against TFT,
    which natively predicts all horizon steps at once.
    """

    FEATURE_COLS = [f"lag_{l}" for l in LAG_FEATURES] + [
        "rolling_mean_24",
        "rolling_std_168",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
    ]

    def __init__(self, horizon: int = HORIZON_HOURS, lgb_params: dict | None = None):
        self.horizon = horizon
        self.lgb_params = lgb_params or {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 63,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.models: dict[int, lgb.Booster] = {}

    def _build_horizon_target(self, df: pd.DataFrame, h: int) -> pd.Series:
        return df.groupby("client_id")["load_kwh"].shift(-h)

    def fit(self, train_df: pd.DataFrame, num_boost_round: int = 500):
        df = make_lag_features(train_df)
        for h in range(1, self.horizon + 1):
            target = self._build_horizon_target(df, h)
            data = df.assign(target=target).dropna(subset=self.FEATURE_COLS + ["target"])
            train_set = lgb.Dataset(data[self.FEATURE_COLS], label=data["target"])
            self.models[h] = lgb.train(
                self.lgb_params, train_set, num_boost_round=num_boost_round
            )
        return self

    def predict(self, eval_df: pd.DataFrame) -> pd.DataFrame:
        """eval_df must already contain lag features (call make_lag_features first
        if predicting on raw data). Returns a dataframe of predictions per horizon."""
        preds = {}
        for h, model in self.models.items():
            valid = eval_df.dropna(subset=self.FEATURE_COLS)
            preds[h] = pd.Series(
                model.predict(valid[self.FEATURE_COLS]), index=valid.index
            )
        return pd.DataFrame(preds)

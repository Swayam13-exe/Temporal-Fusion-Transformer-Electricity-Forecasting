"""
Baseline forecasters, evaluated before the TFT so its lift is quantified
against something a reviewer will actually trust.

1. SeasonalNaive       -- forecast(t+h) = actual(t+h-168)  (same hour, 1 week ago)
2. LightGBMForecaster  -- gradient-boosted trees, direct multi-horizon,
                          trained on the RESIDUAL against seasonal naive.
                          Uses a held-out validation split with early
                          stopping: the residual target has low signal-to-
                          noise (most week-to-week deviation is close to
                          random), so a fixed large num_boost_round easily
                          overfits noise on train and produces occasional
                          large, costly misses on test (visible as inflated
                          RMSE despite good MAE). Early stopping against a
                          genuine held-out split lets the model use more
                          trees where they help, but halts before it starts
                          memorizing noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from pandas.api.types import CategoricalDtype
from tqdm import tqdm

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
    """Adds lag_{h} columns per client_id, sorted by time_idx. Preserves any
    other columns already present (e.g. a '_split' marker column)."""
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
    """One LightGBM model per horizon step (direct multi-horizon strategy),
    trained on the residual against seasonal naive, with early stopping
    against a held-out validation split. client_id is included as a
    categorical feature so the model can still specialize per client.
    """

    FEATURE_COLS = [f"lag_{l}" for l in LAG_FEATURES] + [
        "rolling_mean_24",
        "rolling_std_168",
        "hour",
        "day_of_week",
        "month",
        "is_weekend",
        "client_id_encoded",
    ]

    def __init__(self, horizon: int = HORIZON_HOURS, lgb_params: dict | None = None):
        self.horizon = horizon
        self.lgb_params = lgb_params or {
            "objective": "regression",
            "metric": "mae",
            "num_leaves": 31,
            "min_data_in_leaf": 50,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        self.models: dict[int, lgb.Booster] = {}
        self.client_categories: list | None = None  # fixed at fit() time

    def _client_dtype(self) -> CategoricalDtype:
        if self.client_categories is None:
            raise RuntimeError("client_categories not set -- call fit() first")
        return CategoricalDtype(categories=self.client_categories)

    def _encode_clients(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["client_id_encoded"] = df["client_id"].astype(self._client_dtype()).cat.codes
        return df

    def _naive_shift(self, df: pd.DataFrame, h: int) -> pd.Series:
        """Seasonal-naive forecast value for horizon h: the actual load from
        exactly 168 hours before the target time (t+h). At row t, that's
        168-h rows back."""
        return df.groupby("client_id")["load_kwh"].shift(SEASONAL_LAG_HOURS - h)

    def _build_horizon_residual_target(self, df: pd.DataFrame, h: int) -> pd.Series:
        future_actual = df.groupby("client_id")["load_kwh"].shift(-h)
        naive_h = self._naive_shift(df, h)
        return future_actual - naive_h

    def fit(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_boost_round: int = 2000,
        early_stopping_rounds: int = 50,
    ):
        """train_df and val_df must be chronologically contiguous (val
        immediately after train) so lag features for early val rows can see
        into the end of train -- this is exactly your existing train/val
        split from data_pipeline.py, no changes needed there.
        """
        self.client_categories = sorted(train_df["client_id"].unique())

        combined = pd.concat(
            [train_df.assign(_split="train"), val_df.assign(_split="val")],
            ignore_index=True,
        )
        df = make_lag_features(combined)  # _split column passes through untouched
        df = self._encode_clients(df)

        for h in tqdm(range(1, self.horizon + 1), desc="Training LightGBM w/ early stopping (per horizon)"):
            target = self._build_horizon_residual_target(df, h)
            data = df.assign(target=target).dropna(subset=self.FEATURE_COLS + ["target"])

            train_rows = data[data["_split"] == "train"]
            val_rows = data[data["_split"] == "val"]

            train_set = lgb.Dataset(
                train_rows[self.FEATURE_COLS],
                label=train_rows["target"],
                categorical_feature=["client_id_encoded"],
            )
            val_set = lgb.Dataset(
                val_rows[self.FEATURE_COLS],
                label=val_rows["target"],
                reference=train_set,
                categorical_feature=["client_id_encoded"],
            )

            self.models[h] = lgb.train(
                self.lgb_params,
                train_set,
                num_boost_round=num_boost_round,
                valid_sets=[val_set],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
        return self

    def predict(self, eval_df: pd.DataFrame, return_absolute: bool = True) -> pd.DataFrame:
        """eval_df should contain CONTINUOUS history (not pre-filtered to only
        the evaluation window) so the 168-hour naive lookback is always
        available -- pass the full dataset here, then filter the returned
        predictions down to your evaluation window afterward.
        """
        eval_df = self._encode_clients(eval_df)
        preds = {}
        for h, model in tqdm(self.models.items(), desc="Predicting per horizon step"):
            valid = eval_df.dropna(subset=self.FEATURE_COLS)
            residual_pred = pd.Series(
                model.predict(valid[self.FEATURE_COLS], num_iteration=model.best_iteration),
                index=valid.index,
            )
            if return_absolute:
                naive_h = self._naive_shift(eval_df, h).loc[valid.index]
                combined = (residual_pred + naive_h).dropna()
                preds[h] = combined
            else:
                preds[h] = residual_pred
        return pd.DataFrame(preds)
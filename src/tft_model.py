"""
Temporal Fusion Transformer training pipeline via pytorch-forecasting.

Builds on the same processed parquet data as the other models, but uses
pytorch-forecasting's TimeSeriesDataSet to handle windowing, per-client
normalization (via GroupNormalizer, replacing the manual z-score approach
used in lstm_baseline.py -- TFT does this internally), and quantile loss
natively, so it produces p10/p25/p50/p75/p90 forecasts directly rather than
a single point estimate.

Usage:
    python -m src.tft_model --config config.yaml --mode train
    python -m src.tft_model --config config.yaml --mode evaluate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import QuantileLoss

from src.metrics import summarize, pinball_loss, quantile_coverage

CALENDAR_KNOWN_REALS = ["hour", "day_of_week", "month", "is_weekend"]


def load_full_df(cfg: dict):
    data_dir = Path(cfg["data"]["processed_dir"])
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_df["client_id"] = full_df["client_id"].astype(str)
    return full_df, train_df, val_df, test_df


def build_training_dataset(cfg: dict, full_df: pd.DataFrame, train_df: pd.DataFrame) -> TimeSeriesDataSet:
    lookback = cfg["data"]["lookback_hours"]
    horizon = cfg["data"]["horizon_hours"]
    training_cutoff = train_df["time_idx"].max()

    return TimeSeriesDataSet(
        full_df[full_df.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="load_kwh",
        group_ids=["client_id"],
        min_encoder_length=lookback,
        max_encoder_length=lookback,
        min_prediction_length=horizon,
        max_prediction_length=horizon,
        static_categoricals=["client_id"],
        time_varying_known_reals=CALENDAR_KNOWN_REALS,
        time_varying_unknown_reals=["load_kwh"],
        target_normalizer=GroupNormalizer(groups=["client_id"], transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        allow_missing_timesteps=True,
    )


def build_eval_dataset(training: TimeSeriesDataSet, full_df: pd.DataFrame,
                        min_prediction_idx: int | None = None) -> TimeSeriesDataSet:
    return TimeSeriesDataSet.from_dataset(
        training,
        full_df,
        predict=False,
        stop_randomization=True,
        min_prediction_idx=min_prediction_idx,
    )


def train(cfg: dict):
    torch.set_float32_matmul_precision("medium")
    full_df, train_df, val_df, test_df = load_full_df(cfg)
    tft_cfg = cfg["tft"]

    training = build_training_dataset(cfg, full_df, train_df)

    training_cutoff = train_df["time_idx"].max()
    validation_cutoff = val_df["time_idx"].max()

    val_full = full_df[full_df.time_idx <= validation_cutoff]
    validation = build_eval_dataset(training, val_full, min_prediction_idx=training_cutoff + 1)

    num_workers = tft_cfg.get("num_workers", 0)
    batch_size = tft_cfg["batch_size"]

    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=num_workers)

    print(f"Training dataset: {len(training):,} windows  |  Validation dataset: {len(validation):,} windows")

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=tft_cfg["learning_rate"],
        hidden_size=tft_cfg["hidden_size"],
        attention_head_size=tft_cfg["attention_head_size"],
        dropout=tft_cfg["dropout"],
        hidden_continuous_size=tft_cfg["hidden_continuous_size"],
        loss=QuantileLoss(quantiles=tft_cfg["quantiles"]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    print(f"Model has {sum(p.numel() for p in tft.parameters()):,} parameters")

    Path("models").mkdir(exist_ok=True)
    early_stop = EarlyStopping(
        monitor="val_loss", patience=tft_cfg.get("early_stopping_patience", 5), mode="min"
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath="models", filename="tft_best", monitor="val_loss", mode="min", save_top_k=1
    )

    trainer = pl.Trainer(
        max_epochs=tft_cfg["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=tft_cfg.get("gradient_clip_val", 0.1),
        callbacks=[early_stop, checkpoint_cb],
        limit_train_batches=tft_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=tft_cfg.get("limit_val_batches", 1.0),
        enable_progress_bar=True,
    )

    trainer.fit(tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    print(f"Training complete. Best checkpoint: {checkpoint_cb.best_model_path}")
    return checkpoint_cb.best_model_path


def evaluate(cfg: dict, checkpoint_path: str | None = None):
    full_df, train_df, val_df, test_df = load_full_df(cfg)
    tft_cfg = cfg["tft"]

    training = build_training_dataset(cfg, full_df, train_df)
    validation_cutoff = val_df["time_idx"].max()

    test_dataset = build_eval_dataset(training, full_df, min_prediction_idx=validation_cutoff + 1)
    print(f"Test dataset (all valid windows): {len(test_dataset):,}")

    # Cap evaluation size -- predicting over the full dense window set can
    # exhaust system RAM on a laptop (this is what likely caused the crash).
    # A random subset of this size is still a statistically solid sample for
    # MAE/RMSE/SMAPE/pinball/coverage estimates.
    max_eval_windows = tft_cfg.get("max_eval_windows", 50000)
    sampler = None
    if len(test_dataset) > max_eval_windows:
        sampler = torch.utils.data.RandomSampler(
            test_dataset, replacement=False, num_samples=max_eval_windows
        )
        print(f"Sampling down to {max_eval_windows:,} windows for evaluation")

    test_dataloader = test_dataset.to_dataloader(
        train=False,
        batch_size=tft_cfg["batch_size"],
        num_workers=tft_cfg.get("num_workers", 0),
        sampler=sampler,
    )

    # Correct checkpoint selection: most recently MODIFIED file, not
    # alphabetically last -- "tft_best-v1.ckpt" sorts BEFORE "tft_best.ckpt"
    # alphabetically (the '-' vs '.' character), which would silently pick
    # your very first (barely-trained) checkpoint instead of the latest one.
    if checkpoint_path is None:
        ckpts = list(Path("models").glob("tft_best*.ckpt"))
        if not ckpts:
            raise FileNotFoundError("No checkpoint found in models/ -- run training first.")
        checkpoint_path = str(max(ckpts, key=lambda p: p.stat().st_mtime))
    print(f"Loading checkpoint: {checkpoint_path}")

    best_tft = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    best_tft.eval()

    predictions = best_tft.predict(test_dataloader, mode="quantiles", return_y=True)
    preds = predictions.output.cpu().numpy()
    actuals = predictions.y[0].cpu().numpy()
    del predictions, best_tft, test_dataloader
    torch.cuda.empty_cache()

    quantiles = tft_cfg["quantiles"]
    median_idx = quantiles.index(0.5)
    median_preds = preds[:, :, median_idx]

    point_metrics = summarize(actuals, median_preds)
    print("TFT median (p50) forecast metrics:", point_metrics)

    pinball_results = {
        f"pinball_{q}": pinball_loss(actuals, preds[:, :, i], q)
        for i, q in enumerate(quantiles)
    }
    print("Pinball losses per quantile:", pinball_results)

    lower_idx = quantiles.index(0.1)
    upper_idx = quantiles.index(0.9)
    coverage = quantile_coverage(actuals, preds[:, :, lower_idx], preds[:, :, upper_idx])
    print(f"p10-p90 empirical coverage: {coverage:.3f} (target ~0.80 if well-calibrated)")

    return point_metrics, pinball_results, coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        train(cfg)
    else:
        evaluate(cfg, checkpoint_path=args.checkpoint)
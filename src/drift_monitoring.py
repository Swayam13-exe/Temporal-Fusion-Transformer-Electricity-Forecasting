"""
Forecast-drift monitoring.

Deliberately distinct from the feature-level PSI/OOV-rate drift monitoring
used in earlier projects. Here, "drift" means: as calendar time moves
forward through the test period, does the model's FORECAST quality degrade?
Two independent signals are tracked, both of which only make sense for a
probabilistic forecaster like TFT:

1. Point-accuracy drift: rolling MAE/RMSE/SMAPE per week. If a future week's
   MAE exceeds the reference period's mean by more than a few standard
   deviations, that's a signal the model may need retraining.
2. Calibration drift: rolling p10-p90 empirical coverage per week. Even if
   point accuracy holds up, a probabilistic model can quietly become
   miscalibrated (e.g. bands too narrow/overconfident) without MAE alone
   revealing it -- this is a genuinely different failure mode that a point-
   estimate model (LightGBM, plain LSTM) can't even be checked for.

A third signal, target-distribution drift (PSI on load_kwh itself across
time buckets), reuses the PSI technique from earlier projects but applies it
to a different object (the forecast target over calendar time, not input
features at inference time).

Usage:
    python -m src.drift_monitoring --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

from src.tft_model import load_full_df, build_training_dataset, build_eval_dataset
from pytorch_forecasting import TemporalFusionTransformer

BUCKET_HOURS = 168  # weekly buckets


def get_latest_checkpoint(models_dir: str = "models") -> str:
    ckpts = list(Path(models_dir).glob("tft_best*.ckpt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint found in {models_dir}/ -- run training first.")
    return str(max(ckpts, key=lambda p: p.stat().st_mtime))


def run_predictions_with_metadata(cfg: dict, checkpoint_path: str | None = None,
                                   max_windows: int = 20000) -> pd.DataFrame:
    """Runs TFT predictions across a capped random sample of the test set,
    returning one row per sample with its time_idx, client_id, per-sample
    mean absolute error, p10-p90 coverage fraction, and pinball loss (p50) --
    everything needed to bucket by calendar time downstream."""
    full_df, train_df, val_df, test_df = load_full_df(cfg)
    tft_cfg = cfg["tft"]

    training = build_training_dataset(cfg, full_df, train_df)
    validation_cutoff = val_df["time_idx"].max()
    test_dataset = build_eval_dataset(training, full_df, min_prediction_idx=validation_cutoff + 1)

    sampler = None
    if len(test_dataset) > max_windows:
        sampler = torch.utils.data.RandomSampler(test_dataset, replacement=False, num_samples=max_windows)

    dataloader = test_dataset.to_dataloader(
        train=False, batch_size=tft_cfg["batch_size"],
        num_workers=tft_cfg.get("num_workers", 0), sampler=sampler,
    )

    if checkpoint_path is None:
        checkpoint_path = get_latest_checkpoint()
    print(f"Loading checkpoint: {checkpoint_path}")
    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
    model.eval()

    predictions = model.predict(
        dataloader, mode="quantiles", return_index=True, return_y=True,
        trainer_kwargs={"logger": False},
    )

    preds = predictions.output.cpu().numpy()   # (n, horizon, n_quantiles)
    actuals = predictions.y[0].cpu().numpy()    # (n, horizon)
    index_df = predictions.index.copy()          # time_idx, client_id per sample

    quantiles = tft_cfg["quantiles"]
    median_idx = quantiles.index(0.5)
    lower_idx = quantiles.index(0.1)
    upper_idx = quantiles.index(0.9)

    median_preds = preds[:, :, median_idx]
    lower_preds = preds[:, :, lower_idx]
    upper_preds = preds[:, :, upper_idx]

    index_df["mae"] = np.mean(np.abs(actuals - median_preds), axis=1)
    index_df["mean_actual"] = np.mean(actuals, axis=1)
    inside = (actuals >= lower_preds) & (actuals <= upper_preds)
    index_df["coverage_frac"] = np.mean(inside, axis=1)

    return index_df


def psi_continuous(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index for a continuous variable, bucketed by
    reference-derived quantile edges. >0.25 conventionally indicates
    significant drift, 0.1-0.25 moderate, <0.1 negligible."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, buckets + 1)))
    if len(edges) < 3:
        return 0.0

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def build_rolling_report(df: pd.DataFrame, bucket_hours: int = BUCKET_HOURS) -> pd.DataFrame:
    df = df.copy()
    min_time = df["time_idx"].min()
    df["week_bucket"] = (df["time_idx"] - min_time) // bucket_hours

    rows = []
    for bucket, g in df.groupby("week_bucket"):
        rows.append({
            "week_bucket": int(bucket),
            "n_samples": len(g),
            "mae": g["mae"].mean(),
            "coverage": g["coverage_frac"].mean(),
        })
    return pd.DataFrame(rows).sort_values("week_bucket").reset_index(drop=True)


def detect_drift(rolling_df: pd.DataFrame, reference_weeks: int = 2,
                  mae_threshold_std: float = 2.0, coverage_band: tuple = (0.70, 0.90)) -> pd.DataFrame:
    """Uses the earliest `reference_weeks` buckets (closest to the training
    period, presumed healthy) as the reference baseline. Flags any later
    week whose MAE exceeds reference_mean + mae_threshold_std * reference_std,
    or whose coverage falls outside coverage_band."""
    reference = rolling_df.iloc[:reference_weeks]
    ref_mae_mean = reference["mae"].mean()
    ref_mae_std = reference["mae"].std(ddof=0) or 1e-6

    rolling_df = rolling_df.copy()
    rolling_df["mae_drift_flag"] = rolling_df["mae"] > (ref_mae_mean + mae_threshold_std * ref_mae_std)
    rolling_df["coverage_drift_flag"] = ~rolling_df["coverage"].between(*coverage_band)
    rolling_df["reference_mae_mean"] = ref_mae_mean
    rolling_df["reference_mae_std"] = ref_mae_std
    return rolling_df


def plot_drift_report(rolling_df: pd.DataFrame, out_dir: str = "reports/figures"):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rolling_df["week_bucket"], rolling_df["mae"], marker="o", label="Weekly MAE")
    ref_mean = rolling_df["reference_mae_mean"].iloc[0]
    ref_std = rolling_df["reference_mae_std"].iloc[0]
    ax.axhline(ref_mean, color="gray", linestyle="--", label="Reference mean")
    ax.axhline(ref_mean + 2 * ref_std, color="red", linestyle=":", label="Drift threshold (+2 std)")
    flagged = rolling_df[rolling_df["mae_drift_flag"]]
    if len(flagged):
        ax.scatter(flagged["week_bucket"], flagged["mae"], color="red", zorder=5, label="Flagged")
    ax.set_xlabel("Week (test period)")
    ax.set_ylabel("MAE (kWh)")
    ax.set_title("Forecast Point-Accuracy Drift")
    ax.legend()
    fig.savefig(f"{out_dir}/drift_mae_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(rolling_df["week_bucket"], rolling_df["coverage"], marker="o", label="Weekly p10-p90 coverage")
    ax.axhline(0.80, color="gray", linestyle="--", label="Target (0.80)")
    ax.axhspan(0.70, 0.90, color="gray", alpha=0.1, label="Acceptable band")
    flagged = rolling_df[rolling_df["coverage_drift_flag"]]
    if len(flagged):
        ax.scatter(flagged["week_bucket"], flagged["coverage"], color="red", zorder=5, label="Flagged")
    ax.set_xlabel("Week (test period)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Forecast Calibration Drift")
    ax.legend()
    fig.savefig(f"{out_dir}/drift_coverage_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved drift plots to {out_dir}/")


def main(cfg: dict, checkpoint_path: str | None = None, max_windows: int = 20000):
    df = run_predictions_with_metadata(cfg, checkpoint_path, max_windows)
    print(f"Collected {len(df):,} prediction samples across the test period")

    rolling_df = build_rolling_report(df)
    rolling_df = detect_drift(rolling_df)

    print("\nWeekly drift report:")
    print(rolling_df.to_string(index=False))

    first_bucket = df[df["time_idx"] < df["time_idx"].min() + BUCKET_HOURS * 2]["mean_actual"].values
    last_bucket = df[df["time_idx"] >= df["time_idx"].max() - BUCKET_HOURS * 2]["mean_actual"].values
    psi_score = psi_continuous(first_bucket, last_bucket)
    ks_stat, ks_pvalue = ks_2samp(first_bucket, last_bucket)
    print(f"\nTarget distribution drift (first 2 weeks vs. last 2 weeks of test period):")
    print(f"  PSI: {psi_score:.4f}  ({'>0.25 significant' if psi_score > 0.25 else '>0.1 moderate' if psi_score > 0.1 else 'negligible'})")
    print(f"  KS test: statistic={ks_stat:.4f}, p-value={ks_pvalue:.4f} "
          f"({'significant shift' if ks_pvalue < 0.05 else 'no significant shift'})")

    plot_drift_report(rolling_df)

    Path("reports").mkdir(exist_ok=True)
    rolling_df.to_csv("reports/drift_weekly_report.csv", index=False)
    print("\nSaved reports/drift_weekly_report.csv")

    n_flagged = rolling_df["mae_drift_flag"].sum() + rolling_df["coverage_drift_flag"].sum()
    if n_flagged == 0:
        print("\nNo drift flagged across the test period -- model remains stable.")
    else:
        print(f"\n{n_flagged} week(s) flagged for point-accuracy or calibration drift -- see plots for detail.")

    return rolling_df, psi_score, ks_pvalue


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--max_windows", type=int, default=20000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    main(cfg, checkpoint_path=args.checkpoint, max_windows=args.max_windows)
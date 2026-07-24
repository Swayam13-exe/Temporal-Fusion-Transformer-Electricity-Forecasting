"""
Data pipeline for the UCI Electricity Load Diagrams 2011-2014 dataset.

Source: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
(also mirrored on Kaggle as "Electricity Load Diagrams 2011-2014")

Raw file format:
    - Semicolon-delimited, comma as decimal separator
    - First column: timestamp (15-min resolution)
    - Remaining 370 columns: MT_001 ... MT_370, one per client, in kW
    - Values before a client's first non-zero reading are padding (client not
      yet connected) and must be dropped, not imputed as zero.

This script:
    1. Loads the raw file
    2. Resamples 15-min -> hourly (sum -> kWh)
    3. Drops leading all-zero padding per client
    4. Filters to clients with a complete history over the analysis window
    5. Reshapes into long format: [client_id, timestamp, load, hour, dow, month, is_holiday]
    6. Writes processed parquet files (train/val/test, chronological split)

Usage:
    python src/data_pipeline.py --raw_path data/raw/LD2011_2014.txt
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Analysis window: use the last two full years, where most clients have
# complete, non-padded data (following the windowing used in the original
# TFT paper's electricity benchmark).
ANALYSIS_START = "2012-01-01"
ANALYSIS_END = "2014-12-31 23:00:00"

TRAIN_END = "2014-06-30 23:00:00"
VAL_END = "2014-09-30 23:00:00"
# test: everything after VAL_END through ANALYSIS_END

MIN_NONZERO_FRACTION = 0.95  # drop clients with too much missing/zero history


def load_raw(raw_path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        raw_path,
        sep=";",
        decimal=",",
        index_col=0,
        parse_dates=True,
    )
    df.index.name = "timestamp"
    return df


def resample_hourly(df_15min: pd.DataFrame) -> pd.DataFrame:
    # 15-min kW readings summed over 4 intervals -> hourly kWh
    return df_15min.resample("1h").sum()


def select_clients(df_hourly: pd.DataFrame) -> list[str]:
    window = df_hourly.loc[ANALYSIS_START:ANALYSIS_END]
    nonzero_frac = (window != 0).mean(axis=0)
    keep = nonzero_frac[nonzero_frac >= MIN_NONZERO_FRACTION].index.tolist()
    return keep


def to_long_format(df_hourly: pd.DataFrame, clients: list[str]) -> pd.DataFrame:
    window = df_hourly.loc[ANALYSIS_START:ANALYSIS_END, clients]
    long_df = window.reset_index().melt(
        id_vars="timestamp", var_name="client_id", value_name="load_kwh"
    )
    long_df["hour"] = long_df["timestamp"].dt.hour
    long_df["day_of_week"] = long_df["timestamp"].dt.dayofweek
    long_df["month"] = long_df["timestamp"].dt.month
    long_df["is_weekend"] = (long_df["day_of_week"] >= 5).astype(int)

    # time_idx: integer index required by pytorch-forecasting's TimeSeriesDataSet
    long_df = long_df.sort_values(["client_id", "timestamp"])
    long_df["time_idx"] = (
        long_df.groupby("client_id").cumcount()
    )
    return long_df.reset_index(drop=True)


def chronological_split(long_df: pd.DataFrame):
    train = long_df[long_df["timestamp"] <= TRAIN_END]
    val = long_df[(long_df["timestamp"] > TRAIN_END) & (long_df["timestamp"] <= VAL_END)]
    test = long_df[long_df["timestamp"] > VAL_END]
    return train, val, test


def main(raw_path: str, out_dir: str):
    raw_path = Path(raw_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw file from {raw_path} ...")
    df_15min = load_raw(raw_path)

    print("Resampling 15-min -> hourly ...")
    df_hourly = resample_hourly(df_15min)

    print("Selecting clients with sufficiently complete history ...")
    clients = select_clients(df_hourly)
    print(f"  kept {len(clients)} / {df_hourly.shape[1]} clients")

    print("Reshaping to long format ...")
    long_df = to_long_format(df_hourly, clients)

    print("Splitting chronologically into train/val/test ...")
    train, val, test = chronological_split(long_df)
    print(f"  train: {train.shape}, val: {val.shape}, test: {test.shape}")

    train.to_parquet(out_dir / "train.parquet", index=False)
    val.to_parquet(out_dir / "val.parquet", index=False)
    test.to_parquet(out_dir / "test.parquet", index=False)
    print(f"Wrote processed parquet files to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", type=str, default="data/raw/LD2011_2014.txt")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    args = parser.parse_args()
    main(args.raw_path, args.out_dir)

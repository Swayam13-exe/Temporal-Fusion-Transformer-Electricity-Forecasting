"""
LSTM sequence-to-sequence baseline: deep learning WITHOUT attention.

Adds per-client z-score normalization (mean/std computed from TRAINING data
only, per client) before feeding load values into the network, and
de-normalizes predictions back to kWh before computing metrics. Without this,
314 clients with wildly different consumption scales feeding raw kWh values
into one shared network causes gradients from high-magnitude clients to
dominate, producing large/unstable loss and poor generalization -- the same
underlying issue that hurt the LightGBM baseline before its residual-target
fix. Per-series normalization is standard practice for shared neural
forecasters across heterogeneous series (used internally by DeepAR, N-BEATS,
and TFT).

Usage:
    python -m src.lstm_baseline --config config.yaml --mode train
    python -m src.lstm_baseline --config config.yaml --mode evaluate
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.metrics import summarize

CALENDAR_COLS = ["hour", "day_of_week", "month", "is_weekend"]
MIN_STD = 1e-3  # floor to avoid divide-by-zero for near-constant clients


def compute_client_stats(train_df: pd.DataFrame) -> dict:
    """Per-client (mean, std) of load_kwh, computed from TRAINING data only
    (never val/test) to avoid leakage. Returns {client_id: (mean, std)}."""
    agg = train_df.groupby("client_id")["load_kwh"].agg(["mean", "std"])
    return {
        cid: (float(row["mean"]), float(max(row["std"], MIN_STD)))
        for cid, row in agg.iterrows()
    }


def build_stats_arrays(client_to_idx: dict, client_stats: dict):
    """Returns (mean_arr, std_arr) float32 numpy arrays indexed by the same
    integer client index used everywhere else, for fast vectorized denorm."""
    n = len(client_to_idx)
    mean_arr = np.zeros(n, dtype=np.float32)
    std_arr = np.ones(n, dtype=np.float32)
    for cid, idx in client_to_idx.items():
        mean_arr[idx], std_arr[idx] = client_stats[cid]
    return mean_arr, std_arr


class WindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, lookback: int, horizon: int, client_to_idx: dict,
                 client_stats: dict, stride: int = 6):
        self.lookback = lookback
        self.horizon = horizon
        self.samples = []

        df = df.sort_values(["client_id", "time_idx"])
        for client_id, g in tqdm(df.groupby("client_id"), desc="Building windows"):
            g = g.reset_index(drop=True)
            mean, std = client_stats[client_id]
            load_norm = (g["load_kwh"].values.astype(np.float32) - mean) / std
            calendar = g[CALENDAR_COLS].values.astype(np.float32)
            n = len(g)
            client_idx = client_to_idx[client_id]

            for start in range(0, n - lookback - horizon + 1, stride):
                enc_end = start + lookback
                dec_end = enc_end + horizon
                self.samples.append(
                    (
                        load_norm[start:enc_end],
                        calendar[start:enc_end],
                        calendar[enc_end:dec_end],
                        load_norm[enc_end:dec_end],
                        client_idx,
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        enc_load, enc_cal, dec_cal, target, client_idx = self.samples[idx]
        return (
            torch.tensor(enc_load).unsqueeze(-1),
            torch.tensor(enc_cal),
            torch.tensor(dec_cal),
            torch.tensor(target),
            torch.tensor(client_idx, dtype=torch.long),
        )


class Seq2SeqLSTM(nn.Module):
    def __init__(self, n_clients, n_calendar_features, hidden_size=64, num_layers=2,
                 dropout=0.2, client_embedding_dim=8):
        super().__init__()
        self.client_embedding = nn.Embedding(n_clients, client_embedding_dim)
        enc_input_dim = 1 + n_calendar_features + client_embedding_dim
        dec_input_dim = 1 + n_calendar_features + client_embedding_dim

        self.encoder = nn.LSTM(enc_input_dim, hidden_size, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.decoder = nn.LSTM(dec_input_dim, hidden_size, num_layers, batch_first=True,
                                dropout=dropout if num_layers > 1 else 0.0)
        self.output_proj = nn.Linear(hidden_size, 1)

    def forward(self, enc_load, enc_cal, dec_cal, client_idx, horizon, teacher_force_target=None):
        client_emb = self.client_embedding(client_idx)
        client_emb_enc = client_emb.unsqueeze(1).expand(-1, enc_load.size(1), -1)

        enc_input = torch.cat([enc_load, enc_cal, client_emb_enc], dim=-1)
        _, (h, c) = self.encoder(enc_input)

        prev_load = enc_load[:, -1, :]
        outputs = []

        for t in range(horizon):
            dec_cal_t = dec_cal[:, t, :]
            dec_input = torch.cat([prev_load, dec_cal_t, client_emb], dim=-1).unsqueeze(1)
            out, (h, c) = self.decoder(dec_input, (h, c))
            pred = self.output_proj(out.squeeze(1))
            outputs.append(pred)

            if teacher_force_target is not None:
                prev_load = teacher_force_target[:, t].unsqueeze(-1)
            else:
                prev_load = pred

        return torch.cat(outputs, dim=1)


def build_client_index(df: pd.DataFrame) -> dict:
    clients = sorted(df["client_id"].unique())
    return {c: i for i, c in enumerate(clients)}


def train(cfg: dict):
    data_dir = Path(cfg["data"]["processed_dir"])
    lookback = cfg["data"]["lookback_hours"]
    horizon = cfg["data"]["horizon_hours"]
    lstm_cfg = cfg["lstm"]

    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    client_to_idx = build_client_index(train_df)

    # stats computed from TRAIN ONLY, then reused for val (and later test) --
    # this is the correct way to normalize: val/test never influence the
    # scale the model is trained against.
    client_stats = compute_client_stats(train_df)
    mean_arr, std_arr = build_stats_arrays(client_to_idx, client_stats)

    stride = lstm_cfg.get("window_stride", 6)
    train_ds = WindowDataset(train_df, lookback, horizon, client_to_idx, client_stats, stride=stride)
    val_ds = WindowDataset(val_df, lookback, horizon, client_to_idx, client_stats, stride=stride)
    print(f"Train windows: {len(train_ds):,}  Val windows: {len(val_ds):,}  (stride={stride}h)")

    train_loader = DataLoader(train_ds, batch_size=lstm_cfg["batch_size"], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=lstm_cfg["batch_size"], shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = Seq2SeqLSTM(
        n_clients=len(client_to_idx),
        n_calendar_features=len(CALENDAR_COLS),
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        client_embedding_dim=lstm_cfg["client_embedding_dim"],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lstm_cfg["learning_rate"])
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    Path("models").mkdir(exist_ok=True)

    for epoch in range(lstm_cfg["max_epochs"]):
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{lstm_cfg['max_epochs']} [train]")
        for enc_load, enc_cal, dec_cal, target, client_idx in train_bar:
            enc_load, enc_cal = enc_load.to(device), enc_cal.to(device)
            dec_cal, target = dec_cal.to(device), target.to(device)
            client_idx = client_idx.to(device)

            optimizer.zero_grad()
            preds = model(enc_load, enc_cal, dec_cal, client_idx, horizon, teacher_force_target=target)
            loss = criterion(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * enc_load.size(0)
            train_bar.set_postfix(batch_loss=f"{loss.item():.4f}")
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{lstm_cfg['max_epochs']} [val]")
        with torch.no_grad():
            for enc_load, enc_cal, dec_cal, target, client_idx in val_bar:
                enc_load, enc_cal = enc_load.to(device), enc_cal.to(device)
                dec_cal, target = dec_cal.to(device), target.to(device)
                client_idx = client_idx.to(device)
                preds = model(enc_load, enc_cal, dec_cal, client_idx, horizon)
                batch_loss = criterion(preds, target).item()
                val_loss += batch_loss * enc_load.size(0)
                val_bar.set_postfix(batch_loss=f"{batch_loss:.4f}")
        val_loss /= len(val_ds)

        # loss here is on NORMALIZED scale (roughly unit variance per client),
        # so values near/below ~1.0 are the expected healthy range -- NOT
        # comparable to the raw-kWh losses from before this fix.
        print(f"Epoch {epoch+1}/{lstm_cfg['max_epochs']}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "client_to_idx": client_to_idx,
                    "mean_arr": mean_arr,
                    "std_arr": std_arr,
                    "config": lstm_cfg,
                },
                "models/lstm_seq2seq_best.pt",
            )
            print(f"  -> new best model saved (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= lstm_cfg["early_stopping_patience"]:
                print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
                break

    print("Training complete. Best model saved to models/lstm_seq2seq_best.pt")


def evaluate(cfg: dict, checkpoint_path: str = "models/lstm_seq2seq_best.pt"):
    data_dir = Path(cfg["data"]["processed_dir"])
    lookback = cfg["data"]["lookback_hours"]
    horizon = cfg["data"]["horizon_hours"]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    client_to_idx = checkpoint["client_to_idx"]
    mean_arr = checkpoint["mean_arr"]
    std_arr = checkpoint["std_arr"]
    lstm_cfg = checkpoint["config"]

    # rebuild the same {client_id: (mean, std)} dict the dataset expects,
    # from the arrays saved in the checkpoint (train-derived, leak-free)
    idx_to_client = {idx: cid for cid, idx in client_to_idx.items()}
    client_stats = {
        idx_to_client[i]: (float(mean_arr[i]), float(std_arr[i])) for i in range(len(mean_arr))
    }

    test_df = pd.read_parquet(data_dir / "test.parquet")
    stride = lstm_cfg.get("window_stride", 6)
    test_ds = WindowDataset(test_df, lookback, horizon, client_to_idx, client_stats, stride=stride)
    print(f"Test windows: {len(test_ds):,} (stride={stride}h)")
    test_loader = DataLoader(test_ds, batch_size=lstm_cfg["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2SeqLSTM(
        n_clients=len(client_to_idx),
        n_calendar_features=len(CALENDAR_COLS),
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        client_embedding_dim=lstm_cfg["client_embedding_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_targets, all_client_idx = [], [], []
    with torch.no_grad():
        for enc_load, enc_cal, dec_cal, target, client_idx in tqdm(test_loader, desc="Evaluating on test set"):
            enc_load, enc_cal = enc_load.to(device), enc_cal.to(device)
            dec_cal = dec_cal.to(device)
            client_idx_dev = client_idx.to(device)
            preds = model(enc_load, enc_cal, dec_cal, client_idx_dev, horizon)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.numpy())
            all_client_idx.append(client_idx.numpy())

    all_preds = np.concatenate(all_preds, axis=0)      # normalized scale
    all_targets = np.concatenate(all_targets, axis=0)  # normalized scale
    all_client_idx = np.concatenate(all_client_idx, axis=0)

    # de-normalize back to kWh using each sample's own client stats, so
    # metrics are directly comparable to the seasonal-naive/LightGBM numbers
    sample_mean = mean_arr[all_client_idx].reshape(-1, 1)
    sample_std = std_arr[all_client_idx].reshape(-1, 1)
    preds_kwh = all_preds * sample_std + sample_mean
    targets_kwh = all_targets * sample_std + sample_mean

    metrics = summarize(targets_kwh, preds_kwh)
    print("LSTM seq2seq test set metrics (kWh scale):", metrics)
    return metrics, preds_kwh, targets_kwh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--mode", type=str, choices=["train", "evaluate"], default="train")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode == "train":
        train(cfg)
    else:
        evaluate(cfg)
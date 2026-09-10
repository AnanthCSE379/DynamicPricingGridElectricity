"""
src/pure_dataset.py
Pure Power Load Forecasting Dataset Pipeline.
Inputs strictly contain:
1. Past power consumption (scaled)
2. Cyclical calendar encodings (hour, day of week, month, weekend flag)
TOTAL FEATURES = 8. Zero price information is exposed to the model.
"""

import os
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib

TRAIN_END   = "2013-05-31 23:30:00"
VAL_END     = "2013-09-30 23:30:00"
LOOKBACK    = 48    # 24 hours
TARGET_COL  = "household_mean_kwh"
PURE_SCALER_PATH = "checkpoints/pure_scaler.pkl"

# Exactly 8 features - NO PRICE
PURE_FEATURE_COLS = [
    TARGET_COL,       # consumption
    "sin_hh", "cos_hh",
    "sin_dow", "cos_dow",
    "sin_month", "cos_month",
    "is_weekend",
]

def add_pure_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns([
        pl.col("DateTime").dt.hour().alias("hour"),
        pl.col("DateTime").dt.minute().alias("minute"),
        pl.col("DateTime").dt.weekday().alias("weekday"),   # 1=Mon ... 7=Sun
        pl.col("DateTime").dt.month().alias("month"),
    ])

    df = df.with_columns([
        (pl.col("hour") * 2 + pl.col("minute") // 30).alias("hh_idx"),
        (pl.col("weekday") >= 6).cast(pl.Float32).alias("is_weekend"),
    ])

    df = df.with_columns([
        (2 * np.pi * pl.col("hh_idx") / 48).sin().cast(pl.Float32).alias("sin_hh"),
        (2 * np.pi * pl.col("hh_idx") / 48).cos().cast(pl.Float32).alias("cos_hh"),
        (2 * np.pi * (pl.col("weekday") - 1) / 7).sin().cast(pl.Float32).alias("sin_dow"),
        (2 * np.pi * (pl.col("weekday") - 1) / 7).cos().cast(pl.Float32).alias("cos_dow"),
        (2 * np.pi * (pl.col("month") - 1) / 12).sin().cast(pl.Float32).alias("sin_month"),
        (2 * np.pi * (pl.col("month") - 1) / 12).cos().cast(pl.Float32).alias("cos_month"),
    ])
    return df

def load_pure_splits(parquet_path: str = "data/feeder_aggregate_halfhourly.parquet"):
    df = pl.read_parquet(parquet_path).sort("DateTime")
    df = add_pure_features(df)
    df = df.with_columns(pl.col(TARGET_COL).forward_fill())

    train_df = df.filter(pl.col("DateTime") <= pl.lit(TRAIN_END).str.to_datetime())
    val_df   = df.filter(
        (pl.col("DateTime") > pl.lit(TRAIN_END).str.to_datetime()) &
        (pl.col("DateTime") <= pl.lit(VAL_END).str.to_datetime())
    )
    test_df  = df.filter(pl.col("DateTime") > pl.lit(VAL_END).str.to_datetime())

    print(f"[PureDataset] Train: {train_df.height:,} | Val: {val_df.height:,} | Test: {test_df.height:,} rows")
    return train_df, val_df, test_df

def fit_pure_scaler(train_df: pl.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df.select(TARGET_COL).to_numpy())
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(scaler, PURE_SCALER_PATH)
    print(f"[PureDataset] Scaler saved to {PURE_SCALER_PATH}")
    return scaler

def apply_pure_scaler(df: pl.DataFrame, scaler: StandardScaler) -> np.ndarray:
    arr = df.select(PURE_FEATURE_COLS).to_numpy().astype(np.float32)
    arr[:, 0] = scaler.transform(arr[:, 0].reshape(-1, 1)).flatten()
    return arr

class PureWindowDataset(Dataset):
    def __init__(self, arr: np.ndarray, lookback: int = LOOKBACK):
        self.arr      = arr
        self.lookback = lookback
        self.n        = len(arr) - lookback

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.arr[idx : idx + self.lookback]             # [48, 8]
        y = self.arr[idx + self.lookback, 0:1]              # [1]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def build_pure_loaders(batch_size: int = 64, parquet_path: str = "data/feeder_aggregate_halfhourly.parquet"):
    train_df, val_df, test_df = load_pure_splits(parquet_path)
    scaler = fit_pure_scaler(train_df)

    train_arr = apply_pure_scaler(train_df, scaler)
    val_arr   = apply_pure_scaler(val_df, scaler)
    test_arr  = apply_pure_scaler(test_df, scaler)

    train_ds = PureWindowDataset(train_arr)
    val_ds   = PureWindowDataset(val_arr)
    test_ds  = PureWindowDataset(test_arr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    test_timestamps = test_df["DateTime"].to_list()[LOOKBACK:]
    print(f"[PureDataset] Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader, scaler, test_timestamps, test_arr

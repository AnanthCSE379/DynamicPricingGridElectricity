"""
src/dataset.py
LCL Smart Meter Dataset — Feature Engineering, Windowing, and PyTorch DataLoader.
Reads from feeder_aggregate_halfhourly.parquet (665 KB, 39,727 rows).
Fits scalers strictly on the training split to prevent data leakage.
"""

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os


# ─────────────────────────────────────────────────────────────────────────────
# Chronological Split Constants (no shuffling — time series integrity)
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_END   = "2013-05-31 23:30:00"
VAL_END     = "2013-09-30 23:30:00"
# Everything after VAL_END is Test (~Oct 2013 – Feb 2014)

LOOKBACK    = 48    # 24 hours of history
TARGET_COL  = "household_mean_kwh"
SCALER_PATH = "checkpoints/scaler.pkl"


def _add_cyclical_features(df: pl.DataFrame) -> pl.DataFrame:
    """
    Augment the feeder aggregate DataFrame with cyclical time encodings and
    tariff price signal.
    """
    df = df.with_columns([
        # Cast Tariff to string first to compare safely
        pl.col("Tariff").cast(pl.String).alias("Tariff_str")
    ])

    df = df.with_columns([
        pl.col("DateTime").dt.hour().alias("hour"),
        pl.col("DateTime").dt.minute().alias("minute"),
        pl.col("DateTime").dt.weekday().alias("weekday"),   # 1=Mon … 7=Sun
        pl.col("DateTime").dt.month().alias("month"),
    ])

    df = df.with_columns([
        # half-hour index: 0–47
        (pl.col("hour") * 2 + pl.col("minute") // 30).alias("hh_idx"),
        # weekend flag
        (pl.col("weekday") >= 6).cast(pl.Float32).alias("is_weekend"),
    ])

    df = df.with_columns([
        # Cyclical encodings
        (2 * np.pi * pl.col("hh_idx") / 48).sin().cast(pl.Float32).alias("sin_hh"),
        (2 * np.pi * pl.col("hh_idx") / 48).cos().cast(pl.Float32).alias("cos_hh"),
        (2 * np.pi * (pl.col("weekday") - 1) / 7).sin().cast(pl.Float32).alias("sin_dow"),
        (2 * np.pi * (pl.col("weekday") - 1) / 7).cos().cast(pl.Float32).alias("cos_dow"),
        (2 * np.pi * (pl.col("month") - 1) / 12).sin().cast(pl.Float32).alias("sin_month"),
        (2 * np.pi * (pl.col("month") - 1) / 12).cos().cast(pl.Float32).alias("cos_month"),
    ])

    # Tariff price signal (already in parquet as price_p_per_kwh; fill 11.76 if null)
    df = df.with_columns([
        pl.col("price_p_per_kwh").fill_null(11.76).cast(pl.Float32).alias("tariff_price"),
    ])

    return df


FEATURE_COLS = [
    TARGET_COL,       # consumption — to be scaled
    "sin_hh", "cos_hh",
    "sin_dow", "cos_dow",
    "sin_month", "cos_month",
    "is_weekend",
    "tariff_price",   # dynamic pricing signal
]


def load_and_split(parquet_path: str = "data/feeder_aggregate_halfhourly.parquet"):
    """
    Load the feeder aggregate parquet, engineer features, then
    perform a strict chronological 70/15/15 split.
    Returns (train_df, val_df, test_df) as polars DataFrames.
    """
    df = pl.read_parquet(parquet_path).sort("DateTime")
    df = _add_cyclical_features(df)

    # Forward-fill any residual nulls in consumption
    df = df.with_columns(pl.col(TARGET_COL).forward_fill())

    train_df = df.filter(pl.col("DateTime") <= pl.lit(TRAIN_END).str.to_datetime())
    val_df   = df.filter(
        (pl.col("DateTime") > pl.lit(TRAIN_END).str.to_datetime()) &
        (pl.col("DateTime") <= pl.lit(VAL_END).str.to_datetime())
    )
    test_df  = df.filter(pl.col("DateTime") > pl.lit(VAL_END).str.to_datetime())

    print(f"[Dataset] Train: {train_df.height:,} | Val: {val_df.height:,} | Test: {test_df.height:,} rows")
    return train_df, val_df, test_df


def fit_scaler(train_df: pl.DataFrame) -> StandardScaler:
    """
    Fit a StandardScaler on the TRAINING split only to prevent leakage.
    Saves scaler to disk for reproducibility.
    """
    scaler = StandardScaler()
    # Scale only the consumption column; other features are either cyclical or bounded
    scaler.fit(train_df.select(TARGET_COL).to_numpy())
    os.makedirs("checkpoints", exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[Dataset] Scaler fitted and saved to {SCALER_PATH}")
    return scaler


def apply_scaler(df: pl.DataFrame, scaler: StandardScaler) -> np.ndarray:
    """
    Returns a numpy array with the consumption column scaled and all other
    FEATURE_COLS unchanged.
    """
    arr = df.select(FEATURE_COLS).to_numpy().astype(np.float32)
    arr[:, 0] = scaler.transform(arr[:, 0].reshape(-1, 1)).flatten()
    return arr


class SlidingWindowDataset(Dataset):
    """
    PyTorch Dataset: generates (X, y) pairs using a causal sliding window.
    X shape: [LOOKBACK, num_features]
    y shape: [1]  (scaled next-step consumption)
    """
    def __init__(self, arr: np.ndarray, lookback: int = LOOKBACK):
        self.arr      = arr
        self.lookback = lookback
        self.n        = len(arr) - lookback

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.arr[idx : idx + self.lookback]             # [L, F]
        y = self.arr[idx + self.lookback, 0:1]              # [1] — scaled consumption
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def build_loaders(
    batch_size: int = 64,
    parquet_path: str = "data/feeder_aggregate_halfhourly.parquet",
    num_workers: int = 0,
):
    """
    Full pipeline: load → feature-engineer → split → fit scaler →
    apply scaler → create DataLoaders.
    Returns (train_loader, val_loader, test_loader, scaler, test_timestamps).
    """
    train_df, val_df, test_df = load_and_split(parquet_path)
    scaler   = fit_scaler(train_df)

    train_arr = apply_scaler(train_df, scaler)
    val_arr   = apply_scaler(val_df, scaler)
    test_arr  = apply_scaler(test_df, scaler)

    train_ds = SlidingWindowDataset(train_arr)
    val_ds   = SlidingWindowDataset(val_arr)
    test_ds  = SlidingWindowDataset(test_arr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Preserve test timestamps for plotting (shifted by LOOKBACK)
    test_timestamps = test_df["DateTime"].to_list()[LOOKBACK:]

    # Also preserve test tariff labels for segmented evaluation
    test_tariffs = test_df["Tariff_str"].to_list()[LOOKBACK:]

    print(f"[Dataset] Train batches: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")
    return train_loader, val_loader, test_loader, scaler, test_timestamps, test_tariffs, test_arr

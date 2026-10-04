"""Controlled, post-hoc interrogation of the trained ConsumerTwin GRU.

The deployment path in ``run_6way_benchmark.py`` changes the *final* tariff
value in the 48 half-hour input window before predicting the next half-hour.
This script reproduces that path exactly, then measures its price and lag
sensitivity on held-out data.  It does not retrain or alter the checkpoint.

Outputs are written to plots/customer_response/:
  - price_response_curves.png   conditional demand-vs-price curves
  - tariff_lag_sensitivity.png  effect of a 60p tariff at each history lag
  - response_summary.json       machine-readable curve and surrogate values
  - CUSTOMER_RESPONSE_MODEL.md  equations, definitions, and limitations
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import GRUModel

# CSV is deliberately used here because it is a portable read of the same
# feeder aggregate and avoids PyArrow's unsigned-dictionary pandas conversion.
DATA_PATH = ROOT / "data/feeder_aggregate_halfhourly.csv"
CHECKPOINT_PATH = ROOT / "checkpoints/GRU.pt"
SCALER_PATH = ROOT / "checkpoints/scaler.pkl"
OUT_DIR = ROOT / "plots/customer_response"

PRICES = np.array([5.0, 10.0, 14.23, 20.0, 28.0, 40.0, 60.0, 80.0], dtype=np.float32)
BASE_PRICE = 14.23
HIGH_PRICE = 60.0
LOOKBACK = 48
TARGET_COL = "household_mean_kwh"
MAX_WINDOWS = 1_200  # deterministic representative held-out sample for fast interrogation


def engineer_features(df):
    """Replicates the feature order in src/dataset.py without a Polars dependency."""
    df = df.copy()
    dt = pd.to_datetime(df["DateTime"])
    hh_idx = dt.dt.hour * 2 + dt.dt.minute // 30
    weekday = dt.dt.weekday + 1  # pandas Monday=0; project uses Monday=1
    month = dt.dt.month
    df[TARGET_COL] = df[TARGET_COL].ffill()
    return np.column_stack([
        df[TARGET_COL].to_numpy(dtype=np.float32),
        np.sin(2 * np.pi * hh_idx / 48), np.cos(2 * np.pi * hh_idx / 48),
        np.sin(2 * np.pi * (weekday - 1) / 7), np.cos(2 * np.pi * (weekday - 1) / 7),
        np.sin(2 * np.pi * (month - 1) / 12), np.cos(2 * np.pi * (month - 1) / 12),
        (weekday >= 6).astype(np.float32),
        df["price_p_per_kwh"].fillna(11.76).to_numpy(dtype=np.float32),
    ]).astype(np.float32)


def apply_scaler(arr, scaler):
    """Apply the saved consumption scaler, matching src.dataset.apply_scaler."""
    arr = arr.copy()
    arr[:, 0] = scaler.transform(arr[:, 0].reshape(-1, 1)).ravel()
    return arr


def predict_windows(model, scaler, windows, device, batch_size=512):
    """Predict unscaled kWh/hh for an [N, 48, 9] window tensor."""
    output = []
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            x = torch.from_numpy(windows[start:start + batch_size]).to(device)
            output.append(model(x).cpu().numpy().reshape(-1))
    scaled = np.concatenate(output)
    return scaler.inverse_transform(scaled.reshape(-1, 1)).reshape(-1)


def cohort_masks(timestamps):
    """Representative held-out operating contexts; masks are mutually readable."""
    month = np.array([t.month for t in timestamps])
    hour = np.array([t.hour + t.minute / 60 for t in timestamps])
    return {
        "All held-out intervals": np.ones(len(timestamps), dtype=bool),
        "Winter evening (Dec–Feb, 17:00–20:30)": np.isin(month, [12, 1, 2]) & (hour >= 17) & (hour <= 20.5),
        "Winter overnight (Dec–Feb, 00:00–05:30)": np.isin(month, [12, 1, 2]) & (hour < 6),
        "Autumn evening (Oct–Nov, 17:00–20:30)": np.isin(month, [10, 11]) & (hour >= 17) & (hour <= 20.5),
    }


def make_windows(arr):
    return np.stack([arr[i:i + LOOKBACK] for i in range(len(arr) - LOOKBACK)]).astype(np.float32)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # CPU is deterministic and avoids GPU startup overhead for this small audit.
    device = torch.device("cpu")

    df = pd.read_csv(DATA_PATH, parse_dates=["DateTime"]).sort_values("DateTime")
    test = df.loc[df["DateTime"] > pd.Timestamp("2013-09-30 23:30:00")].copy()
    scaler = joblib.load(SCALER_PATH)
    arr = apply_scaler(engineer_features(test), scaler)
    timestamps = test["DateTime"].iloc[LOOKBACK:].tolist()
    windows = make_windows(arr)

    if len(windows) > MAX_WINDOWS:
        rng = np.random.default_rng(20261004)
        chosen = np.sort(rng.choice(len(windows), size=MAX_WINDOWS, replace=False))
        windows = windows[chosen]
        timestamps = [timestamps[i] for i in chosen]

    model = GRUModel()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.to(device).eval()

    masks = cohort_masks(timestamps)
    curves = {}
    all_predictions = {}
    for price in PRICES:
        altered = windows.copy()
        # This is the exact intervention used by consumer_predict_one().
        altered[:, -1, -1] = price
        all_predictions[float(price)] = np.clip(predict_windows(model, scaler, altered, device), 0.01, None)

    for label, mask in masks.items():
        mean_curve = np.array([all_predictions[float(p)][mask].mean() for p in PRICES])
        # A local, directly usable second-order surrogate over the tested domain.
        c2, c1, c0 = np.polyfit(PRICES - BASE_PRICE, mean_curve, deg=2)
        p_lo, p_hi = 13.0, 15.0
        beta = (np.interp(p_hi, PRICES, mean_curve) - np.interp(p_lo, PRICES, mean_curve)) / (p_hi - p_lo)
        q0 = float(np.interp(BASE_PRICE, PRICES, mean_curve))
        elasticity = float(beta * BASE_PRICE / q0)
        curves[label] = {
            "n_intervals": int(mask.sum()),
            "prices_p_per_kwh": PRICES.tolist(),
            "mean_demand_kwh_per_hh": mean_curve.round(6).tolist(),
            "demand_at_14_23p_kwh_per_hh": round(q0, 6),
            "local_slope_kwh_per_hh_per_p": round(float(beta), 8),
            "local_price_elasticity": round(elasticity, 6),
            "quadratic_surrogate": {
                "intercept": round(float(c0), 8),
                "linear": round(float(c1), 8),
                "quadratic": round(float(c2), 10),
                "formula": "q(p) = intercept + linear*(p-14.23) + quadratic*(p-14.23)^2",
            },
        }

    # Lag test: replace the entire observed tariff history with the base price,
    # then inject a single 60p price signal at each of the 48 lags.
    reference = windows.copy()
    reference[:, :, -1] = BASE_PRICE
    reference_prediction = np.clip(predict_windows(model, scaler, reference, device), 0.01, None)
    winter_peak = masks["Winter evening (Dec–Feb, 17:00–20:30)"]
    lag_effect = []
    for lag in range(LOOKBACK):
        altered = reference.copy()
        altered[:, lag, -1] = HIGH_PRICE
        prediction = np.clip(predict_windows(model, scaler, altered, device), 0.01, None)
        # lag=47 is the price intervention used by the live controller.
        lag_effect.append(float((prediction[winter_peak] - reference_prediction[winter_peak]).mean()))

    summary = {
        "model": "GRUModel / ConsumerTwin checkpoint GRU.pt",
        "test_period": "2013-10-01 through 2014-02-28",
        "n_sampled_held_out_windows": len(windows),
        "lookback_half_hours": LOOKBACK,
        "deployment_intervention": "Replace tariff_price in final input row (lag 0.5 h before prediction).",
        "price_sweep": curves,
        "lag_test": {
            "context": "Winter evening (Dec–Feb, 17:00–20:30)",
            "baseline_price_p_per_kwh": BASE_PRICE,
            "intervention_price_p_per_kwh": HIGH_PRICE,
            "effect_kwh_per_hh_by_oldest_to_newest_lag": [round(v, 7) for v in lag_effect],
            "newest_lag_index": LOOKBACK - 1,
        },
    }
    (OUT_DIR / "response_summary.json").write_text(json.dumps(summary, indent=2))

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))
    for label, values in curves.items():
        if label == "All held-out intervals":
            style = {"color": "#222222", "linewidth": 2.8}
        else:
            style = {"linewidth": 2.0}
        ax.plot(PRICES, values["mean_demand_kwh_per_hh"], marker="o", label=label, **style)
    ax.axvline(BASE_PRICE, color="#777777", linestyle="--", linewidth=1, label="Historical flat tariff (14.23p)")
    ax.set(xlabel="Intervened tariff at final input step (p/kWh)", ylabel="Predicted household demand (kWh / half-hour)",
           title="CustomerTwin conditional response to the deployed tariff input")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "price_response_curves.png", dpi=220)
    plt.close(fig)

    hours_before = (LOOKBACK - 1 - np.arange(LOOKBACK)) / 2
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(hours_before, lag_effect, marker="o", color="#9b2226", linewidth=2)
    ax.axhline(0, color="#444444", linewidth=1)
    ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
    ax.set(xlabel="How long before the prediction was the 60p intervention? (hours)",
           ylabel="Change in predicted demand (kWh / half-hour)",
           title="Where the ConsumerTwin uses a single tariff shock: winter evening context")
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tariff_lag_sensitivity.png", dpi=220)
    plt.close(fig)

    md = rf"""# Customer demand-response model: empirical interrogation

## What is deployed

For each next half-hour \(t\), the ConsumerTwin predicts normalized demand from a 48-step (24-hour) history:

\[z_t=f_\theta(X_{{t-48:t-1}}), \qquad \hat q_t=\max(0.01,\mu+\sigma z_t).\]

Each row in \(X\) contains previous normalized consumption, sine/cosine encodings of half-hour, weekday and month, a weekend indicator, and a tariff.  The deployed controller only overwrites the final tariff entry, \(p_{{t-1}}\), before predicting \(q_t\).  Thus the test measures a **one-half-hour-lagged, conditional association**, not a contemporaneous structural demand curve.

## Usable local equations

For each context below, use the reported surrogate over the tested 5–80 p/kWh range:

\[\bar q(p)\approx a+b(p-14.23)+c(p-14.23)^2.\]

The JSON file is the authoritative machine-readable coefficient source.  The linear derivative at the historical tariff is \(b\), and the corresponding point elasticity is \(b\,14.23/\bar q(14.23)\).

## Interpretation limit

The GRU is a nonlinear predictive model trained on observational trial data.  Its response curve should be treated as a conditional simulator under the original data distribution—not proof that changing prices causes the same effect in the real world, particularly at 40–80 p/kWh.  For a portable causal equation, validate the curve against randomized/experimental price variation or fit a demand model with explicit identification assumptions.
"""
    (OUT_DIR / "CUSTOMER_RESPONSE_MODEL.md").write_text(md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

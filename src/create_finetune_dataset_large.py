"""
src/create_finetune_dataset_large.py
Builds a 15,000-sample stratified, peak-balanced dataset from the 2-year LCL archive.
Partitioning:
- 35% Peak Stress & Substation Congestion (load > 0.23 kWh/hh or wholesale > 15p)
- 25% Off-Peak Valley Excess (load < 0.16 kWh/hh and wholesale < 10p)
- 40% Standard Baseline Operations (0.16 <= load <= 0.23 kWh/hh)
Also generates a balanced 1,000-sample validation set from the validation split.
"""

import os
import sys
import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from src.dynamic_pricing_controller import DynamicPricingController
from src.grid_economics import wholesale_price_at, FEEDER_THRESHOLD

LOOKBACK = 12
ctrl = DynamicPricingController()

def load_data():
    parquet_path = os.path.join(ROOT_DIR, "data/feeder_aggregate_halfhourly.parquet")
    df = pd.read_parquet(parquet_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    return df

def generate_stratified_split(df_split, target_samples, split_name):
    print(f"\n[Dataset Builder] Constructing {split_name} split ({target_samples:,} samples)...", flush=True)
    
    loads = df_split["household_mean_kwh"].values
    dts   = df_split["DateTime"].tolist()
    n_rows = len(loads)

    peak_indices = []
    valley_indices = []
    baseline_indices = []

    for i in range(LOOKBACK, n_rows - 1):
        target_load = float(loads[i])
        dt = dts[i]
        wh = wholesale_price_at(dt)

        if target_load > 0.23 or wh > 15.0:
            peak_indices.append(i)
        elif target_load < 0.16 and wh < 10.0:
            valley_indices.append(i)
        else:
            baseline_indices.append(i)

    print(f"  Available in pool: {len(peak_indices):,} Peaks, {len(valley_indices):,} Valleys, {len(baseline_indices):,} Baselines", flush=True)

    # Calculate target counts
    n_peak = int(target_samples * 0.35)
    n_val  = int(target_samples * 0.25)
    n_base = target_samples - (n_peak + n_val)

    # Sample with replacement if needed, or sample without replacement
    sampled_peaks = random.sample(peak_indices, min(n_peak, len(peak_indices)))
    if len(sampled_peaks) < n_peak:
        sampled_peaks += random.choices(peak_indices, k=n_peak - len(sampled_peaks))

    sampled_valleys = random.sample(valley_indices, min(n_val, len(valley_indices)))
    if len(sampled_valleys) < n_val:
        sampled_valleys += random.choices(valley_indices, k=n_val - len(sampled_valleys))

    sampled_baselines = random.sample(baseline_indices, min(n_base, len(baseline_indices)))
    if len(sampled_baselines) < n_base:
        sampled_baselines += random.choices(baseline_indices, k=n_base - len(sampled_baselines))

    selected_indices = sampled_peaks + sampled_valleys + sampled_baselines
    random.seed(42)
    random.shuffle(selected_indices)

    system_prompt = (
        "You are an autonomous Grid Operator AI Agent for a UK power feeder. "
        "Forecast the next electricity load and set the optimal retail tariff. "
        "Return exactly a JSON object with 'reasoning', 'predicted_load', and 'tariff_p'."
    )

    records = []
    for idx in selected_indices:
        dt = dts[idx]
        wh = wholesale_price_at(dt)
        history = [round(float(v), 4) for v in loads[idx - LOOKBACK : idx]]
        target_load = float(loads[idx])

        # Ground truth dynamic tariff from math controller
        optimal_tariff = ctrl.compute_tariff(target_load)

        # Explicit contextual reasoning
        if target_load > FEEDER_THRESHOLD or optimal_tariff > 16.0:
            reason = f"Grid congestion imminent ({target_load:.3f} kWh/hh). Raising tariff to shave peak load and preserve retailer margin."
        elif optimal_tariff < 10.0:
            reason = f"Off-peak valley excess ({target_load:.3f} kWh/hh). Lowering tariff to 4p-8p to incentivize storage and EV charging."
        else:
            reason = f"Normal grid operating conditions ({target_load:.3f} kWh/hh). Maintaining stable baseline retail rate."

        assistant_target = json.dumps({
            "reasoning": reason,
            "predicted_load": round(target_load, 4),
            "tariff_p": round(optimal_tariff, 2)
        })

        user_prompt = (
            f"Time: {dt.strftime('%A %H:%M')} | Wholesale: {wh:.2f}p/kWh | Grid Stress Threshold: 0.25 kWh/hh\n"
            f"Past 12 half-hour loads (kWh/hh): {history}\n"
            f"Task 1 — Forecast next half-hour load from the trend.\n"
            f"Task 2 — Set optimal tariff between 3.99 and 67.20 pence/kWh."
        )

        row = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": assistant_target}
            ]
        }
        records.append(row)

    return records

def main():
    random.seed(42)
    df = load_data()

    # Split: Train up to 2013-05-31 (approx 26,670 rows), Val between 2013-06-01 and 2013-09-30
    train_mask = df["DateTime"] <= "2013-05-31 23:30:00"
    val_mask   = (df["DateTime"] > "2013-05-31 23:30:00") & (df["DateTime"] <= "2013-09-30 23:30:00")

    df_train = df[train_mask].reset_index(drop=True)
    df_val   = df[val_mask].reset_index(drop=True)

    train_records = generate_stratified_split(df_train, 15000, "Train")
    val_records   = generate_stratified_split(df_val, 1000, "Validation")

    train_path = os.path.join(ROOT_DIR, "data/train.jsonl")
    valid_path = os.path.join(ROOT_DIR, "data/valid.jsonl")

    print(f"\nWriting {len(train_records):,} records to {train_path}...", flush=True)
    with open(train_path, "w") as f:
        for r in train_records:
            f.write(json.dumps(r) + "\n")

    print(f"Writing {len(val_records):,} records to {valid_path}...", flush=True)
    with open(valid_path, "w") as f:
        for r in val_records:
            f.write(json.dumps(r) + "\n")

    print("\nDataset generation complete! Ready for overnight fine-tuning.", flush=True)

if __name__ == "__main__":
    main()

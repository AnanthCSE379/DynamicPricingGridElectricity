#!/usr/bin/env python3
"""
extract_std_sample.py
Streams CC_LCL-FullData.csv (8.54 GB) out-of-core without loading it into RAM.
Computes:
1. Aggregate half-hourly demand for Standard (Std) flat-rate customers
2. Extracts a sample of 50 high-continuity Std households for comparative research
"""

import os
import time
import polars as pl

def process_std(input_csv="CC_LCL-FullData.csv"):
    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        return

    os.makedirs("data", exist_ok=True)
    print(f"Starting out-of-core scan of {input_csv} for Standard (Std) customers...")
    t0 = time.time()

    lazy_std = (
        pl.scan_csv(input_csv)
        .filter(pl.col("stdorToU") == "Std")
        .select([
            pl.col("LCLid").cast(pl.Categorical),
            pl.col("DateTime").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S"),
            pl.col("KWH/hh (per half hour) ").str.strip_chars().cast(pl.Float32, strict=False).alias("kwh_hh")
        ])
        .filter(pl.col("kwh_hh").is_not_null())
    )

    print("Computing half-hourly Std cohort aggregate load...")
    std_agg = (
        lazy_std
        .group_by("DateTime")
        .agg([
            pl.col("kwh_hh").sum().alias("std_grid_total_kwh"),
            pl.col("kwh_hh").mean().alias("std_household_mean_kwh"),
            pl.col("kwh_hh").std().alias("std_household_std_kwh"),
            pl.len().alias("std_active_meters")
        ])
        .sort("DateTime")
        .collect(engine="streaming")
    )
    std_agg = std_agg.with_columns(pl.col("DateTime").cast(pl.Datetime("ms")))
    
    out_parquet = "data/std_feeder_aggregate_halfhourly.parquet"
    std_agg.write_parquet(out_parquet)
    print(f"Successfully processed and saved Std aggregate to {out_parquet} ({std_agg.height} time steps) in {time.time() - t0:.2f}s!")

if __name__ == "__main__":
    process_std()

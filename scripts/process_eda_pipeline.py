#!/usr/bin/env python3
"""
process_eda_pipeline.py
Memory-efficient preprocessing, aggregation, and EDA pipeline for the LCL ToU dataset.
Runs out-of-core on Polars, keeping peak RAM well under 1 GB on Apple Silicon M3 Air.
"""

import os
import time
import json
import polars as pl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_pipeline():
    start_time = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs("eda_plots", exist_ok=True)

    print("=" * 70)
    print("STEP 1: Scanning and Aggregating lcl_tou.csv (Half-Hourly Grid Load)")
    print("=" * 70)

    feeder_parquet = "data/feeder_aggregate_halfhourly.parquet"
    feeder_csv = "data/feeder_aggregate_halfhourly.csv"

    if os.path.exists(feeder_parquet):
        print(f"Found existing {feeder_parquet}, loading directly...")
        merged_grid = pl.read_parquet(feeder_parquet)
    else:
        # 1. Lazy Scan of lcl_tou.csv
        lazy_df = (
            pl.scan_csv("lcl_tou.csv")
            .select([
                pl.col("LCLid").cast(pl.Categorical),
                pl.col("DateTime").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S"),
                pl.col("KWH/hh (per half hour) ").str.strip_chars().cast(pl.Float32, strict=False).alias("kwh_hh")
            ])
            .filter(pl.col("kwh_hh").is_not_null())
        )

        # 2. Compute Half-Hourly Grid / Feeder Aggregate
        print("Computing half-hourly grid metrics (sum, mean, std, active meters)...")
        t0 = time.time()
        halfhourly_agg = (
            lazy_df
            .group_by("DateTime")
            .agg([
                pl.col("kwh_hh").sum().alias("grid_total_kwh"),
                pl.col("kwh_hh").mean().alias("household_mean_kwh"),
                pl.col("kwh_hh").std().alias("household_std_kwh"),
                pl.col("kwh_hh").quantile(0.50).alias("household_median_kwh"),
                pl.col("kwh_hh").quantile(0.95).alias("household_p95_kwh"),
                pl.len().alias("active_meters")
            ])
            .sort("DateTime")
            .collect(engine="streaming")
        )
        print(f"Aggregation complete in {time.time() - t0:.2f}s! Total half-hour intervals: {halfhourly_agg.height}")

        # 3. Join with Tariffs (2013 trial)
        print("\n" + "=" * 70)
        print("STEP 2: Merging with Dynamic Time-of-Use Tariffs")
        print("=" * 70)
        tariffs_df = pl.read_parquet("tariffs.parquet")
        
        # Cast DateTime to match
        halfhourly_agg = halfhourly_agg.with_columns(pl.col("DateTime").cast(pl.Datetime("ms")))
        
        merged_grid = halfhourly_agg.join(tariffs_df, on="DateTime", how="left")
        # For timestamps outside 2013, fill Tariff with 'Baseline' and price with 11.76 (standard flat rate)
        merged_grid = merged_grid.with_columns([
            pl.col("Tariff").cast(pl.String).fill_null("Baseline").cast(pl.Categorical),
            pl.col("price_p_per_kwh").fill_null(11.76)
        ])

        merged_grid.write_parquet(feeder_parquet)
        merged_grid.write_csv(feeder_csv)
        print(f"Saved feeder aggregate time-series:\n - {feeder_parquet}\n - {feeder_csv}")

    # 4. Extract Representative Household Sample (top 50 high-continuity meters)
    print("\n" + "=" * 70)
    print("STEP 3: Extracting Representative Sample of 50 Households")
    print("=" * 70)
    t0 = time.time()
    
    sample_parquet = "data/tou_household_samples.parquet"
    if os.path.exists(sample_parquet):
        print(f"Found existing {sample_parquet}, loading directly...")
    else:
        # Filter 2013 records for household continuity check
        top_households = (
            lazy_df
            .filter((pl.col("DateTime") >= pl.datetime(2013, 1, 1)) & (pl.col("DateTime") <= pl.datetime(2013, 12, 31, 23, 30)))
            .group_by("LCLid")
            .agg([
                pl.len().alias("record_count"),
                pl.col("kwh_hh").mean().alias("avg_kwh")
            ])
            .filter(pl.col("record_count") >= 17000) # Near complete 2013 records (max 17520)
            .sort("avg_kwh")
            .collect()
        )
        print(f"Found {top_households.height} households with >97% completeness in 2013.")
        
        # Pick 50 evenly spaced across consumption tiers (low, medium, high consumers)
        step = max(1, top_households.height // 50)
        selected_ids = top_households["LCLid"].to_list()[::step][:50]
        
        sample_df = (
            lazy_df
            .filter(pl.col("LCLid").is_in(selected_ids))
            .collect(engine="streaming")
        )
        sample_df.write_parquet(sample_parquet)
        print(f"Saved {sample_df.height} records for 50 representative households to {sample_parquet} in {time.time() - t0:.2f}s")

    # 5. Exploratory Data Analysis & Visualizations
    print("\n" + "=" * 70)
    print("STEP 4: Generating Exploratory Data Analysis (EDA) Visualizations")
    print("=" * 70)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Add temporal features to merged grid
    mg = merged_grid.with_columns([
        pl.col("DateTime").dt.hour().alias("hour"),
        pl.col("DateTime").dt.minute().alias("minute"),
        pl.col("DateTime").dt.weekday().alias("weekday"), # 1=Mon, 7=Sun
        pl.col("DateTime").dt.month().alias("month"),
        (pl.col("DateTime").dt.hour() * 2 + pl.col("DateTime").dt.minute() // 30).alias("half_hour_idx"),
        pl.when(pl.col("DateTime").dt.weekday() >= 6).then(pl.lit("Weekend")).otherwise(pl.lit("Weekday")).alias("day_type")
    ])

    # Convert to Pandas only for aggregate plotting (tiny ~35,000 rows, < 5 MB)
    pdf = mg.to_pandas()

    # --- Plot 1: Diurnal Load Profile ---
    print("Generating Plot 1: Diurnal Load Profile...")
    diurnal = pdf.groupby(["half_hour_idx", "day_type"])["household_mean_kwh"].agg(["mean", "std"]).reset_index()
    fig, ax = plt.subplots(figsize=(11, 5))
    for dt, group in diurnal.groupby("day_type"):
        ax.plot(group["half_hour_idx"], group["mean"], label=f"{dt} Mean", linewidth=2.2)
        ax.fill_between(group["half_hour_idx"], group["mean"] - 0.2*group["std"], group["mean"] + 0.2*group["std"], alpha=0.15)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Time of Day (Half-Hourly)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Consumption (kWh per half-hour)", fontsize=11, fontweight="bold")
    ax.set_title("London Smart Meters: Diurnal Household Demand Curve (Weekday vs Weekend)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig("eda_plots/01_diurnal_load_profile.png", dpi=300)
    plt.close(fig)

    # --- Plot 2: Seasonal Demand Profiles ---
    print("Generating Plot 2: Seasonal Patterns...")
    def get_season(m):
        if m in [12, 1, 2]: return "Winter"
        elif m in [3, 4, 5]: return "Spring"
        elif m in [6, 7, 8]: return "Summer"
        else: return "Autumn"
    pdf["season"] = pdf["month"].apply(get_season)
    seasonal_diurnal = pdf.groupby(["half_hour_idx", "season"])["household_mean_kwh"].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(11, 5))
    season_colors = {"Winter": "#1f77b4", "Spring": "#2ca02c", "Summer": "#ff7f0e", "Autumn": "#d62728"}
    for s in ["Winter", "Spring", "Summer", "Autumn"]:
        if s in seasonal_diurnal.columns:
            ax.plot(seasonal_diurnal.index, seasonal_diurnal[s], label=s, color=season_colors[s], linewidth=2.2)
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Time of Day (Half-Hourly)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Consumption (kWh/hh)", fontsize=11, fontweight="bold")
    ax.set_title("Seasonal Diurnal Demand Variation", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig("eda_plots/02_weekly_seasonal_patterns.png", dpi=300)
    plt.close(fig)

    # --- Plot 3: Dynamic Tariff Distribution & Trigger Hours ---
    print("Generating Plot 3: Dynamic Tariff Breakdown & Trigger Patterns...")
    trial_2013 = pdf[pdf["DateTime"].between("2013-01-01", "2013-12-31 23:30:00")].copy()
    
    # Dynamic Tariff Breakdown
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    tariff_counts = trial_2013["Tariff"].value_counts()
    colors = {"Normal": "#3498db", "Low": "#2ecc71", "High": "#e74c3c", "Baseline": "#95a5a6"}
    wedge_colors = [colors.get(k, "#bdc3c7") for k in tariff_counts.index]
    explode = [0.04] * len(tariff_counts)
    ax1.pie(tariff_counts, labels=tariff_counts.index, autopct="%1.1f%%", colors=wedge_colors, startangle=140,
            explode=explode, textprops={"fontsize": 10, "fontweight": "bold"})
    ax1.set_title("2013 Dynamic Tariff Event Share", fontsize=12, fontweight="bold")

    # High tariff trigger frequency by hour of day
    high_events = trial_2013[trial_2013["Tariff"] == "High"]
    hourly_high = high_events.groupby("hour").size()
    ax2.bar(hourly_high.index, hourly_high.values, color="#e74c3c", edgecolor="black", alpha=0.85)
    ax2.set_xlabel("Hour of Day (24h)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Number of 'High' Tariff Periods", fontsize=11, fontweight="bold")
    ax2.set_title("Grid Stress: When are 'High' Tariffs Triggered?", fontsize=12, fontweight="bold")
    ax2.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig("eda_plots/03_tariff_distribution_and_pricing.png", dpi=300)
    plt.close(fig)

    # --- Plot 4: Demand Response: High vs Normal vs Low Consumption ---
    print("Generating Plot 4: Demand Response Impact (High vs Normal vs Low)...")
    tariff_impact = trial_2013.groupby(["half_hour_idx", "Tariff"], observed=False)["household_mean_kwh"].mean().unstack()
    
    fig, ax = plt.subplots(figsize=(11, 5))
    if "Normal" in tariff_impact.columns:
        ax.plot(tariff_impact.index, tariff_impact["Normal"], label="Normal Tariff (11.76 p/kWh)", color="#3498db", linewidth=2.5)
    if "High" in tariff_impact.columns:
        ax.plot(tariff_impact.index, tariff_impact["High"], label="High Tariff (67.20 p/kWh)", color="#e74c3c", linewidth=2.5, linestyle="--")
    if "Low" in tariff_impact.columns:
        ax.plot(tariff_impact.index, tariff_impact["Low"], label="Low Tariff (3.99 p/kWh)", color="#2ecc71", linewidth=2.5, linestyle=":")
    
    ax.set_xticks(range(0, 48, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)])
    ax.set_xlabel("Time of Day (Half-Hourly)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Average Consumption (kWh/hh)", fontsize=11, fontweight="bold")
    ax.set_title("Consumer Demand Response to Dynamic Pricing Signals (2013 Trial)", fontsize=13, fontweight="bold", pad=12)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig("eda_plots/04_demand_response_high_vs_normal.png", dpi=300)
    plt.close(fig)

    # --- Plot 5: Aggregated 2-Year Feeder Load Time-Series ---
    print("Generating Plot 5: 2-Year Feeder Load Timeline...")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(pdf["DateTime"], pdf["grid_total_kwh"], color="#2c3e50", linewidth=0.6, alpha=0.85)
    ax.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax.set_ylabel("Total Grid Demand (kWh/hh)", fontsize=11, fontweight="bold")
    ax.set_title("Aggregated London Feeder Demand Timeline (Nov 2011 - Feb 2014)", fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig("eda_plots/05_feeder_timeline_2year.png", dpi=300)
    plt.close(fig)

    # 6. Output Summary Metrics JSON
    print("\n" + "=" * 70)
    print("STEP 5: Writing Summary Metrics JSON")
    print("=" * 70)
    
    # Calculate price elasticity / response
    overall_mean_normal = float(trial_2013[trial_2013["Tariff"] == "Normal"]["household_mean_kwh"].mean())
    overall_mean_high = float(trial_2013[trial_2013["Tariff"] == "High"]["household_mean_kwh"].mean())
    overall_mean_low = float(trial_2013[trial_2013["Tariff"] == "Low"]["household_mean_kwh"].mean())
    
    # High tariff peak hours (typically 16:00 to 20:00)
    peak_slice = trial_2013[(trial_2013["hour"] >= 16) & (trial_2013["hour"] <= 20)]
    peak_normal = float(peak_slice[peak_slice["Tariff"] == "Normal"]["household_mean_kwh"].mean())
    peak_high = float(peak_slice[peak_slice["Tariff"] == "High"]["household_mean_kwh"].mean())
    peak_curtailment_pct = ((peak_normal - peak_high) / peak_normal) * 100 if peak_normal > 0 else 0.0

    summary_metrics = {
        "dataset_name": "Low Carbon London (LCL) - ToU Smart Meter Cohort",
        "total_raw_rows": 33783771,
        "total_households": 1123,
        "date_range": {
            "start": str(pdf["DateTime"].min()),
            "end": str(pdf["DateTime"].max())
        },
        "halfhourly_grid_intervals": int(merged_grid.height),
        "consumption_kwh_per_hh": {
            "mean": round(float(pdf["household_mean_kwh"].mean()), 4),
            "median": round(float(pdf["household_mean_kwh"].median()), 4),
            "std": round(float(pdf["household_mean_kwh"].std()), 4),
            "min": round(float(pdf["household_mean_kwh"].min()), 4),
            "max": round(float(pdf["household_mean_kwh"].max()), 4)
        },
        "tariff_events_2013": {
            "normal_periods": int(tariff_counts.get("Normal", 0)),
            "high_periods": int(tariff_counts.get("High", 0)),
            "low_periods": int(tariff_counts.get("Low", 0))
        },
        "demand_response_metrics": {
            "overall_avg_kwh_normal": round(overall_mean_normal, 4),
            "overall_avg_kwh_high": round(overall_mean_high, 4),
            "overall_avg_kwh_low": round(overall_mean_low, 4),
            "evening_peak_normal_kwh": round(peak_normal, 4),
            "evening_peak_high_kwh": round(peak_high, 4),
            "evening_peak_demand_reduction_percent": round(peak_curtailment_pct, 2)
        },
        "execution_time_seconds": round(time.time() - start_time, 2)
    }

    with open("data/eda_summary_metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Pipeline executed successfully in {time.time() - start_time:.2f} seconds!")
    print(json.dumps(summary_metrics, indent=2))

if __name__ == "__main__":
    run_pipeline()

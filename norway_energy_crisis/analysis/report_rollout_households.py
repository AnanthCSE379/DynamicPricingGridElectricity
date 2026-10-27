"""Household-level summaries and plots for the explicit sliding-window audit."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from norway_energy_crisis.clean_data import ROOT, require, save_json


def report(output):
    output = Path(output)
    summary = json.loads((output / "summary.json").read_text())
    actual = np.load(output / "actual.npz")["demand"]
    ids = np.asarray(summary["household_ids"])
    require(
        np.array_equal(ids, np.load(ROOT / "cleaned/household_ids.npy")), "ID order."
    )
    primary = np.load(ROOT / "cleaned/cohort_mask.npy")
    areas = np.load(ROOT / "cleaned/area_index.npy")
    state = json.loads(
        (ROOT / "runs/response_refinement_v2/prepared/preprocessing.json").read_text()
    )
    scales = np.asarray(state["demand_scales"])
    rows = []
    for engine in ("price_aware", "price_free"):
        arrays = np.load(output / engine / "recorded_paths.npz")
        for mode in ("recursive", "measurement_refreshed"):
            q = arrays[mode]
            error = q.astype(np.float64) - actual
            rmse = np.sqrt((error**2).mean((0, 2)))
            bias = error.mean((0, 2))
            day_rmse = [
                np.sqrt((error[:, :, d * 24 : (d + 1) * 24] ** 2).mean((0, 2)))
                for d in range(3)
            ]
            predicted_mean = [
                q[:, :, d * 24 : (d + 1) * 24].mean((0, 2)) for d in range(3)
            ]
            measured_mean = [
                actual[:, :, d * 24 : (d + 1) * 24].mean((0, 2)) for d in range(3)
            ]
            drift = (predicted_mean[2] - predicted_mean[0]) - (
                measured_mean[2] - measured_mean[0]
            )
            for h, household in enumerate(ids):
                rows.append(
                    {
                        "household_id": int(household),
                        "primary_spot_household": bool(primary[h]),
                        "region": ("NO1", "NO5")[areas[h]],
                        "engine": engine,
                        "feedback": mode,
                        "rmse_72h_kwh": float(rmse[h]),
                        "bias_kwh_per_hour": float(bias[h]),
                        "day1_rmse_kwh": float(day_rmse[0][h]),
                        "day2_rmse_kwh": float(day_rmse[1][h]),
                        "day3_rmse_kwh": float(day_rmse[2][h]),
                        "day1_predicted_mean": float(predicted_mean[0][h]),
                        "day3_predicted_mean": float(predicted_mean[2][h]),
                        "day1_actual_mean": float(measured_mean[0][h]),
                        "day3_actual_mean": float(measured_mean[2][h]),
                        "excess_day3_vs_day1_drift_kwh_per_hour": float(drift[h]),
                    }
                )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "household_summary.csv", index=False)
    chosen = []
    for is_primary in (True, False):
        for area in (0, 1):
            indices = np.flatnonzero((primary == is_primary) & (areas == area))
            ordered = indices[np.argsort(scales[indices])]
            chosen.extend(
                ordered[
                    np.rint(np.array([0.1, 0.5, 0.9]) * (len(ordered) - 1)).astype(int)
                ]
            )
    require(len(set(chosen)) == len(chosen), "Example household duplicates.")
    example = len(summary["origins"]) // 2
    # Load each compressed array once, before indexing individual household hours.
    with np.load(output / "price_aware/recorded_paths.npz") as stored:
        price_arrays = {
            mode: stored[mode] for mode in ("recursive", "measurement_refreshed")
        }
    values = []
    for h in chosen:
        for hour in range(72):
            values.append(
                {
                    "household_id": int(ids[h]),
                    "hour": hour + 1,
                    "actual_kwh": float(actual[example, h, hour]),
                    "recursive_kwh": float(price_arrays["recursive"][example, h, hour]),
                    "refreshed_kwh": float(
                        price_arrays["measurement_refreshed"][example, h, hour]
                    ),
                }
            )
    pd.DataFrame(values).to_csv(output / "example_hourly.csv", index=False)
    save_json(
        output / "household_examples.json",
        {
            "origin": summary["origins"][example],
            "selection": "Training scale ranks 10/50/90% in each region and cohort.",
            "household_ids": [int(ids[h]) for h in chosen],
        },
    )
    make_plots(
        output, frame, actual, price_arrays, chosen, ids, primary, areas, example
    )
    description = [
        "# Household-level sliding-window results",
        "",
        "household_summary.csv has one row per household, model and feedback mode: "
        "4,544 rows covering every one of the 1,136 metered households.",
        "",
        "Examples are selected from training scale ranks, independently of forecast "
        "performance. Traces use a fixed middle origin. Other/unknown-contract "
        "households are labelled as additional-cohort diagnostics.",
        "",
        "![Household examples](household_sequences.png)",
        "",
        "The drift histogram subtracts the actual day-3 minus day-1 load change "
        "from the corresponding predicted change. Zero means no additional mean "
        "drift relative to that recorded trajectory. It does not imply an exact "
        "hour-by-hour forecast.",
        "",
        "![Drift across households](household_drift.png)",
        "",
        "The refreshed path receives each real load reading after predicting it. "
        "It has more information over time than the fully recursive path. "
        "This comparison diagnoses feedback error; it is not a fair contest between "
        "two forecasts issued with the same information at the initial hour.",
        "",
        "Individual hourly fluctuations can be smoother in a squared-error point "
        "forecast than in real readings. Smoothing alone does not establish a "
        "failure; errors, bias, peaks, drift and price-response evidence must also "
        "be considered.",
    ]
    (output / "HOUSEHOLDS.md").write_text("\n".join(description) + "\n")
    if (output / "price_channels/summary.json").exists():
        explain_findings(output, summary)
    return frame


def explain_findings(output, summary):
    """Explain observed forecast drift separately from constructed price effects."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channels = json.loads((output / "price_channels/summary.json").read_text())
    results = summary["engines"]["price_aware"]
    cohort = results["primary_717"]
    high = results["price_paths"]["constant_high"]["primary_717"]
    ratio = np.asarray(high["aggregate_ratio_by_hour"])
    rows = [
        "| Forecast hours | Recursive RMSE | Reading-refreshed RMSE | "
        "Recursive aggregate bias |",
        "| --- | ---: | ---: | ---: |",
    ]
    for day in range(1, 4):
        a = cohort["recursive"]["days"][str(day)]
        b = cohort["measurement_refreshed"]["days"][str(day)]
        rows.append(
            f"| {(day-1)*24+1}–{day*24} | {a['rmse']:.3f} | {b['rmse']:.3f} | "
            f"{100*(a['aggregate_energy_ratio']-1):+.2f}% |"
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(np.arange(1, 73), 100 * (ratio - 1), color="#a54632")
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set(
        xlabel="Recursive forecast hour",
        ylabel="Demand difference (%)",
        title="Constant 2.00 vs training-median price",
    )
    components = ("absolute", "change", "history")
    for i, engine in enumerate(("ensemble", "seed2026")):
        values = channels["contributions"][engine]
        x = np.arange(3) + (i - 0.5) * 0.35
        axes[1].bar(
            x,
            [100 * values["2"][c] for c in components],
            0.35,
            label=engine,
        )
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set(
        xticks=np.arange(3),
        xticklabels=["Price level", "Price change", "Price history"],
        ylabel="Day-two contribution (percentage points)",
        title="Model input contributions, fixed measured loads",
    )
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(output / "price_mechanism.png", dpi=170)
    plt.close(fig)
    channel_rows = [
        "| Model / period | Price level | Price change | Price history | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for engine, values in channels["contributions"].items():
        for period in ("immediate", "1", "2", "3"):
            v = values[period]
            numbers = " | ".join(f"{100*v[c]:+.3f}%" for c in (*components, "total"))
            channel_rows.append(f"| {engine} / {period} | {numbers} |")
    daily = ", ".join(
        f"day {i+1}: {100*(value-1):+.2f}%"
        for i, value in enumerate(high["energy_ratio_by_day"])
    )
    prose = [
        "# Findings from explicit 72-hour sliding forecasts",
        "",
        "All 1,136 metered households were tested at 84 eligible midnight origins "
        "across January–March 2022. Every forecast uses the trained 24-hour input "
        "window: predict one hour, remove the oldest hour, append the latest load "
        "and price, and repeat 72 times. This does not change the model to a "
        "72-hour input window. No new model was trained.",
        "",
        "## 1. Recursive load feedback accumulates error",
        "",
        "Primary-cohort results (717 spot-contract households): RMSE is kWh per "
        "household per hour; aggregate bias is predicted energy relative to "
        "recorded energy in that 24-hour block.",
        "",
        *rows,
        "",
        "The refreshed comparison receives a real reading only after predicting "
        "that hour. Its nearly stable error shows that prediction feedback "
        "contributes to deterioration. It receives more information than the "
        "initial 72-hour forecast, so this is a diagnostic, not an equally informed "
        "forecast competition. Both comparisons persist origin temperature.",
        "",
        "![Forecast errors](forecast_errors.png)",
        "",
        "No household-origin forecast was exactly flat under the declared "
        "1e-5-kWh range threshold. Predictions vary, but their variation is smaller "
        "than measured loads. This is expected partly from squared-error point "
        "forecasting and should not itself be called successful behavioral modeling.",
        "",
        "## 2. Immediate and sustained price responses differ",
        "",
        "The earlier next-hour price probe did not describe the full multi-hour "
        "response. On the same history, increasing the candidate price from "
        "0.05 to 2.00 NOK/kWh raises the immediate prediction in "
        f"{100*results['endpoint_price_probe']['positive_fraction']:.2f}% of "
        "tested primary-household states. In contrast, holding 2.00 throughout "
        "the recursive simulation eventually reduces predicted demand relative "
        f"to holding {summary['normal_price']:.5f} NOK/kWh: {daily}.",
        "",
        f"Across the six constructed starts, the aggregate high-price curve "
        f"first falls below its normal-price counterpart at hour "
        f"{int(np.flatnonzero(ratio < 1)[0])+1}. This crossing describes the "
        "pooled curve, not a universal household response time.",
        "",
        "![Price mechanisms](price_mechanism.png)",
        "",
        "## 3. Which model inputs contribute?",
        "",
        "We separately changed the candidate price level, its difference from "
        "the preceding price, and encoded historical prices, while keeping "
        "measured demand histories identical. The resulting contributions "
        "sum exactly to the total high-minus-normal prediction difference. "
        "The order is price level, then price change, then history. Intermediate "
        "feature combinations are algebraic diagnostics rather than executable "
        "pricing policies. Nonlinear attribution can depend on this order.",
        "",
        *channel_rows,
        "",
        "In the ensemble, the price-level and price-change channels both raise "
        "the immediate prediction. As higher prices accumulate in the history, "
        "the historical-price channel lowers predictions enough to outweigh "
        "the positive price-level contribution. The model therefore encodes "
        "competing immediate and delayed effects. This is an explanation of "
        "its learned calculation, not evidence that either effect is causal.",
        "",
        "Seed 2026 also has a positive immediate total and a negative day-2/day-3 "
        "total in this diagnostic, but its price-change contribution has the "
        "opposite sign from the ensemble. The models agree on this broad delay "
        "pattern while disagreeing on some of the internal sensitivities.",
        "",
        "Percentages use normal-price predicted energy in the same period as "
        "denominator; immediate means the first hour and 1/2/3 are 24-hour "
        "blocks. Seed 2026 checks these clamped-history channels independently; "
        "it is not an independent real-world experiment. This table is not the "
        "fully recursive price experiment above: measured loads are held common "
        "here specifically to isolate model input channels.",
        "",
        "## 4. What can be concluded",
        "",
        "The model has delayed price sensitivity as well as an immediate "
        "response, so it is incorrect to summarize it as always predicting "
        "higher demand at higher prices. Its immediate response, longer-run "
        "drift, and sensitivity to rapid switches still require scrutiny before "
        "using it as an RL environment. An RL controller could exploit those "
        "patterns without producing a credible consumer policy.",
        "",
        "These tests establish numerical behavior, not causal elasticity, "
        "outage recovery, or conserved load shifting. No measured demand exists "
        "under the constructed alternative prices. The 72-hour experiment also "
        "extends beyond the 48-hour training rollout. The additional 419 "
        "households have other/unknown contracts and were not spot-price "
        "fine-tuning targets; their price-aware results remain exploratory.",
        "",
        "The price-free control remains invariant to prices. Independent CPU "
        "raw-input forecasts, changed inference batches, frozen-weight checks, "
        "production-cache agreement and feature-attribution identities are "
        "recorded in verification.json and price_channels/summary.json. "
        "No future measured demand enters the recursive loop; refreshed "
        "measurements enter only after their hour has been predicted.",
        "",
        "[Full household results](HOUSEHOLDS.md), "
        "[household CSV](household_summary.csv), "
        "[methodology](REPORT.md), [numerical summary](summary.json).",
    ]
    (output / "FINDINGS.md").write_text("\n".join(prose) + "\n")


def make_plots(output, frame, actual, arrays, chosen, ids, primary, areas, example):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(6, 2, figsize=(13, 16), sharex=True)
    for ax, h in zip(axes.ravel(), chosen):
        ax.plot(
            np.arange(1, 73),
            actual[example, h],
            color="black",
            linewidth=0.9,
            label="Recorded",
        )
        ax.plot(
            np.arange(1, 73),
            arrays["recursive"][example, h],
            color="#d38629",
            label="Recursive",
        )
        ax.plot(
            np.arange(1, 73),
            arrays["measurement_refreshed"][example, h],
            color="#247b91",
            label="Reading refreshed",
        )
        ax.set(
            title=f"HH {ids[h]}, {('NO1','NO5')[areas[h]]}, "
            f"{'primary' if primary[h] else 'additional'}",
            ylabel="kWh/hour",
        )
    for ax in axes[-1]:
        ax.set_xlabel("Hour")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output / "household_sequences.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    selected = frame[frame.engine.eq("price_aware")]
    for ax, is_primary in zip(axes, (True, False)):
        for mode in ("recursive", "measurement_refreshed"):
            values = selected.loc[
                selected.primary_spot_household.eq(is_primary)
                & selected.feedback.eq(mode),
                "excess_day3_vs_day1_drift_kwh_per_hour",
            ]
            ax.hist(values, bins=35, alpha=0.5, label=mode)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set(
            title=(
                "717 primary households" if is_primary else "419 additional households"
            ),
            xlabel="Extra day-3 minus day-1 drift (kWh/hour)",
            ylabel="Households",
        )
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "household_drift.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/sliding_72h_investigation_v1"
    )
    report(parser.parse_args().output)

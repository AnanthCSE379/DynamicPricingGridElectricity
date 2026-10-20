"""Create a readable report for the joint five-hour household GRU."""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/simple-response-matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from scripts.split_simple_response import ROOT


def report(folder):
    folder = Path(folder)
    model = folder / "model"
    settings = json.loads((model / "settings.json").read_text())
    manifest = json.loads((folder / "data/manifest.json").read_text())
    metrics = json.loads((model / "metrics.json").read_text())
    history = pd.read_csv(model / "training.csv")
    request = json.loads((model / "example_input.json").read_text())
    examples = pd.read_csv(model / "example_predictions.csv")
    validation = metrics["validation"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axes[0].plot(history.epoch, history.validation_rmse_kwh, color="#2463ac", linewidth=2, label="GRU: all observed targets")
    axes[0].axhline(validation["persistence"]["rmse_kwh"], color="#777777", linestyle="--", label="Persistence: observed current + target")
    best = history[history.epoch.eq(metrics["best_epoch"])].iloc[0]
    axes[0].scatter([best.epoch], [best.validation_rmse_kwh], color="#2463ac", zorder=3)
    axes[0].set(title="Validation prediction error", xlabel="Epoch", ylabel="kWh per household per half-hour")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(history.epoch, history.validation_balanced_mse_kwh2, color="#147d64", linewidth=2)
    axes[1].scatter([best.epoch], [best.validation_balanced_mse_kwh2], color="#147d64", zorder=3)
    axes[1].set(title="Score used to select the checkpoint", xlabel="Epoch", ylabel="Mean seasonal/tariff MSE (kWh²)")
    for axis in axes:
        axis.grid(alpha=.15)
    fig.suptitle("Five-hour multivariate GRU: lower is better; dot marks selected epoch")
    fig.savefig(folder / "training.png", dpi=170)
    plt.close(fig)
    lines = ["# Five-hour demand response across all households", "",
             f"**10 × {settings['households']:,} separate household demands + next price → "
             f"{settings['households']:,} next-demand predictions.**", "",
             "All demand values are kWh per half-hour; tariff is pence/kWh. "
             "One multivariate GRU reads the joint household matrix in oldest-to-newest order. "
             "The next tariff is supplied after the ten-step history encoder. "
             "No future demand or additional future price schedule is supplied.", "",
             f"Fresh single-layer GRU, {settings['width']} hidden units, {settings['parameters']:,} parameters. "
             f"Ran {len(history)} epochs; validation selected epoch {metrics['best_epoch']}. "
             "Earlier model weights were not loaded.", "",
             "## Actual raw data and missing readings", "",
             f"Preparation read {manifest['cleaning']['rows']:,} raw ToU rows against the original tariff "
             f"workbook and retained {manifest['households']:,} households. "
             f"There are {manifest['missing_readings']:,} missing household/time cells. "
             f"The observed household count ranges from {manifest['observed_households_min']:,} "
             f"to {manifest['observed_households_max']:,} per half-hour. "
             "Households remain separate throughout training and evaluation.", "",
             "Missing inputs use household means fitted on training timestamps only, with explicit "
             "observed flags. Missing targets are never imputed and contribute no loss or error. "
             "All 1,117 household outputs are still produced. Demand scales use each household's "
             "training-only maximum; price uses the actual fixed LCL range.", "",
             "## Seasonal splitting", "",
             "The original shared 80/10/10 seasonal timestamp calendar is retained. Every historical "
             "timestamp and its target must lie in the same partition, so no input/target observation "
             "crosses partitions. Purging five-hour boundary windows changes sample percentages slightly. "
             "All 12 season/tariff combinations remain in every split. Observed targets receive "
             "equal total loss weight per season/tariff combination.", "",
             "| Partition | Joint time-series windows |", "|---|---:|"]
    for name, count in settings["window_counts"].items():
        lines.append(f"| {name} | {count:,} |")
    lines += ["", "These are joint time-series samples, each containing ten rows of all household "
              "readings. They are not millions of independently sampled household transitions.", "",
              "## Held-out test errors", "",
              "| Method | Observed targets | RMSE (kWh) | MAE (kWh) |", "|---|---:|---:|---:|"]
    for name, label in [("gru_all_observed_targets", "GRU, all measured targets"),
                        ("gru_persistence_comparable", "GRU, same pairs as persistence"),
                        ("persistence", "Demand stays unchanged")]:
        score = metrics["test"][name]
        lines.append(f"| {label} | {score['count']:,} | {score['rmse_kwh']:.6f} | {score['mae_kwh']:.6f} |")
    gru = metrics["test"]["gru_persistence_comparable"]
    baseline = metrics["test"]["persistence"]
    change = 100 * (gru["rmse_kwh"] / baseline["rmse_kwh"] - 1)
    direction = "higher" if change > 0 else "lower"
    lines += ["", f"On exactly matched targets, GRU RMSE is {abs(change):.2f}% {direction} than persistence. "
              "Test errors were evaluated after checkpoint selection. Comparison with previous runs "
              "is not exact because the five-hour purge and joint sample definition differ.", "",
              "The plotted GRU curve includes all measured targets; the persistence line requires "
              "both a measured current and next reading. The matched comparison is in the table.", "",
              "![Training curves](training.png)", "", "## One saved input and output", "",
              f"History begins {request['history_timestamps'][0]} and ends {request['history_timestamps'][-1]}. "
              f"The next timestamp is {request['next_timestamp']}, with recorded tariff "
              f"{request['next_price_pence_per_kwh']:.2f} p/kWh. "
              "The complete JSON contains all ten demand rows and all household IDs. "
              "The following are the first eight outputs; actual demand is shown only for comparison.", "",
              "| Household | Current measured kWh | Predicted next kWh | Actual next kWh |", "|---|---:|---:|---:|"]
    for row in examples.head(8).itertuples():
        current = "missing" if pd.isna(row.current_demand_kwh) else f"{row.current_demand_kwh:.3f}"
        actual = "missing" if pd.isna(row.actual_next_demand_kwh) else f"{row.actual_next_demand_kwh:.3f}"
        lines.append(f"| {row.household} | {current} | {row.predicted_next_demand_kwh:.3f} | {actual} |")
    lines += ["", "```sh", "python -m household_response \\",
              "  --input results/five_hour_response/model/example_input.json \\",
              "  --output results/five_hour_response/my_prediction.json", "```", "",
              "Preserve the saved household order. The CLI validates ten consecutive half-hours "
              "when timestamps are supplied and accepts JSON null for missing readings. Output "
              "includes one prediction per household and their model-predicted sum.", "",
              "## Limits", "",
              "This is a same-year, same-household prediction task with three actual LCL tariffs. "
              "A longer history permits temporal modeling but does not itself establish a causal "
              "price response. The 64-unit joint state compresses 1,117 household histories and "
              "may discard individual detail. Missing targets cannot be verified. Summing available "
              "meter readings is not treated as a fully measured, fixed-population grid total.", "",
              "Reproduction and input details: `household_response/README.md`. "
              "Raw preparation: `scripts/prepare_household_panel.py`. "
              "Training: `scripts/train_household_response.py`."]
    (folder / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(folder / "REPORT.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=ROOT / "results/five_hour_response")
    report(parser.parse_args().folder)

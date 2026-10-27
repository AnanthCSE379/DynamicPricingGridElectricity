"""Render the load-support audit and its limits without changing any model."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from norway_energy_crisis.clean_data import ROOT, save_json
from norway_energy_crisis.preprocess import PreparedData


def read(path):
    return json.loads(path.read_text())


def report(folder, prepared):
    distribution = read(folder / "load_distribution.json")
    forecasts = read(folder / "rollout_errors.json")
    one_hour = read(folder / "one_hour_errors.json")
    probes = read(folder / "reduced_history_probes.json")
    data = PreparedData(prepared)
    training = np.asarray(data.demand[:, data.partitions == 0])
    household_quantiles = np.quantile(training, [0, 0.01, 0.05], axis=1)
    floor_support = {
        str(floor): dict(
            below_household_min_percent=float(
                100 * (floor < household_quantiles[0]).mean()
            ),
            below_household_p01_percent=float(
                100 * (floor < household_quantiles[1]).mean()
            ),
            below_household_p05_percent=float(
                100 * (floor < household_quantiles[2]).mean()
            ),
        )
        for floor in (0, 0.05, 0.1, 0.25, 0.5, 1)
    }
    save_json(folder / "candidate_floor_support.json", floor_support)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    aggregates = {}
    for axis, (label, selected) in zip(
        axes,
        (
            ("NO1 / Oslo", data.area_index == 0),
            ("NO5 / Bergen", data.area_index == 1),
            ("Cohort total (descriptive)", np.ones(data.households, dtype=bool)),
        ),
    ):
        aggregate = data.demand[selected].sum(0, dtype=np.float64)
        bins = np.linspace(0, np.ceil(aggregate.max() / 100) * 100, 31)
        aggregates[label] = {}
        for partition, split in enumerate(("train", "validation", "followup")):
            values = aggregate[data.partitions == partition]
            counts = np.histogram(values, bins)[0]
            axis.stairs(100 * counts / len(values), bins, label=split, linewidth=1.6)
            aggregates[label][split] = dict(
                minimum_kw=float(values.min()),
                median_kw=float(np.median(values)),
                maximum_kw=float(values.max()),
                hours=len(values),
                edges_kw=bins.tolist(),
                counts=counts.tolist(),
            )
        axis.set(
            title=label,
            xlabel="Hourly average aggregate demand (kW)",
            ylabel="Hours per bin (%)",
        )
        axis.legend(fontsize=8)
    figure.savefig(folder / "aggregate_load_frequency.png", dpi=180)
    plt.close(figure)
    save_json(folder / "aggregate_distribution.json", aggregates)

    lines = [
        "# Frozen response model: load support and longer rollouts",
        "",
        "Retrospective diagnostic. No model weights, RL environments, selection",
        "manifests or completed experiment results were changed.",
        "",
        "## Data and units",
        "",
        "The modeled cohort contains 717 households and 9,412,776 hourly readings.",
        "Values are kWh per one-hour interval; their numerical value equals hourly",
        "average kW. They do not measure instantaneous power or connection ratings.",
        "The complete 1,136-household panel was also checked separately.",
        "",
        "![Household frequencies](load_frequency.png)",
        "",
        "![Regional and total frequencies](aggregate_load_frequency.png)",
        "",
        "The combined cohort is a descriptive total, not one physical feeder.",
        "",
        "Training-household load frequency (left-inclusive, right-exclusive bins):",
        "",
        "| Average load (kW) | Observations | Frequency |",
        "|---|---:|---:|",
    ]
    for label, count, percent in zip(
        distribution["bin_labels_kw"],
        distribution["train"]["bin_counts"],
        distribution["train"]["bin_percent"],
    ):
        lines.append(f"| {label} | {count:,} | {percent:.3f}% |")
    lines.extend(
        [
            "",
            "Training range: 0–18.922 kW; training 1st–99th percentiles:",
            "0.041–7.604 kW. The central 90% is 0.100–4.905 kW.",
            "These are marginal pooled ranges, not guaranteed input reliability.",
            "",
            "## A minimum supply cannot be identified from the histogram",
            "",
            "A natural low demand and a demand forced down by unavailable supply are",
            "different conditions. The model has no curtailment/outage indicator,",
            "unserved-energy state or explicit appliance/deferred-demand state.",
            "The dataset does not supply paired counterfactual demand after "
            "controlled cuts.",
            "Consequently neither essential needs nor a reliable minimum delivery can",
            "be estimated merely by selecting a percentile.",
            "",
            "| Proposed uniform input (kW) | Below household training minimum |",
            "|---|---:|",
        ]
    )
    for floor, row in floor_support.items():
        lines.append(
            f"| {float(floor):.2f} | {row['below_household_min_percent']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "Household training 1st percentiles range from 0 to 2.200 kW",
            "(median 0.182 kW). Their 5th percentiles range from 0.001 to 2.556 kW",
            "(median 0.270 kW). The CSV retains all household-specific summaries.",
            "Use these as support warnings for reset generation, never as established",
            "essential-load floors. Remaining inside a marginal range does not imply",
            "the joint history, price, weather and household context is familiar.",
            "",
            "## Accuracy following naturally low loads",
            "",
            "Every eligible hourly origin after a complete 24-hour history was scored:",
            "516,240 household-hour predictions in January and 998,064 "
            "in February–March.",
            "The ensemble averages seeds 42 and 7 at each recursive step; seed 2026",
            "is evaluated as an additional previously examined sensitivity model.",
            "",
            "| Prior measured load | January N | January RMSE | "
            "Follow-up N | Follow-up RMSE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for january, followup in zip(
        one_hour["validation"]["ensemble"], one_hour["followup"]["ensemble"]
    ):
        if january["count"] == 0 and followup["count"] == 0:
            continue
        lines.append(
            f"| {january['band_kw']} | {january['count']:,} | "
            f"{january['rmse']:.4f} | {followup['count']:,} | {followup['rmse']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Errors are kWh; a small absolute error at low consumption need not be",
            "a small relative error. These observations assess ordinary conditional",
            "forecasting, not the response to forcibly curtailed deliveries.",
            "",
            "## Zero-load persistence weakness",
            "",
            "The checkpoint anchors its prediction to inverse-softplus(previous scaled",
            "demand). At zero it uses a 1e-6 scaled anchor. A finite neural correction",
            "therefore usually leaves the next prediction extremely small.",
            "Among measured zero-preceding cases, 16/47 January cases and 13/30",
            "follow-up cases have positive next-hour demand. Their actual next-hour",
            "means are 0.730 and 0.268 kWh, respectively. Those subsets are small.",
            "",
            "Artificial last-hour and whole-history reductions were probed at three",
            "candidate prices on six origins (4,302 household-origin contexts).",
            "They have no measured counterfactual labels; no interpolation or outage",
            "accuracy score is assigned to them.",
            "",
            "| Artificial zero history | Engine | Price | Mean next prediction (kWh) |",
            "|---|---|---:|---:|",
        ]
    )
    for row in probes["results"]:
        if row["factor"] == 0 and row["price"] == 1:
            lines.append(
                f"| {row['scope']} | {row['model']} | 1.00 | "
                f"{row['prediction']['mean']:.8f} |"
            )
    lines.extend(
        [
            "",
            "Being inside a numeric input range is insufficient: zero itself is",
            "observed in training, yet recovery from it is poorly represented by the",
            "persistence anchor. Do not feed curtailed delivery into this model and",
            "interpret the resulting low requests as validated consumer adaptation.",
            "",
            "## 24-, 48- and 72-hour recursion",
            "",
            "All horizons use identical midnight origins with a complete preceding",
            "24-hour history and all 72 targets inside the same partition: 28 January",
            "and 56 February–March origins. Windows overlap. These are descriptive",
            "errors, not confidence intervals or a fresh holdout comparison.",
            "Actual future consumption is used only for scoring. Candidate prices are",
            "the recorded future prices; temperature stays at the origin value.",
            "For deployment, prices beyond the next day would require an explicitly",
            "declared exogenous scenario or a forecast, "
            "not presumed day-ahead knowledge.",
            "",
            "| Partition | Model | 24h cumulative RMSE | 48h | 72h | Day-3 RMSE |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for split, row in forecasts.items():
        for name, result in row["models"].items():
            cumulative = result["cumulative"]
            lines.append(
                f"| {split} | {name} | {cumulative['24']['rmse']:.4f} | "
                f"{cumulative['48']['rmse']:.4f} | {cumulative['72']['rmse']:.4f} | "
                f"{result['day']['3']['rmse']:.4f} |"
            )
    lines.extend(
        [
            "",
            "![Forecast errors](forecast_reliability.png)",
            "",
            "The model was trained with recursive losses up to 24 hours; 48/72 hours",
            "are beyond that training horizon. Longer episodes can reveal simulated",
            "carryover but cannot establish load shifting: there is no deferred-energy",
            "queue or energy-conservation rule. Positive later forecasts alone do not",
            "prove that an earlier curtailed or price-suppressed service "
            "was recovered.",
            "",
            "## Implication for the proposed loop",
            "",
            "Use a coherent 24-hour starting history, then feed predicted requested",
            "demand and the chosen price back into the frozen model. Account for",
            "allocated delivery separately. That is the interface supported by the",
            "current training contract. Resampling complete household/region/season",
            "contexts is more defensible than independently randomizing load values.",
            "A consumer model that responds to outages would need identified outage",
            "examples or an explicitly declared physical/behavioral mechanism.",
            "Do not infer a reliable supply floor from marginal frequencies or force",
            "one merely to keep predictions away from zero.",
            "",
            "No RL retraining or episode-length change was performed by this audit.",
            "",
            "Reproduce diagnostics:",
            "",
            "```sh",
            "python -m norway_energy_crisis.analysis.audit_load_support --device mps",
            "python -m norway_energy_crisis.analysis.report_load_support",
            "```",
            "",
            "The audit requires a new output directory on rerun; use --output and",
            "pass the same directory to this reporting command.",
            "",
        ]
    )
    (folder / "REPORT.md").write_text("\n".join(lines))
    print(folder / "REPORT.md")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/load_support_v1")
    parser.add_argument("--prepared", type=Path, default=ROOT / "prepared")
    args = parser.parse_args()
    report(args.output, args.prepared)


if __name__ == "__main__":
    main()

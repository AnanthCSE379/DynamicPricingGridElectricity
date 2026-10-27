"""Inspect observed load support and frozen-model forecasts without retraining.

Historical forecast errors can assess prediction after naturally low loads.
Artificially reduced histories have no counterfactual labels in this dataset;
their predictions are sensitivity probes, never estimates of outage accuracy.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.preprocess import PreparedData, calendar
from norway_energy_crisis.train_response import load_model


EDGES = np.array([0, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, np.inf])
LABELS = [
    "0–0.1",
    "0.1–0.25",
    "0.25–0.5",
    "0.5–1",
    "1–2",
    "2–3",
    "3–5",
    "5–10",
    "10–20",
    "≥20",
]
QUANTILES = [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1]


def describe(values):
    values = np.asarray(values)
    counts = np.histogram(values, EDGES)[0]
    return dict(
        count=values.size,
        zeros=int((values == 0).sum()),
        mean=float(values.mean()),
        quantiles={
            str(q): float(v) for q, v in zip(QUANTILES, np.quantile(values, QUANTILES))
        },
        bin_counts=counts.tolist(),
        bin_percent=(100 * counts / values.size).tolist(),
    )


def distributions(data, cleaned, output):
    rows = {
        name: describe(data.demand[:, data.partitions == i])
        for i, name in enumerate(("train", "validation", "followup"))
    }
    rows["primary_all"] = describe(data.demand)
    all_demand = np.load(cleaned / "demand.npy", mmap_mode="r")
    rows["all_1136_households"] = describe(all_demand)
    train = np.asarray(data.demand[:, data.partitions == 0])
    household_quantiles = np.quantile(train, [0, 0.01, 0.05, 0.5, 0.95, 1], axis=1)
    with (output / "household_training_support.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "household_id",
                "region",
                "minimum_kw",
                "p01_kw",
                "p05_kw",
                "median_kw",
                "p95_kw",
                "maximum_kw",
                "zero_count",
            ]
        )
        for i, household in enumerate(data.preprocessor.ids):
            writer.writerow(
                [
                    household,
                    data.state["areas"][int(data.area_index[i])],
                    *household_quantiles[:, i],
                    int((train[i] == 0).sum()),
                ]
            )
    rows["household_p01_summary"] = describe(household_quantiles[1])
    rows["household_p05_summary"] = describe(household_quantiles[2])
    rows["edges_kw"] = [float(x) if np.isfinite(x) else None for x in EDGES]
    rows["bin_labels_kw"] = LABELS
    save_json(output / "load_distribution.json", rows)

    figure, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    positions = np.arange(len(LABELS))
    for offset, (name, label) in enumerate(
        (
            ("train", "Training"),
            ("validation", "January 2022"),
            ("followup", "February–March 2022"),
        )
    ):
        axes[0].bar(
            positions + (offset - 1) * 0.26,
            rows[name]["bin_percent"],
            width=0.26,
            label=label,
        )
    axes[0].set(
        xticks=positions,
        xticklabels=LABELS,
        ylabel="Household-hour observations (%)",
        xlabel="Average household power during one hour (kW)",
        title="717 spot-contract households: observed load frequency",
    )
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=9)
    for name, label in (
        ("train", "Training"),
        ("validation", "January"),
        ("followup", "February–March"),
    ):
        values = np.sort(
            np.asarray(
                data.demand[
                    :,
                    data.partitions == ("train", "validation", "followup").index(name),
                ]
            ).ravel()
        )
        indices = np.unique(np.linspace(0, len(values) - 1, 6000).astype(int))
        axes[1].plot(values[indices], 100 * (indices + 1) / len(values), label=label)
    axes[1].set(
        xlabel="Average household power (kW)",
        ylabel="Observations at or below load (%)",
        title="Cumulative distribution; full observed range",
    )
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    figure.savefig(output / "load_frequency.png", dpi=180)
    plt.close(figure)
    return rows, household_quantiles


def inputs(data, households, origins, horizon):
    """Return permitted inputs only; future measured demand/weather are excluded."""
    history_indices = origins[:, None] + np.arange(-24, 0)
    target_indices = origins[:, None] + np.arange(horizon)
    areas = data.area_index[households]
    return dict(
        history=data.preprocessor.history_features(
            data.demand[households[:, None], history_indices],
            data.prices[areas[:, None], history_indices],
            data.temperature[areas[:, None], history_indices],
            data.times[history_indices],
            households,
        ),
        static=np.asarray(data.static[households]),
        scales=data.preprocessor.scales[households],
        prices=data.preprocessor.price(data.prices[areas[:, None], target_indices]),
        calendar=calendar(data.times[target_indices]),
    )


@torch.inference_mode()
def predict(models, values, device):
    """Shared-mean recursive rollout, persisting temperature at the origin."""
    tensors = {
        key: torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in values.items()
    }
    history = tensors["history"]
    temperature = history[:, -1, 2:3]
    predictions = []
    for hour in range(tensors["prices"].shape[1]):
        price = tensors["prices"][:, hour]
        target_calendar = tensors["calendar"][:, hour]
        predicted = torch.stack(
            [
                model(history, tensors["static"], price, target_calendar)
                for model in models
            ]
        ).mean(0)
        predictions.append(predicted * tensors["scales"])
        next_row = torch.cat(
            (predicted[:, None], price[:, None], temperature, target_calendar), dim=1
        )
        history = torch.cat((history[:, 1:], next_row[:, None]), dim=1)
    result = torch.stack(predictions, dim=1).cpu().numpy()
    require(np.isfinite(result).all() and (result >= 0).all(), "Invalid prediction.")
    return result


def errors(predicted, actual):
    error = np.asarray(predicted, dtype=np.float64) - actual
    return dict(
        count=error.size,
        rmse=float(np.sqrt(np.mean(error**2))),
        mae=float(np.mean(np.abs(error))),
        bias=float(error.mean()),
    )


def one_hour(data, engines, device, output, batch_size):
    """Evaluate all eligible hourly origins; stratify by preceding measured load."""
    results = {}
    for partition, split in ((1, "validation"), (2, "followup")):
        hours = np.flatnonzero(data.partitions == partition)
        origins = np.arange(hours[0] + 24, hours[-1] + 1)
        total = len(origins) * data.households
        accumulators = {
            name: np.zeros((len(LABELS) + 1, 6), dtype=np.float64) for name in engines
        }
        for start in range(0, total, batch_size):
            flat = np.arange(start, min(start + batch_size, total))
            households, selected = (
                flat % data.households,
                origins[flat // data.households],
            )
            values = inputs(data, households, selected, 1)
            previous = np.asarray(data.demand[households, selected - 1])
            actual = np.asarray(data.demand[households, selected])
            bins = np.searchsorted(EDGES[1:], previous, side="right")
            for name, models in engines.items():
                predicted = predict(models, values, device)[:, 0]
                error = predicted.astype(float) - actual
                persistence = previous.astype(float) - actual
                for index in range(len(LABELS) + 1):
                    mask = (bins == index) if index < len(LABELS) else (previous == 0)
                    accumulator = accumulators[name][index]
                    accumulator += [
                        int(mask.sum()),
                        np.square(error[mask]).sum(),
                        np.abs(error[mask]).sum(),
                        error[mask].sum(),
                        np.square(persistence[mask]).sum(),
                        np.abs(persistence[mask]).sum(),
                    ]
            if start % (batch_size * 25) == 0:
                print(f"one-hour {split}: {start}/{total}", flush=True)
        results[split] = {}
        for name, accumulator in accumulators.items():
            result = []
            for label, row in zip([*LABELS, "exactly zero (subset)"], accumulator):
                count, squared, absolute, bias, baseline_squared, baseline_absolute = (
                    row
                )
                result.append(
                    dict(
                        band_kw=label,
                        count=int(count),
                        rmse=float(np.sqrt(squared / count)) if count else None,
                        mae=float(absolute / count) if count else None,
                        bias=float(bias / count) if count else None,
                        persistence_rmse=(
                            float(np.sqrt(baseline_squared / count)) if count else None
                        ),
                        persistence_mae=(
                            float(baseline_absolute / count) if count else None
                        ),
                    )
                )
            results[split][name] = result
        save_json(output / "one_hour_errors.json", results)
    return results


def long_rollouts(data, engines, device, output, days_per_batch):
    """Compare 24/48/72 hours on identical origins within each partition."""
    results = {}
    for partition, split in ((1, "validation"), (2, "followup")):
        origins = np.array(
            [
                origin
                for origin in range(24, len(data.times) - 71)
                if origin % 24 == 0
                and np.all(data.partitions[origin - 24 : origin + 72] == partition)
            ]
        )
        require(len(origins) > 0, "No complete 72-hour windows.")
        forecasts = {name: [] for name in engines}
        measured, persistence, daily, previous = [], [], [], []
        for start in range(0, len(origins), days_per_batch):
            chosen = origins[start : start + days_per_batch]
            households = np.tile(np.arange(data.households), len(chosen))
            selected = np.repeat(chosen, data.households)
            values = inputs(data, households, selected, 72)
            for name, models in engines.items():
                forecasts[name].append(
                    predict(models, values, device).reshape(
                        len(chosen), data.households, 72
                    )
                )
            # Labels and baselines remain separate from the model-input function.
            measured.append(
                np.asarray(
                    data.demand[households[:, None], selected[:, None] + np.arange(72)]
                ).reshape(len(chosen), data.households, 72)
            )
            past = np.asarray(
                data.demand[households[:, None], selected[:, None] + np.arange(-24, 0)]
            )
            persistence.append(
                np.repeat(past[:, -1, None], 72, axis=1).reshape(
                    len(chosen), data.households, 72
                )
            )
            daily.append(
                np.tile(past, (1, 3)).reshape(len(chosen), data.households, 72)
            )
            previous.append(past[:, -1].reshape(len(chosen), data.households))
            print(
                f"72-hour {split}: "
                f"{min(start + days_per_batch, len(origins))}/{len(origins)} days",
                flush=True,
            )
        actual = np.concatenate(measured)
        forecasts = {name: np.concatenate(rows) for name, rows in forecasts.items()}
        forecasts.update(
            persistence=np.concatenate(persistence),
            repeat_previous_day=np.concatenate(daily),
        )
        np.savez_compressed(
            output / f"{split}_72h_forecasts.npz",
            origins=origins,
            actual=actual,
            previous=np.concatenate(previous),
            **forecasts,
        )
        rows = {}
        for name, predicted in forecasts.items():
            error = predicted.astype(float) - actual
            rows[name] = dict(
                cumulative={
                    str(h): errors(predicted[:, :, :h], actual[:, :, :h])
                    for h in (1, 24, 48, 72)
                },
                day={
                    str(day + 1): errors(
                        predicted[:, :, day * 24 : (day + 1) * 24],
                        actual[:, :, day * 24 : (day + 1) * 24],
                    )
                    for day in range(3)
                },
                rmse_by_horizon=np.sqrt(np.mean(error**2, axis=(0, 1))).tolist(),
                mean_predicted_by_day=[
                    float(predicted[:, :, day * 24 : (day + 1) * 24].mean())
                    for day in range(3)
                ],
            )
        results[split] = dict(
            origins=len(origins),
            first=str(np.datetime64(int(data.times[origins[0]]), "ns")),
            last=str(np.datetime64(int(data.times[origins[-1]]), "ns")),
            models=rows,
        )
        save_json(output / "rollout_errors.json", results)
    return results


def reduced_history_probes(data, engines, device, output, household_quantiles):
    """Numerical sensitivity only: no measured labels exist for interventions."""
    dates = [
        "2022-01-02",
        "2022-01-16",
        "2022-01-29",
        "2022-02-02",
        "2022-02-28",
        "2022-03-29",
    ]
    origins = np.searchsorted(
        data.times, np.array(dates, dtype="datetime64[ns]").astype("int64")
    )
    households = np.tile(np.arange(data.households), len(origins))
    selected = np.repeat(origins, data.households)
    original = inputs(data, households, selected, 1)
    rows = []
    for scope in ("last_hour", "all_24_hours"):
        for factor in (0, 0.25, 0.5, 0.75, 1):
            values = {key: value.copy() for key, value in original.items()}
            if scope == "last_hour":
                values["history"][:, -1, 0] *= factor
            else:
                values["history"][:, :, 0] *= factor
            previous = values["history"][:, -1, 0] * values["scales"]
            for raw_price in (0.05, 1, 2):
                values["prices"][:] = data.preprocessor.price(raw_price)
                for name, models in engines.items():
                    predicted = predict(models, values, device)[:, 0]
                    rows.append(
                        dict(
                            scope=scope,
                            factor=factor,
                            price=raw_price,
                            model=name,
                            previous_mean_kw=float(previous.mean()),
                            prediction=describe(predicted),
                            previous_below_household_training_minimum=float(
                                (previous < household_quantiles[0, households]).mean()
                            ),
                            previous_below_household_training_p01=float(
                                (previous < household_quantiles[1, households]).mean()
                            ),
                            label=(
                                "Sensitivity only; no counterfactual truth "
                                "or accuracy score."
                            ),
                        )
                    )
    result = dict(origins=dates, household_origins=len(households), results=rows)
    save_json(output / "reduced_history_probes.json", result)
    return result


def plot_errors(one_hour_results, rollout_results, output):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for split, style in (("validation", "-"), ("followup", "--")):
        rows = one_hour_results[split]["ensemble"][: len(LABELS)]
        axes[0].plot(
            np.arange(len(LABELS)),
            [row["rmse"] or np.nan for row in rows],
            style,
            marker="o",
            label=f"{split}: ensemble",
        )
        axes[0].plot(
            np.arange(len(LABELS)),
            [row["persistence_rmse"] or np.nan for row in rows],
            style,
            marker=".",
            alpha=0.65,
            label=f"{split}: persistence",
        )
    axes[0].set(
        xticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        xlabel="Preceding observed household load (kW)",
        ylabel="Next-hour RMSE (kWh)",
        title="Natural low-load histories: forecast accuracy",
    )
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].legend(fontsize=8)
    for name, row in rollout_results["followup"]["models"].items():
        axes[1].plot(np.arange(1, 73), row["rmse_by_horizon"], label=name)
    axes[1].axvline(24, color="grey", alpha=0.4)
    axes[1].axvline(48, color="grey", alpha=0.4)
    axes[1].set(
        xlabel="Recursive forecast lead (hours)",
        ylabel="RMSE (kWh)",
        title="February–March: same origins at all forecast horizons",
    )
    axes[1].legend(fontsize=8)
    figure.savefig(output / "forecast_reliability.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=ROOT / "prepared")
    parser.add_argument("--cleaned", type=Path, default=ROOT / "cleaned")
    parser.add_argument("--responses", type=Path, default=ROOT / "runs/comparison_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/load_support_v1")
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--days-per-batch", type=int, default=3)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a new output directory.")
    require(
        args.batch_size > 0 and args.days_per_batch > 0, "Batch sizes must be positive."
    )
    if args.device == "mps":
        require(
            torch.backends.mps.is_available(),
            "MPS unavailable; no silent CPU fallback.",
        )
    torch.set_num_threads(4)
    data = PreparedData(args.prepared)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    distributions_result, household_quantiles = distributions(
        data, args.cleaned, args.output
    )
    models, hashes = {}, {}
    for seed in (42, 7, 2026):
        path = args.responses / f"transformer_24h_seed{seed}_price/best.pt"
        model, bundle = load_model(path)
        require(bundle["preprocessing"] == data.state, "Preprocessing mismatch.")
        require(
            bundle["prepared_manifest_sha256"]
            == digest(args.prepared / "manifest.json"),
            "Data mismatch.",
        )
        models[seed] = model.to(args.device).eval().requires_grad_(False)
        hashes[str(seed)] = digest(path)
    engines = dict(ensemble=[models[42], models[7]], reserved=[models[2026]])
    save_json(
        args.output / "provenance.json",
        dict(
            prepared=digest(args.prepared / "manifest.json"),
            response_weights=hashes,
            source=digest(__file__),
            device=args.device,
            torch=torch.__version__,
            numpy=np.__version__,
            temperature="persist origin",
            scope="Retrospective diagnostic; no training or model selection.",
            units=(
                "kWh per one-hour interval equals numerical hourly-average kW; "
                "no instantaneous-power claims."
            ),
            low_history_probe=(
                "No counterfactual observations; interpolation/extrapolation "
                "accuracy is unidentified."
            ),
        ),
    )
    one_hour_results = one_hour(
        data, engines, args.device, args.output, args.batch_size
    )
    rollout_results = long_rollouts(
        data, engines, args.device, args.output, args.days_per_batch
    )
    reduced_history_probes(data, engines, args.device, args.output, household_quantiles)
    plot_errors(one_hour_results, rollout_results, args.output)
    save_json(
        args.output / "status.json",
        dict(
            status="complete",
            elapsed_seconds=time.monotonic() - started,
            observations=distributions_result["primary_all"]["count"],
        ),
    )
    print(
        json.dumps(
            dict(
                status="complete",
                output=str(args.output),
                elapsed=time.monotonic() - started,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

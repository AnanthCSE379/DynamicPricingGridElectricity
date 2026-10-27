"""Investigate frozen consumer models one explicit sliding-window hour at a time."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.response_v2.data import ConsumerData
from norway_energy_crisis.response_v2.model import ConsumerSimulator, load_model


def advance(history, scaled_load, price, target_calendar, temperature):
    """Drop the oldest row, append the latest load and rebuild derived features.

    This implementation deliberately reconstructs the whole visible window's
    derived channels independently of the production append helper.
    """
    base_row = torch.cat(
        (scaled_load[:, None], price[:, None], temperature[:, None], target_calendar),
        dim=-1,
    )
    base = torch.cat((history[:, 1:, :9], base_row[:, None]), dim=1)
    changes = torch.cat(
        (torch.zeros_like(base[:, :1, 1]), base[:, 1:, 1] - base[:, :-1, 1]), dim=1
    )
    available = torch.cat(
        (torch.zeros_like(changes[:, :1]), torch.ones_like(changes[:, 1:])), dim=1
    )
    return torch.cat(
        (base, torch.log1p(base[:, :, :1]), changes[:, :, None], available[:, :, None]),
        dim=-1,
    )


@torch.inference_mode()
def sliding_forecast(
    models,
    inputs,
    measured_after_hour=None,
    probe_prices=False,
    standardized_price_step=None,
    endpoints=None,
    trace=False,
):
    """Predict first, then append either that prediction or the new measurement.

    The open-loop path receives no measured future loads. The refreshed path
    assimilates the reading for hour t only after forecasting hour t. Both retain
    the same origin temperature, isolating load-history feedback differences.
    """
    history = inputs.history.clone()
    horizon = inputs.prices.shape[1]
    if measured_after_hour is not None:
        require(
            measured_after_hour.shape == inputs.prices.shape,
            "Measurement-feedback shape mismatch.",
        )
        require(
            bool(torch.isfinite(measured_after_hour).all())
            and bool((measured_after_hour >= 0).all()),
            "Invalid measurements.",
        )
    if probe_prices:
        require(
            standardized_price_step > 0 and len(endpoints) == 2,
            "Price probes require a step and endpoints.",
        )
    temperature = history[:, -1, 2]
    predictions, local_changes, endpoint_changes, traces = [], [], [], []
    for hour in range(horizon):
        if trace:
            traces.append(history.cpu().numpy().copy())
        price = inputs.prices[:, hour]
        target_calendar = inputs.calendar[:, hour]
        encoded = [model.encode(history) for model in models]

        def predict(candidate):
            return (
                torch.stack(
                    [
                        model.predict_encoded(
                            context, history, inputs.static, candidate, target_calendar
                        )
                        for model, context in zip(models, encoded)
                    ]
                ).mean(0)
                * inputs.scales
            )

        predicted = predict(price)
        predictions.append(predicted)
        if probe_prices:
            local_changes.append(
                predict(price + standardized_price_step)
                - predict(price - standardized_price_step)
            )
            endpoint_changes.append(
                predict(torch.full_like(price, endpoints[1]))
                - predict(torch.full_like(price, endpoints[0]))
            )
        latest = (
            predicted if measured_after_hour is None else measured_after_hour[:, hour]
        )
        history = advance(
            history, latest / inputs.scales, price, target_calendar, temperature
        )
    prediction = torch.stack(predictions, dim=1).cpu().numpy()
    require(
        np.isfinite(prediction).all() and (prediction >= 0).all(),
        "Nonfinite or negative predictions.",
    )
    return {
        "prediction": prediction,
        "local_change": (
            torch.stack(local_changes, 1).cpu().numpy() if probe_prices else None
        ),
        "endpoint_change": (
            torch.stack(endpoint_changes, 1).cpu().numpy() if probe_prices else None
        ),
        "trace": traces,
    }


def groups(data):
    mask = np.zeros(len(data.ids), dtype=bool)
    mask[data.primary] = True
    return {
        "primary_717": mask,
        "additional_419": ~mask,
        "all_1136": np.ones(len(mask), dtype=bool),
    }


def metrics(predicted, actual, mask):
    p, y = predicted[:, mask], actual[:, mask]
    error = p.astype(np.float64) - y
    result = {
        "households": int(mask.sum()),
        "origins": len(p),
        "rmse_by_hour": np.sqrt((error**2).mean((0, 1))).tolist(),
        "mae_by_hour": np.abs(error).mean((0, 1)).tolist(),
        "bias_by_hour": error.mean((0, 1)).tolist(),
        "flat_72h_fraction": float((np.ptp(p, axis=-1) <= 1e-5).mean()),
        "mean_temporal_std_prediction": float(p.std(-1).mean()),
        "mean_temporal_std_actual": float(y.std(-1).mean()),
        "regional_pool_note": "Region-specific results are reported separately.",
        "days": {},
    }
    for day in range(3):
        sl = slice(day * 24, (day + 1) * 24)
        e = error[:, :, sl]
        result["days"][str(day + 1)] = {
            "rmse": float(np.sqrt((e**2).mean())),
            "mae": float(np.abs(e).mean()),
            "bias": float(e.mean()),
            "predicted_mean_kwh": float(p[:, :, sl].mean()),
            "actual_mean_kwh": float(y[:, :, sl].mean()),
            "aggregate_energy_ratio": float(
                p[:, :, sl].sum() / max(y[:, :, sl].sum(), 1e-6)
            ),
        }
    return result


def summarize_feedback(data, result, actual):
    summary = {}
    for cohort, mask in groups(data).items():
        modes = {
            mode: metrics(result[mode], actual, mask)
            for mode in ("recursive", "measurement_refreshed")
        }
        for area, name in enumerate(data.state["areas"]):
            chosen = mask & (data.area_index == area)
            if chosen.any():
                modes[name] = {
                    mode: metrics(result[mode], actual, chosen)
                    for mode in ("recursive", "measurement_refreshed")
                }
        summary[cohort] = modes
    primary = data.primary
    for kind in ("local", "endpoint"):
        values = result[f"{kind}_change"][:, primary]
        summary[f"{kind}_price_probe"] = {
            "positive_fraction": float((values > 1e-6).mean()),
            "negative_fraction": float((values < -1e-6).mean()),
            "flat_fraction": float((np.abs(values) <= 1e-6).mean()),
            "median_change_kwh": float(np.median(values)),
            "mean_change_kwh": float(values.mean()),
            "positive_fraction_by_day": [
                float((values[:, :, d * 24 : (d + 1) * 24] > 1e-6).mean())
                for d in range(3)
            ],
            "scope": "Fixed refreshed histories; artificial candidate prices.",
        }
    return summary


def load_engines(selection, data, device):
    engines = {}
    for kind, key in (
        ("price_aware", "price_checkpoints"),
        ("price_free", "control_checkpoints"),
    ):
        paths = selection[key][:2]
        models = []
        for path in paths:
            require(
                digest(path) == selection["checkpoint_sha256"][path],
                "Selected checkpoint changed.",
            )
            model, saved = load_model(path, device)
            require(
                saved["preprocessing"] == data.state
                and saved["data_fingerprint"] == data.fingerprint,
                "Response/data mismatch.",
            )
            models.append(model)
        engines[kind] = models
    return engines


def recorded_paths(data, origins, models, device, output, batch_size):
    """Enumerate every metered household at every eligible midnight origin."""
    modes = {
        key: []
        for key in (
            "recursive",
            "measurement_refreshed",
            "local_change",
            "endpoint_change",
        )
    }
    step = 0.05 / data.state["price_std"]
    endpoints = [float(data.features.base.price(p)) for p in (0.05, 2.0)]
    for number, origin in enumerate(origins):
        day = {key: [] for key in modes}
        for start in range(0, len(data.ids), batch_size):
            hh = data.all_households[start : start + batch_size]
            inputs = data.inputs(hh, np.full(len(hh), origin), 72).to(device)
            measured = data.targets(hh, np.full(len(hh), origin), 72).to(device)
            recursive = sliding_forecast(models, inputs)
            refreshed = sliding_forecast(
                models, inputs, measured, True, step, endpoints
            )
            day["recursive"].append(recursive["prediction"])
            day["measurement_refreshed"].append(refreshed["prediction"])
            day["local_change"].append(refreshed["local_change"])
            day["endpoint_change"].append(refreshed["endpoint_change"])
        for key in modes:
            modes[key].append(np.concatenate(day[key]))
        save_json(
            output / "progress.json",
            {
                "stage": "recorded_paths",
                "origin": number + 1,
                "total_origins": len(origins),
                "time": str(data.times[origin]),
            },
        )
        if number % 12 == 0 or number == len(origins) - 1:
            print(
                f"{output.name}: recorded origins {number+1}/{len(origins)}", flush=True
            )
    result = {k: np.stack(v) for k, v in modes.items()}
    np.savez_compressed(
        output / "recorded_paths.npz", origins=origins, household_ids=data.ids, **result
    )
    return result


def selected_origins(origins):
    return origins[np.linspace(0, len(origins) - 1, min(6, len(origins)), dtype=int)]


def price_paths(
    data, origins, models, device, output, normal_price, batch_size, price_free=False
):
    schedules = {
        "constant_normal": np.full(72, normal_price),
        "constant_low": np.full(72, 0.05),
        "constant_high": np.full(72, 2.0),
    }
    if not price_free:
        schedules.update(
            {
                "high_6h_then_normal": np.r_[
                    np.full(6, 2.0), np.full(66, normal_price)
                ],
                "low_6h_then_normal": np.r_[
                    np.full(6, 0.05), np.full(66, normal_price)
                ],
                "alternating_6h": np.tile(np.r_[np.full(6, 0.05), np.full(6, 2.0)], 6),
            }
        )
    results = {}
    for name, schedule in schedules.items():
        days = []
        for origin in origins:
            values = []
            for start in range(0, len(data.ids), batch_size):
                hh = data.all_households[start : start + batch_size]
                inputs = data.inputs(hh, np.full(len(hh), origin), 72).to(device)
                raw = np.broadcast_to(schedule, (len(hh), 72)).copy()
                prices = torch.tensor(data.features.base.price(raw), device=device)
                values.append(
                    sliding_forecast(models, replace(inputs, prices=prices))[
                        "prediction"
                    ]
                )
            days.append(np.concatenate(values))
        result = np.stack(days)
        results[name] = result
        np.savez_compressed(
            output / f"{name}.npz",
            prediction=result,
            prices=schedule,
            origins=origins,
            household_ids=data.ids,
        )
        print(f"{output.name}: {name} complete", flush=True)
    normal = results["constant_normal"]
    summary = {}
    for name, result in results.items():
        summary[name] = {}
        for cohort, chosen in groups(data).items():
            q = result[:, chosen]
            reference = normal[:, chosen]
            ratio = q.sum((0, 1)) / np.maximum(reference.sum((0, 1)), 1e-6)
            summary[name][cohort] = {
                "aggregate_ratio_by_hour": ratio.tolist(),
                "energy_ratio_by_day": [
                    float(
                        q[:, :, d * 24 : (d + 1) * 24].sum()
                        / max(reference[:, :, d * 24 : (d + 1) * 24].sum(), 1e-6)
                    )
                    for d in range(3)
                ],
                "day3_household_ratio_quantiles": np.quantile(
                    q[:, :, -24:].sum(-1)
                    / np.maximum(reference[:, :, -24:].sum(-1), 1e-6),
                    [0, 0.1, 0.5, 0.9, 1],
                ).tolist(),
            }
        if price_free:
            np.testing.assert_allclose(result, normal, rtol=0, atol=0)
    save_json(output / "price_paths.json", summary)
    return summary


def verify_raw_loop(
    paths, data, origin, household_positions, reference, refreshed=False
):
    """Independent CPU loop through raw public API; predict one hour per call."""
    hh = np.asarray(household_positions)
    indices = np.arange(origin - 24, origin)
    q = np.array(data.demand[hh[:, None], indices], copy=True)
    price = np.array(data.prices[data.area_index[hh, None], indices], copy=True)
    temp = np.array(data.temperature[data.area_index[hh, None], indices], copy=True)
    stamps = data.times[indices].copy()
    last_temp = temp[:, -1].copy()
    simulator = ConsumerSimulator(paths, "cpu")
    values = []
    for hour in range(72):
        p = np.asarray(data.prices[data.area_index[hh], origin + hour])
        predicted = simulator.forecast(
            q, price, temp, stamps, [data.ids[h] for h in hh], p[:, None]
        )[:, 0]
        values.append(predicted)
        latest = np.asarray(data.demand[hh, origin + hour]) if refreshed else predicted
        q = np.column_stack((q[:, 1:], latest))
        price = np.column_stack((price[:, 1:], p))
        temp = np.column_stack((temp[:, 1:], last_temp))
        stamps = np.r_[stamps[1:], data.times[origin + hour]]
    result = np.stack(values, -1)
    expected = reference[hh]
    np.testing.assert_allclose(result, expected, rtol=3e-3, atol=3e-4)
    return float(np.max(np.abs(result - expected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument("--cleaned", type=Path, default=ROOT / "cleaned")
    parser.add_argument(
        "--responses", type=Path, default=ROOT / "runs/response_refinement_v2"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/sliding_72h_investigation_v1"
    )
    parser.add_argument("--device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a new output directory.")
    if args.device == "mps":
        require(torch.backends.mps.is_available(), "MPS unavailable.")
        require(
            __import__("os").environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
            "Disable MPS fallback explicitly.",
        )
    torch.set_num_threads(4)
    started = time.monotonic()
    data = ConsumerData(args.cleaned, args.data)
    selection = json.loads((args.responses / "selection.json").read_text())
    engines = load_engines(selection, data, args.device)
    origins = np.r_[data.origins["validation"], data.origins["test"]]
    args.output.mkdir(parents=True)
    actual = np.stack([np.asarray(data.demand[:, t + np.arange(72)]) for t in origins])
    np.savez_compressed(
        args.output / "actual.npz",
        demand=actual,
        origins=origins,
        household_ids=data.ids,
    )
    normal = float(
        np.median(
            data.prices[
                data.area_index[data.primary, None],
                np.flatnonzero(data.partitions == 0),
            ]
        )
    )
    summary = {
        "engines": {},
        "normal_price": normal,
        "origins": origins.tolist(),
        "household_ids": data.ids,
        "additional_households": len(data.ids) - len(data.primary),
        "primary_households": len(data.primary),
    }
    raw_checks = {}
    for engine, models in engines.items():
        save_json(args.output / "status.json", {"stage": "running", "engine": engine})
        folder = args.output / engine
        folder.mkdir()
        frozen = [[p.cpu().clone() for p in m.parameters()] for m in models]
        result = recorded_paths(
            data, origins, models, args.device, folder, args.batch_size
        )
        summary["engines"][engine] = summarize_feedback(data, result, actual)
        summary["engines"][engine]["price_paths"] = price_paths(
            data,
            selected_origins(origins),
            models,
            args.device,
            folder,
            normal,
            args.batch_size,
            engine == "price_free",
        )
        chosen = []
        for area in (0, 1):
            members = np.flatnonzero(data.area_index == area)
            chosen.extend(members[np.linspace(0, len(members) - 1, 6, dtype=int)])
        paths = selection[
            "price_checkpoints" if engine == "price_aware" else "control_checkpoints"
        ][:2]
        raw_checks[engine] = {
            mode: verify_raw_loop(
                paths,
                data,
                origins[0],
                chosen,
                result[mode][0],
                mode == "measurement_refreshed",
            )
            for mode in ("recursive", "measurement_refreshed")
        }
        # Repeat all households at a different inference batch size.
        repeated = []
        for start in range(0, len(data.ids), max(1, args.batch_size // 2)):
            hh = data.all_households[start : start + max(1, args.batch_size // 2)]
            inputs = data.inputs(hh, np.full(len(hh), origins[0]), 72).to(args.device)
            repeated.append(sliding_forecast(models, inputs)["prediction"])
        repeated = np.concatenate(repeated)
        np.testing.assert_allclose(
            repeated, result["recursive"][0], rtol=3e-3, atol=3e-4
        )
        raw_checks[engine]["batch_repeat_max_difference"] = float(
            np.max(np.abs(repeated - result["recursive"][0]))
        )
        require(
            all(
                torch.equal(p.cpu(), value)
                for m, stored in zip(models, frozen)
                for p, value in zip(m.parameters(), stored)
            ),
            "Frozen weights changed.",
        )
        if engine == "price_aware":
            cached = np.concatenate(
                [
                    np.load(
                        args.responses / "predictions" / f"{s}_revised_ensemble.npz"
                    )["prediction"]
                    for s in ("validation", "test")
                ]
            )
            comparison = result["recursive"][:, data.primary]
            np.testing.assert_allclose(comparison, cached, rtol=3e-3, atol=3e-4)
            raw_checks[engine]["production_cache_max_difference"] = float(
                np.max(np.abs(comparison - cached))
            )
        else:
            require(
                np.max(np.abs(result["local_change"])) == 0
                and np.max(np.abs(result["endpoint_change"])) == 0,
                "Price-free probe invariance failed.",
            )
        save_json(args.output / "summary.json", summary)
        del result
    save_json(
        args.output / "verification.json",
        {
            "raw_cpu_checks": raw_checks,
            "weights_unchanged": True,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    provenance = {
        "data": data.fingerprint,
        "selection_sha256": digest(args.responses / "selection.json"),
        "source_sha256": digest(__file__),
        "device": args.device,
        "temperature": "Origin persistence in both feedback modes.",
        "extra_cohort_note": (
            "Additional 419 lack verified hourly spot exposure and were not "
            "fine-tuning targets; their price-response results are exploratory."
        ),
        "labels": "Refreshed mode appends actual load only after forecasting it.",
        "interventions": "Constructed prices have no measured response targets.",
    }
    save_json(args.output / "provenance.json", provenance)
    write_report(args.output, data, summary, actual)
    save_json(
        args.output / "status.json",
        {"stage": "complete", "seconds": time.monotonic() - started},
    )


def write_report(output, data, summary, actual):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = summary["engines"]["price_aware"]
    rows = [
        "| Cohort | Feedback | Day 1 RMSE | Day 2 RMSE | Day 3 RMSE |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for cohort in ("primary_717", "additional_419"):
        for mode in ("recursive", "measurement_refreshed"):
            d = results[cohort][mode]["days"]
            rows.append(
                f"| {cohort} | {mode} | {d['1']['rmse']:.5f} | "
                f"{d['2']['rmse']:.5f} | {d['3']['rmse']:.5f} |"
            )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, cohort in zip(axes, ("primary_717", "additional_419")):
        for engine, engine_results in summary["engines"].items():
            for mode in ("recursive", "measurement_refreshed"):
                ax.plot(
                    np.arange(1, 73),
                    engine_results[cohort][mode]["rmse_by_hour"],
                    label=f"{engine}, {mode}",
                )
        ax.set(title=cohort, xlabel="Forecast hour", ylabel="Household RMSE (kWh)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "forecast_errors.png", dpi=160)
    plt.close(fig)
    saved = np.load(output / "price_aware/recorded_paths.npz")
    example = len(summary["origins"]) // 2
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(
        np.arange(1, 73),
        actual[example, data.primary].sum(0),
        label="Recorded demand",
        color="black",
    )
    for mode in ("recursive", "measurement_refreshed"):
        ax.plot(np.arange(1, 73), saved[mode][example, data.primary].sum(0), label=mode)
    ax.set(
        xlabel="Hour",
        ylabel="717-household aggregate kWh/hour",
        title=f"Fixed example origin: {data.times[summary['origins'][example]]}",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "observed_vs_sliding.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name, cohorts in results["price_paths"].items():
        ax.plot(
            np.arange(1, 73),
            cohorts["primary_717"]["aggregate_ratio_by_hour"],
            label=name,
        )
    ax.set(
        xlabel="Hour",
        ylabel="Predicted demand / constant-normal prediction",
        title="Constructed price paths: primary cohort",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "recursive_price_effects.png", dpi=160)
    plt.close(fig)
    text = [
        "# Explicit 72-hour sliding consumer forecasts",
        "",
        "All 1,136 metered households were included: 717 primary and 419 additional "
        "households, reported separately. No model was trained or modified.",
        "",
        "For each eligible midnight origin in January-March, predict one hour, "
        "remove the oldest hour, append the newest value and repeat for 72 hours. "
        "Recursive mode appends predictions. Refreshed mode appends the actual "
        "reading after forecasting its hour. Both persist origin temperature.",
        "",
        "Refreshed mode uses new observations arriving each hour. It is not "
        "a 72-hour forecast made entirely at the original start time.",
        "",
        *rows,
        "",
        "![Forecast errors](forecast_errors.png)",
        "",
        "![Example sequence](observed_vs_sliding.png)",
        "",
        "## Price interventions",
        "",
        "![Price paths](recursive_price_effects.png)",
        "",
        "Normal price is the primary-cohort training median. Low/high prices are "
        "0.05/2.00 NOK/kWh. Six timestamp-selected origins and all households "
        "are used. No measured counterfactual targets exist for these curves.",
        "",
        "Local probes use recorded price +/-0.05 NOK/kWh on a shared refreshed "
        "history. Endpoint probes use 0.05 versus 2.00 on that same history. "
        "Results refer to primary households unless stated otherwise.",
        "",
        "## Evidence boundaries",
        "",
        "The additional 419 have other/unknown contracts and unavailable paid "
        "hourly tariffs. They appeared in price-free pretraining, not spot-cohort "
        "fine-tuning. Their price-aware curves are extrapolation diagnostics. "
        "Price-free forecasts are reported separately in summary.json.",
        "",
        "These are retrospective comparisons on previously examined months. "
        "Origins overlap and households share regional prices; predictions are "
        "not independent trials. Conditional-mean smoothing alone does not "
        "establish a model failure.",
        "",
        "The 72-hour recursion exceeds the 48-hour training horizon. There is "
        "no allocation, clipping, outage feedback or deferred-energy queue. "
        "Recorded prices condition the scenario; three future days of prices "
        "are not claimed to be known operationally at the start.",
        "",
        "summary.json contains every lead, day, region and cohort. Trajectories "
        "are in recorded_paths.npz. verification.json records CPU raw-API loops, "
        "changed-batch repeats, cache agreement and frozen weights.",
    ]
    (output / "REPORT.md").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()

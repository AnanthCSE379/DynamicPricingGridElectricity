"""Paired household forecasts, baselines, uncertainty and sequence diagnostics."""

from pathlib import Path
import json

import numpy as np
import torch

from norway_energy_crisis.clean_data import require, save_json
from norway_energy_crisis.response_v2.data import origins_for
from norway_energy_crisis.response_v2.model import (
    ConsumerSimulator,
    append_prediction,
    rollout,
)


@torch.inference_mode()
def forecast(models, data, origins, horizon, device, batch_size=128):
    """Enumerate the same primary households at each declared origin."""
    predictions = []
    for origin in origins:
        day = []
        for start in range(0, len(data.primary), batch_size):
            h = data.primary[start : start + batch_size]
            inputs = data.inputs(h, np.full(len(h), origin), horizon).to(device)
            day.append(rollout(models, inputs).cpu().numpy())
        predictions.append(np.concatenate(day))
    require(len(predictions) > 0, "No evaluation origins.")
    return np.stack(predictions)


def actuals(data, origins, horizon):
    return np.stack(
        [
            np.asarray(data.demand[data.primary[:, None], origin + np.arange(horizon)])
            for origin in origins
        ]
    )


def summarize(predicted, actual, data, origins):
    require(predicted.shape == actual.shape, "Forecast/target shape mismatch.")
    require(np.isfinite(predicted).all() and (predicted >= 0).all(), "Bad prediction.")
    error = predicted.astype(np.float64) - actual
    horizon = predicted.shape[-1]
    result = {
        "days": len(origins),
        "households": len(data.primary),
        "horizon": horizon,
        "origins": [int(x) for x in origins],
        "rmse_by_horizon": np.sqrt((error**2).mean((0, 1))).tolist(),
        "mae_by_horizon": np.abs(error).mean((0, 1)).tolist(),
        "energy_bias_kwh_per_household": float(error.sum(-1).mean()),
        "regions": {},
        "variety": {
            "mean_within_household_temporal_std_kwh": float(predicted.std(-1).mean()),
            "mean_across_household_std_kwh": float(predicted.std(1).mean()),
            "flat_household_day_fraction": float(
                (np.ptp(predicted, axis=-1) <= 1e-5).mean()
            ),
        },
    }
    for end in (1, 6, 24, 48, 72):
        if end <= horizon:
            e = error[:, :, :end]
            result[f"rmse_{end}h"] = float(np.sqrt((e**2).mean()))
            result[f"mae_{end}h"] = float(np.abs(e).mean())
            result[f"daily_mse_{end}h"] = (e**2).mean((1, 2)).tolist()
    result["day_errors"] = {
        str(start // 24 + 1): {
            "rmse": float(np.sqrt((error[:, :, start : start + 24] ** 2).mean())),
            "bias": float(error[:, :, start : start + 24].mean()),
        }
        for start in range(0, horizon, 24)
    }
    for number, name in enumerate(data.state["areas"]):
        mask = data.area_index[data.primary] == number
        regional = error[:, mask].sum(1)
        pred = predicted[:, mask].sum(1)
        truth = actual[:, mask].sum(1)
        result["regions"][name] = {
            "aggregate_rmse_24h": float(np.sqrt((regional[:, :24] ** 2).mean())),
            "aggregate_rmse": float(np.sqrt((regional**2).mean())),
            "mean_daily_peak_error_kw": float((pred.max(1) - truth.max(1)).mean()),
        }
    return result


def baseline_predictions(data, origins, horizon):
    persistence, yesterday = [], []
    for origin in origins:
        persistence.append(
            np.repeat(
                data.demand[data.primary, origin - 1, None],
                horizon,
                axis=1,
            )
        )
        yesterday.append(
            np.asarray(
                data.demand[
                    data.primary[:, None],
                    origin - 24 + np.arange(horizon)[None] % 24,
                ]
            )
        )
    return {"persistence": np.stack(persistence), "previous_day": np.stack(yesterday)}


def paired_blocks(candidate, reference, block=7, repetitions=5000):
    """Circular moving blocks preserve paired whole days, regions and households."""
    candidate, reference = np.asarray(candidate), np.asarray(reference)
    require(candidate.shape == reference.shape and len(candidate) > 0, "Pairing.")
    rng = np.random.default_rng(42)
    n = len(candidate)
    starts = rng.integers(n, size=(repetitions, (n + block - 1) // block))
    indices = (starts[..., None] + np.arange(block)) % n
    indices = indices.reshape(repetitions, -1)[:, :n]
    difference = (candidate - reference)[indices].mean(1)
    return {
        "mean_daily_mse_difference": float((candidate - reference).mean()),
        "ci95": np.quantile(difference, [0.025, 0.975]).tolist(),
        "block_days": block,
        "days": n,
        "resamples": repetitions,
    }


@torch.inference_mode()
def natural_low_errors(models, data, split, device, batch_size=256):
    """Score every eligible naturally low preceding reading, including true zeros."""
    split_id = ("train", "validation", "test").index(split)
    origins = origins_for(data.partitions, split_id, 1, daily=False)
    previous = np.asarray(data.demand[data.primary[:, None], origins - 1])
    hh, tt = np.nonzero(previous <= 0.25)
    households, times = data.primary[hh], origins[tt]
    predictions = []
    for start in range(0, len(hh), batch_size):
        h, t = households[start : start + batch_size], times[start : start + batch_size]
        inputs = data.inputs(h, t, 1).to(device)
        predictions.append(rollout(models, inputs).cpu().numpy()[:, 0])
    if not predictions:
        return {"events": 0, "bands": {}}
    predicted = np.concatenate(predictions)
    measured = np.asarray(data.demand[households, times])
    prior = previous[hh, tt]
    result = {"events": len(hh), "bands": {}}
    bands = {
        "zero": prior == 0,
        "zero_to_positive": (prior == 0) & (measured > 0),
        "0_to_0.05": (prior > 0) & (prior <= 0.05),
        "0.05_to_0.1": (prior > 0.05) & (prior <= 0.1),
        "0.1_to_0.25": (prior > 0.1) & (prior <= 0.25),
    }
    for name, chosen in bands.items():
        if chosen.any():
            error = predicted[chosen].astype(np.float64) - measured[chosen]
            result["bands"][name] = {
                "count": int(chosen.sum()),
                "households": int(len(np.unique(households[chosen]))),
                "rmse": float(np.sqrt((error**2).mean())),
                "mae": float(np.abs(error).mean()),
                "bias": float(error.mean()),
                "mean_prediction": float(predicted[chosen].mean()),
                "mean_measured": float(measured[chosen].mean()),
            }
    return result


@torch.inference_mode()
def price_curves(models, data, origins, device, batch_size=128):
    """Compare all 21 prices at identical histories without forcing a direction."""
    grid = np.array([0.05, *np.arange(0.1, 2.01, 0.1)], dtype=np.float32)
    days = []
    for origin in origins:
        curves = []
        for start in range(0, len(data.primary), batch_size):
            hh = data.primary[start : start + batch_size]
            inputs = data.inputs(hh, np.full(len(hh), origin), 1).to(device)
            encoded = [m.encode(inputs.history) for m in models]
            outputs = []
            for price in grid:
                p = torch.full_like(
                    inputs.prices[:, 0], float(data.features.base.price(price))
                )
                scaled = torch.stack(
                    [
                        m.predict_encoded(
                            e, inputs.history, inputs.static, p, inputs.calendar[:, 0]
                        )
                        for m, e in zip(models, encoded)
                    ]
                ).mean(0)
                outputs.append((scaled * inputs.scales).cpu().numpy())
            curves.append(np.stack(outputs, -1))
        days.append(np.concatenate(curves))
    curves = np.stack(days)
    span = curves.max(-1) - curves.min(-1)
    mean = np.maximum(curves.mean(-1), 1e-6)
    return {
        "grid": grid.tolist(),
        "origins": [int(x) for x in origins],
        "mean_household_curve_kwh": curves.mean((0, 1)).tolist(),
        "regional_curves_kwh": {
            name: curves[:, data.area_index[data.primary] == i].sum(1).mean(0).tolist()
            for i, name in enumerate(data.state["areas"])
        },
        "household_relative_span_quantiles": np.quantile(
            span / mean, [0, 0.1, 0.5, 0.9, 1]
        ).tolist(),
        "flat_response_fraction": float((span <= 1e-6).mean()),
        "high_price_increases_fraction": float(
            (curves[:, :, -1] > curves[:, :, 0]).mean()
        ),
        "household_curves": curves,
    }


@torch.inference_mode()
def sequence(models, inputs, normal, high, pulse_hours=0, cap=None, cut_hours=0):
    """Diagnostic-only altered-consumption feedback; normal rollout never does this."""
    history = inputs.history
    temperature = history[:, -1, 2]
    requested, consumed = [], []
    for t in range(inputs.prices.shape[1]):
        pulse = 24 <= t < 24 + pulse_hours
        p = torch.full_like(inputs.prices[:, t], high if pulse else normal)
        scaled = torch.stack(
            [m(history, inputs.static, p, inputs.calendar[:, t]) for m in models]
        ).mean(0)
        q = scaled * inputs.scales
        used = q.clamp_max(cap) if cap is not None and 24 <= t < 24 + cut_hours else q
        requested.append(q)
        consumed.append(used)
        history = append_prediction(
            history,
            used / inputs.scales,
            p,
            temperature,
            inputs.calendar[:, t],
        )
    return torch.stack(requested, 1), torch.stack(consumed, 1)


@torch.inference_mode()
def sequence_suite(
    models, data, origins, device, output, batch_size=128, check_budget=None
):
    """Run all households with 72 recovery hours even after the longest pulse."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    normal_raw = float(np.median(data.prices[:, data.partitions == 0]))
    normal = float(data.features.base.price(normal_raw))
    high = float(data.features.base.price(2.0))
    cases = {"normal": (0, None, 0)}
    for duration in (1, 6, 24):
        cases[f"price_pulse_{duration}h"] = (duration, None, 0)
    for cap in (0, 0.05, 0.1, 0.75, 1.25):
        for duration in (1, 6):
            cases[f"cap_{cap:g}_{duration}h"] = (0, cap, duration)
    trajectories, bindings = {}, {}
    frozen = [
        [p.detach().cpu().clone() for p in model.parameters()] for model in models
    ]
    require(
        all(not model.training for model in models),
        "Sequence models must be in eval mode.",
    )
    for name, (pulse, cap, duration) in cases.items():
        if check_budget is not None:
            check_budget()
        values, uses = [], []
        for origin in origins:
            day, used_day = [], []
            for start in range(0, len(data.primary), batch_size):
                hh = data.primary[start : start + batch_size]
                inputs = data.inputs(hh, np.full(len(hh), origin), 120).to(device)
                q, used = sequence(models, inputs, normal, high, pulse, cap, duration)
                day.append(q.cpu().numpy())
                used_day.append(used.cpu().numpy())
            values.append(np.concatenate(day))
            uses.append(np.concatenate(used_day))
        q, used = np.stack(values), np.stack(uses)
        require(np.isfinite(q).all() and (q >= 0).all(), "Invalid sequence.")
        require(np.all(used <= q), "Cap raised consumption.")
        trajectories[name] = q
        bindings[name] = int((used < q).sum())
        np.savez_compressed(output / f"{name}.npz", requested=q, consumed=used)
    control = trajectories["normal"]
    metrics = {}
    for name, values in trajectories.items():
        require(
            np.array_equal(values[:, :, :24], control[:, :, :24]), "Warmup changed."
        )
        pulse, cap, duration = cases[name]
        release = 24 + max(pulse, duration)
        following = values[:, :, release : release + 72]
        baseline = control[:, :, release : release + 72]
        require(following.shape[-1] == 72, "Recovery window was truncated.")
        metrics[name] = {
            "recovery_hours": following.shape[-1],
            "first_release_aggregate_ratio": float(
                following[:, :, 0].sum() / max(baseline[:, :, 0].sum(), 1e-6)
            ),
            "last_24h_aggregate_ratio": float(
                following[:, :, -24:].sum() / max(baseline[:, :, -24:].sum(), 1e-6)
            ),
            "last_24h_households_within_control_band_fraction": float(
                (
                    np.abs(following[:, :, -24:] - baseline[:, :, -24:])
                    <= np.maximum(0.1 * baseline[:, :, -24:], 0.05)
                )
                .all(-1)
                .mean()
            ),
            "changed_consumption_readings": bindings[name],
        }
    repeated = {}
    for name in ("normal", "cap_0.1_1h"):
        if check_budget is not None:
            check_budget()
        pulse, cap, duration = cases[name]
        rows = []
        smaller = max(1, batch_size // 2)
        for start in range(0, len(data.primary), smaller):
            hh = data.primary[start : start + smaller]
            inputs = data.inputs(hh, np.full(len(hh), origins[0]), 120).to(device)
            q, _ = sequence(models, inputs, normal, high, pulse, cap, duration)
            rows.append(q.cpu().numpy())
        repeated_values = np.concatenate(rows)
        expected = trajectories[name][0]
        np.testing.assert_allclose(repeated_values, expected, rtol=3e-3, atol=3e-4)
        repeated[name] = float(np.max(np.abs(repeated_values - expected)))
    for model, original in zip(models, frozen):
        require(
            all(
                torch.equal(p.detach().cpu(), v)
                for p, v in zip(model.parameters(), original)
            ),
            "Sequence inference changed frozen model weights.",
        )
    if all(not m.config["use_price"] for m in models):
        for name in ("price_pulse_1h", "price_pulse_6h", "price_pulse_24h"):
            np.testing.assert_allclose(trajectories[name], control, rtol=0, atol=0)
    save_json(
        output / "metrics.json",
        {
            "normal_price_nok_kwh": normal_raw,
            "origins": [int(t) for t in origins],
            "cases": metrics,
            "repeat_max_abs_difference_kwh": repeated,
            "frozen_weights_unchanged": True,
            "scope": "Constructed probes, not measured recovery accuracy.",
        },
    )
    return metrics


def verify_trained_raw_api(checkpoints, data, sequence_output):
    """Rebuild raw inputs on CPU and compare with the saved MPS recursion."""
    sequence_output = Path(sequence_output)
    metadata = json.loads((sequence_output / "metrics.json").read_text())
    origin = metadata["origins"][0]
    positions = []
    for area in range(len(data.state["areas"])):
        chosen = np.flatnonzero(data.area_index[data.primary] == area)
        if len(chosen):
            chosen = chosen[np.argsort(data.features.scales[data.primary[chosen]])]
            positions.extend(
                chosen[np.linspace(0, len(chosen) - 1, min(6, len(chosen)), dtype=int)]
            )
    positions = np.asarray(positions, dtype=int)
    hh = data.primary[positions]
    past = np.arange(origin - 24, origin)
    simulator = ConsumerSimulator(checkpoints, "cpu")
    result = simulator.forecast(
        data.demand[hh[:, None], past],
        data.prices[data.area_index[hh, None], past],
        data.temperature[data.area_index[hh, None], past],
        data.times[past],
        [data.ids[i] for i in hh],
        np.full((len(hh), 120), metadata["normal_price_nok_kwh"], np.float32),
    )
    reference = np.load(sequence_output / "normal.npz")["requested"][0, positions]
    np.testing.assert_allclose(result, reference, rtol=3e-3, atol=3e-4)
    checked = {
        "household_ids": [data.ids[i] for i in hh],
        "hours": 120,
        "max_abs_cpu_mps_difference_kwh": float(np.max(np.abs(result - reference))),
        "raw_api_consistent": True,
        "scope": "Raw-input reconstruction with independent CPU checkpoint loading.",
    }
    save_json(sequence_output / "raw_api_verification.json", checked)
    return checked


def write_report(output):
    """Write a results-led report and standard scientific plots from saved outputs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    selection = json.loads((output / "selection.json").read_text())
    metrics = json.loads((output / "evaluation.json").read_text())
    rows = [
        "| Model | 24h RMSE | 48h RMSE | 72h RMSE |",
        "| --- | ---: | ---: | ---: |",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, result in metrics["test"].items():
        rows.append(
            f"| {name} | {result['rmse_24h']:.5f} | "
            f"{result['rmse_48h']:.5f} | {result['rmse_72h']:.5f} |"
        )
        if name in (
            "incumbent",
            "revised_ensemble",
            "price_free_ensemble",
            "previous_day",
        ):
            ax.plot(np.arange(1, 73), result["rmse_by_horizon"], label=name)
    ax.set(xlabel="Forecast lead (hours)", ylabel="Household RMSE (kWh)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "forecast_horizons.png", dpi=160)
    plt.close(fig)
    curves = metrics["curves"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, result in curves.items():
        ax.plot(result["grid"], result["mean_household_curve_kwh"], label=name)
    ax.set(xlabel="Supplied price (NOK/kWh)", ylabel="Mean predicted next-hour kWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "price_response.png", dpi=160)
    plt.close(fig)
    chosen = selection["configuration"]
    low_rows = [
        "| Engine | Zero → positive N | RMSE | MAE |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in (
        "incumbent",
        "revised_ensemble",
        "price_free_ensemble",
        f"{chosen}_seed2026_price",
    ):
        value = metrics["natural_low"][f"test_{name}"]["bands"].get("zero_to_positive")
        if value:
            low_rows.append(
                f"| {name} | {value['count']} | {value['rmse']:.5f} | "
                f"{value['mae']:.5f} |"
            )
    sequence_plot = False
    seq = output / "sequences/revised_ensemble"
    if (seq / "normal.npz").exists():
        control = np.load(seq / "normal.npz")["requested"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, duration in zip(axes, (1, 6)):
            for cap in (0.05, 0.1, 0.75, 1.25):
                values = np.load(seq / f"cap_{cap:g}_{duration}h.npz")["requested"]
                ratio = values.sum((0, 1)) / np.maximum(control.sum((0, 1)), 1e-6)
                ax.plot(np.arange(120) - 24, ratio, label=f"{cap:g} kW")
            ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
            ax.axvspan(0, duration, color="grey", alpha=0.15)
            ax.set(
                title=f"{duration}-hour constructed cap",
                xlabel="Hours from cap start",
                ylabel="Predicted request / uncut control",
            )
            ax.legend()
        fig.tight_layout()
        fig.savefig(output / "sequence_recovery.png", dpi=160)
        plt.close(fig)
        sequence_plot = True
    passed = metrics["acceptance"]
    report = "\n".join(
        [
            "# Consumer simulator refinement",
            "",
            f"Selected configuration: {selection['configuration']}.",
            "",
            "No allocation or price-selector policy was trained. Consumer outputs are "
            "hourly household consumption. Price responsiveness was measured, "
            "not forced by a diversity reward or a monotonicity assumption.",
            "",
            "## February-March retrospective follow-up",
            "",
            *rows,
            "",
            "![Forecast errors](forecast_horizons.png)",
            "",
            "## Price sensitivity",
            "",
            "![Conditional price curves](price_response.png)",
            "",
            "Flat or inconsistent price effects are reported as limitations. "
            "Consumption variation and price sensitivity are reported separately "
            "in evaluation.json. Seed disagreement is not calibrated uncertainty.",
            "",
            "## Natural low-load accuracy",
            "",
            *low_rows,
            "",
            "Counts are sparse and households share calendar conditions. "
            "These measured transitions are not confirmed curtailment events.",
            "",
            "## Constructed sequence diagnostics",
            "",
            (
                "![Paired cap probes](sequence_recovery.png)"
                if sequence_plot
                else "Sequence files are separate diagnostic outputs."
            ),
            "",
            "Artificial recovery curves have no measured counterfactual target. "
            "Returning to the control curve is not imposed as a training label.",
            "",
            "## Acceptance checks",
            "",
            "```json",
            json.dumps(passed, indent=2),
            "```",
            "",
            "## Evidence boundaries",
            "",
            "Contract/descriptive survey answers are assumed stable retrospectively. "
            "Regional reanalysis temperature is held at its origin value. "
            "Rollouts condition on recorded candidate-price paths; prices beyond "
            "day ahead are not claimed to have been known operationally. "
            "January and February-March were previously examined: development "
            "and retrospective comparisons, not fresh holdouts. The 72-hour evaluation "
            "exceeds the 48-hour training horizon. Cap tests have no measured "
            "counterfactual recovery targets. Forecast gains and price "
            "curves do not establish causal elasticity or outage behaviour.",
            "",
            "See manifest.json, preflight.json, fit histories, selection.json and "
            "evaluation.json for provenance, all seeds and paired uncertainty.",
        ]
    )
    (output / "REPORT.md").write_text(report + "\n")

"""Evaluate sub-0.1 kW histories and demand-capped minimum-supply reservations.

Measured next-hour errors and artificial curtailment probes are separate. The
latter have no counterfactual truth and cannot identify a reliable supply floor.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from norway_energy_crisis.analysis.audit_load_support import inputs, predict
from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.minimum_supply import MinimumSupply
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.train_response import load_model
from norway_energy_crisis.capacity.scenarios import calibrate


EDGES = np.array(
    [0, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.25], dtype=np.float32
)
LABELS = ["zero"] + [f"({low:g}, {high:g}]" for low, high in zip(EDGES[:-1], EDGES[1:])]
CAPS = (
    None,
    0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.075,
    0.099,
    0.1,
    0.25,
    "adaptive",
)


def load_engines(data, responses, device):
    models, hashes = {}, {}
    for seed in (42, 7, 2026):
        path = responses / f"transformer_24h_seed{seed}_price/best.pt"
        model, bundle = load_model(path)
        require(
            bundle["preprocessing"] == data.state
            and bundle["prepared_manifest_sha256"]
            == digest(data.folder / "manifest.json"),
            "Frozen response/data mismatch.",
        )
        models[seed] = model.to(device).eval().requires_grad_(False)
        hashes[str(path)] = digest(path)
    return dict(ensemble=[models[42], models[7]], reserved=[models[2026]]), hashes


def measured_errors(data, engines, device, output, batch_size):
    """Enumerate naturally low preceding loads; never relabel a forced cut."""
    result = {}
    for partition, split in enumerate(("train", "validation", "followup")):
        hours = np.flatnonzero(data.partitions == partition)
        origins = np.arange(hours[0] + 24, hours[-1] + 1)
        previous_panel = np.asarray(data.demand[:, origins - 1])
        bins = np.searchsorted(EDGES, previous_panel, side="left")
        counts = np.bincount(bins.ravel(), minlength=len(EDGES) + 1)
        result[split] = dict(
            total_household_hours=int(previous_panel.size),
            counts=counts[: len(EDGES)].tolist(),
            labels=LABELS,
        )
        if split == "train":
            continue
        households, positions = np.nonzero(previous_panel <= EDGES[-1])
        selected_origins = origins[positions]
        previous = previous_panel[households, positions]
        band = bins[households, positions]
        actual = np.asarray(data.demand[households, selected_origins])
        forecasts = {name: np.empty(len(households), np.float32) for name in engines}
        for start in range(0, len(households), batch_size):
            chosen = slice(start, start + batch_size)
            values = inputs(data, households[chosen], selected_origins[chosen], 1)
            for name, models in engines.items():
                forecasts[name][chosen] = predict(models, values, device)[:, 0]
        result[split]["models"] = {}
        for name, forecast in forecasts.items():
            rows = []
            error = forecast.astype(float) - actual
            persistence = previous.astype(float) - actual
            for index, label in enumerate(LABELS):
                mask = band == index
                count = int(mask.sum())
                rows.append(
                    dict(
                        band_kw=label,
                        count=count,
                        households=int(np.unique(households[mask]).size),
                        rmse=(
                            float(np.sqrt(np.mean(error[mask] ** 2))) if count else None
                        ),
                        mae=float(np.mean(np.abs(error[mask]))) if count else None,
                        bias=float(error[mask].mean()) if count else None,
                        persistence_rmse=(
                            float(np.sqrt(np.mean(persistence[mask] ** 2)))
                            if count
                            else None
                        ),
                        actual_next_mean=float(actual[mask].mean()) if count else None,
                        predicted_next_mean=(
                            float(forecast[mask].mean()) if count else None
                        ),
                    )
                )
            result[split]["models"][name] = rows
        np.savez_compressed(
            output / f"{split}_natural_low_forecasts.npz",
            households=households,
            origins=selected_origins,
            previous=previous,
            actual=actual,
            **forecasts,
        )
        save_json(output / "natural_low_load_errors.json", result)
        print(f"natural {split}: {len(households):,} forecasts per engine", flush=True)
    return result


@torch.inference_mode()
def intervention_rollout(models, values, cap_kw, device, cut_hours=6):
    """Diagnostic only: feed restricted delivery back for six hours, then release.

    This intentionally probes the unsupported feedback path. The production
    environment continues to append requests. No future measured load or weather
    is read here, and no counterfactual accuracy is assigned to the output.
    """
    tensors = {
        key: torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in values.items()
    }
    history = tensors["history"]
    temperature = history[:, -1, 2:3]
    cap = (
        None
        if cap_kw is None
        else torch.as_tensor(
            np.broadcast_to(cap_kw, len(history)).copy(),
            dtype=torch.float32,
            device=device,
        )
    )
    predictions, feedbacks = [], []
    for hour in range(tensors["prices"].shape[1]):
        price, target_calendar = (
            tensors["prices"][:, hour],
            tensors["calendar"][:, hour],
        )
        predicted = torch.stack(
            [
                model(history, tensors["static"], price, target_calendar)
                for model in models
            ]
        ).mean(0)
        demand = predicted * tensors["scales"]
        feedback = (
            torch.minimum(demand, cap)
            if cap is not None and hour < cut_hours
            else demand
        )
        predictions.append(demand)
        feedbacks.append(feedback)
        next_row = torch.cat(
            (
                feedback[:, None] / tensors["scales"][:, None],
                price[:, None],
                temperature,
                target_calendar,
            ),
            dim=1,
        )
        history = torch.cat((history[:, 1:], next_row[:, None]), dim=1)
    predicted = torch.stack(predictions, dim=1).cpu().numpy()
    feedback = torch.stack(feedbacks, dim=1).cpu().numpy()
    require(
        np.isfinite(predicted).all()
        and (predicted >= 0).all()
        and np.isfinite(feedback).all()
        and (feedback >= 0).all(),
        "Nonfinite or negative intervention trajectory.",
    )
    return predicted, feedback


def intervention_tests(data, engines, baseline, device, output):
    dates = ["2022-01-16", "2022-02-28"]
    origins = np.searchsorted(
        data.times, np.array(dates, dtype="datetime64[ns]").astype("int64")
    )
    households = np.tile(np.arange(data.households), len(origins))
    selected = np.repeat(origins, data.households)
    values = inputs(data, households, selected, 72)
    training_min = np.min(data.demand[:, data.partitions == 0], axis=1)[households]
    adaptive = np.asarray(baseline.floor_kw)[households]
    rows = []
    for price in (0.05, 1, 2):
        values["prices"][:] = data.preprocessor.price(price)
        for name, models in engines.items():
            uncut, _ = intervention_rollout(models, values, None, device)
            for cap in CAPS:
                label = "uncut" if cap is None else str(cap)
                cap_values = adaptive if cap == "adaptive" else cap
                predicted, feedback = (
                    (uncut, uncut)
                    if cap is None
                    else intervention_rollout(models, values, cap_values, device)
                )
                daily = [
                    float(predicted[:, day * 24 : (day + 1) * 24].mean())
                    for day in range(3)
                ]
                uncut_daily = [
                    float(uncut[:, day * 24 : (day + 1) * 24].mean())
                    for day in range(3)
                ]
                rows.append(
                    dict(
                        model=name,
                        price_nok_kwh=price,
                        cap_kw=label,
                        mean_request_by_hour=predicted.mean(0).astype(float).tolist(),
                        mean_history_feedback_by_hour=feedback.mean(0)
                        .astype(float)
                        .tolist(),
                        day_mean_request_kwh=daily,
                        day_mean_difference_from_uncut_kwh=[
                            a - b for a, b in zip(daily, uncut_daily)
                        ],
                        day3_request_ratio_to_uncut=(
                            daily[2] / uncut_daily[2] if uncut_daily[2] else None
                        ),
                        history_below_household_training_minimum_fraction=float(
                            (feedback < training_min[:, None]).mean()
                        ),
                        post_release_below_0_001_fraction=float(
                            (predicted[:, 6:] < 0.001).mean()
                        ),
                        first_six_hour_curtailment_kwh=float(
                            (predicted[:, :6] - feedback[:, :6]).sum()
                        ),
                        finite_nonnegative=True,
                        interpretation=(
                            "Unlabelled input intervention; differences from "
                            "uncut are not prediction errors."
                        ),
                    )
                )
                print(f"72h stress {name}, price {price}, cap {label}", flush=True)
                save_json(
                    output / "intervention_stress.json",
                    dict(
                        dates=dates,
                        households=data.households,
                        cut_hours=6,
                        release_hours=66,
                        results=rows,
                    ),
                )
    return rows


def allocation_tests(data, baseline, output):
    rng = np.random.default_rng(717)
    summaries, total = [], 0
    for name, supply in (
        ("adaptive", baseline),
        ("fixed_0.1", MinimumSupply.fixed(data)),
    ):
        floors = np.asarray(supply.floor_kw)
        rejected = served = 0
        for area in range(2):
            indices = np.flatnonzero(data.area_index == area)
            for _ in range(30):
                requested = rng.uniform(0, 3, len(indices))
                requested[::13] = 0
                essential = rng.uniform(0, 0.5, len(indices)) * requested
                protected = np.maximum(
                    essential, np.minimum(requested, floors[indices])
                )
                previous = rng.uniform(0, 30, len(indices))
                previous_served = previous * rng.uniform(0, 1, len(indices))
                for multiplier in (0, 0.5, 0.999, 1, 1.001, 1.5, 4):
                    budget = float(protected.sum() * multiplier)
                    for method in ("fair", "fcfs", "proportional"):
                        total += 1
                        if multiplier < 1:
                            try:
                                supply.allocate(
                                    requested,
                                    essential,
                                    budget,
                                    previous,
                                    previous_served,
                                    method,
                                    rng.permutation(len(indices)),
                                    household_indices=indices,
                                )
                            except ValueError as error:
                                require(
                                    "Capacity cannot meet" in str(error),
                                    "Unexpected rejection.",
                                )
                                rejected += 1
                            else:
                                raise AssertionError(
                                    "Infeasible baseline was silently accepted."
                                )
                            continue
                        result = supply.allocate(
                            requested,
                            essential,
                            budget,
                            previous,
                            previous_served,
                            method,
                            rng.permutation(len(indices)),
                            household_indices=indices,
                        )
                        require(
                            (result.delivered >= protected - 1e-8).all()
                            and (result.delivered <= requested + 1e-8).all()
                            and result.delivered.sum() <= budget + 1e-8,
                            "Baseline allocation violated constraints.",
                        )
                        np.testing.assert_allclose(
                            result.essential_delivered, essential
                        )
                        np.testing.assert_allclose(
                            result.flexible_delivered + essential, result.delivered
                        )
                        require(
                            (result.delivered[requested == 0] == 0).all(),
                            "Artificial consumption.",
                        )
                        served += 1
        summaries.append(
            dict(
                baseline=name,
                valid_allocations=served,
                infeasible_explicit_rejections=rejected,
                regional_reserved_kw=[
                    float(floors[data.area_index == area].sum()) for area in range(2)
                ],
            )
        )
    scenarios = calibrate(data)
    results = dict(
        cases=total,
        summaries=summaries,
        scenario_capacities=[asdict(scenario) for scenario in scenarios],
        all_constraints_passed=True,
    )
    save_json(output / "allocation_stress.json", results)
    return results


def create_report(data, baseline, natural, interventions, allocations, output):
    floors = np.asarray(baseline.floor_kw)
    summary = dict(
        minimum=float(floors.min()),
        median=float(np.median(floors)),
        maximum=float(floors.max()),
        below_0_1_households=int((floors < 0.1).sum()),
        household_count=len(floors),
        rule=baseline.rule,
        validated_behavioral_floor=False,
        selection=(
            "Declared positive training 1st percentile; not selected "
            "by counterfactual accuracy."
        ),
    )
    save_json(output / "baseline_summary.json", summary)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")
    x = np.arange(len(LABELS))
    for split, style in (("validation", "-"), ("followup", "--")):
        rows = natural[split]["models"]["ensemble"]
        axes[0].plot(
            x,
            [r["rmse"] if r["rmse"] is not None else np.nan for r in rows],
            style,
            marker="o",
            label=split,
        )
    axes[0].set(
        xticks=x,
        xticklabels=LABELS,
        ylabel="Next-hour RMSE (kWh)",
        xlabel="Naturally observed preceding load (kW)",
        title="Observed low-load prediction errors",
    )
    axes[0].tick_params(axis="x", rotation=65)
    axes[0].legend()
    for cap in ("uncut", "0", "0.001", "0.01", "0.05", "0.1", "adaptive"):
        row = next(
            r
            for r in interventions
            if r["model"] == "ensemble"
            and r["price_nok_kwh"] == 1
            and r["cap_kw"] == cap
        )
        axes[1].plot(np.arange(1, 73), row["mean_request_by_hour"], label=cap)
    axes[1].axvline(6, color="grey", linestyle=":", label="Artificial cut ends")
    axes[1].set(
        xlabel="Hour",
        ylabel="Mean predicted request (kWh)",
        title="Unvalidated six-hour input cuts, then release",
    )
    axes[1].legend(fontsize=8)
    figure.savefig(output / "low_load_stress.png", dpi=180)
    plt.close(figure)

    lines = [
        "# Minimum-supply implementation and sub-0.1 kW stress tests",
        "",
        "## Conclusion",
        "",
        "No reliable minimum supply, or most accurate counterfactual floor, can be",
        "identified from this observational dataset. A positive allocation floor",
        "does not validate the consumer model's response to forced curtailment.",
        "",
        "Implemented an optional household-specific reservation: the linear 1st",
        "percentile of that household's strictly positive training consumption.",
        "The 1% choice is a declared lower-tail support assumption, not an",
        "optimized accuracy threshold. It avoids basing a reservation on one",
        "isolated minimum and uses no validation/follow-up data for fitting.",
        "",
        f"Reservation range: **{floors.min():.4f}–{floors.max():.4f} kW**;",
        f"median **{np.median(floors):.4f} kW**. {int((floors < 0.1).sum())}/717",
        "households have reservations below 0.1 kW.",
        "",
        "## Implemented behavior",
        "",
        "For each household, protect max(assumed essential requirement,",
        "min(current request, baseline × interval hours)). Allocate any remaining",
        "regional budget with the selected existing allocator. Original essential",
        "and flexible-service accounting is retained. Fair priority accounts for",
        "baseline flexible energy already served in the current hour.",
        "",
        "Never force a household to consume its reservation. A zero request remains",
        "zero. Reject a scenario explicitly if its combined protected requirements",
        "cannot fit the regional capacity; no cross-region energy transfer is used.",
        "Consumer histories still receive requested demand, not curtailed delivery.",
        "No consumption-history clipping or replacement of real zero observations",
        "was introduced. This means the supply safeguard does not repair the",
        "checkpoint's existing near-zero persistence weakness.",
        "",
        "## Measured next-hour accuracy",
        "",
        "All eligible hourly origins with preceding measured consumption ≤0.25 kW",
        "were evaluated. History/target windows remain inside their partition.",
        "Future demand is used only as a label; future measured weather is excluded.",
        "Errors below are ensemble-42/7 RMSE in kWh. These evaluate natural loads,",
        "not consumers forcibly restricted to those levels.",
        "",
        "| Prior load (kW) | Train N | January N | January RMSE | "
        "Follow-up N | Follow-up RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for index, (january, followup) in enumerate(
        zip(
            natural["validation"]["models"]["ensemble"],
            natural["followup"]["models"]["ensemble"],
        )
    ):
        jan_rmse = f"{january['rmse']:.4f}" if january["count"] else "—"
        follow_rmse = f"{followup['rmse']:.4f}" if followup["count"] else "—"
        lines.append(
            f"| {LABELS[index]} | {natural['train']['counts'][index]:,} | "
            f"{january['count']:,} | {jan_rmse} | "
            f"{followup['count']:,} | {follow_rmse} |"
        )
    lines.extend(
        [
            "",
            "Lower absolute error in a low-load bin is not proof of accurate",
            "relative response or a justified allocation floor. Small counts and",
            "household concentration are available in the JSON diagnostics.",
            "",
            "## Artificial sub-0.1 kW stress",
            "",
            "Two real starting contexts (16 January and 28 February), all 717",
            "households, three constant prices (0.05, 1, 2 NOK/kWh), 13 input",
            "cap scenarios, and two response engines (42/7 ensemble and 2026).",
            "For six hours only, feed min(predicted request, test cap) into history;",
            "then release the cap for 66 hours. The adaptive case uses the fitted",
            "household reservation as its cap. This is intentionally an unsupported",
            "feedback path and is never activated by the production allocator.",
            "",
            "The inputs span zero, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075,",
            "0.099, 0.1 and 0.25 kW, plus adaptive and uncut controls.",
            "All trajectories use known calendar and candidate price schedules",
            "and persist origin temperature. No future measured load is supplied.",
            "",
            "Differences below compare modeled day-three requests with the same",
            "engine's uncut trajectory. They are not accuracy estimates, measured",
            "load recovery, validated rebound or consumer welfare.",
            "",
            "| Engine | Input cap | Day-3 request / uncut, price 1 |",
            "|---|---|---:|",
        ]
    )
    for row in interventions:
        if row["price_nok_kwh"] == 1:
            lines.append(
                f"| {row['model']} | {row['cap_kw']} | "
                f"{100*row['day3_request_ratio_to_uncut']:.2f}% |"
            )
    lines.extend(
        [
            "",
            "![Stress results](low_load_stress.png)",
            "",
            "## Allocation stress",
            "",
            f"{allocations['cases']:,} constructed allocation cases across two",
            "regional household pools, all three allocators, heterogeneous demand,",
            "zero requests, original essential requirements "
            "and prior service histories.",
            "Both adaptive and fixed-0.1 reservations were tested. Conservation,",
            "demand bounds, protected supply and explicit infeasibility rejection",
            "passed. Numerical outcomes do not establish behavioral realism.",
            "",
            "## Use",
            "",
            "```python",
            "from norway_energy_crisis.minimum_supply import MinimumSupply",
            "from norway_energy_crisis.capacity.environment import CapacityPriceEnv",
            "",
            "baseline = MinimumSupply.load('path/to/minimum_supply.json')",
            "env = CapacityPriceEnv(provider, engine, scenarios, cache_folder,",
            "                       minimum_supply=baseline)",
            "```",
            "",
            "The optional argument activates the new allocation rule. Omitting it",
            "preserves the old allocation behavior. Snapshots reject mismatched",
            "baseline fingerprints. Existing completed results were not overwritten;",
            "source changes require a new run folder for future training.",
            "",
            "Reproduce:",
            "",
            "```sh",
            "python -m norway_energy_crisis.analysis.stress_minimum_supply"
            " --device mps",
            "```",
            "",
            "Choose a new --output directory on rerun. No RL or response-model",
            "retraining was performed. No 'lowest reliable' floor was established.",
            "",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument("--responses", type=Path, default=ROOT / "runs/comparison_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/minimum_supply_v1")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a new output directory.")
    require(args.batch_size > 0, "Batch size must be positive.")
    if args.device == "mps":
        require(
            torch.backends.mps.is_available(),
            "MPS unavailable; no silent CPU fallback.",
        )
    torch.set_num_threads(4)
    data = PreparedData(args.data)
    args.output.mkdir(parents=True)
    started = time.monotonic()
    baseline = MinimumSupply.fit(data)
    baseline.save(args.output / "minimum_supply.json")
    MinimumSupply.fixed(data).save(args.output / "fixed_0_1_supply.json")
    engines, weights = load_engines(data, args.responses, args.device)
    sources = [
        Path(__file__),
        ROOT / "minimum_supply.py",
        ROOT / "analysis/audit_load_support.py",
        ROOT / "capacity/environment.py",
        ROOT / "capacity/allocation.py",
        ROOT / "preprocess.py",
        ROOT / "train_response.py",
    ]
    provenance = dict(
        prepared_manifest=digest(data.folder / "manifest.json"),
        sources={str(path): digest(path) for path in sources},
        response_weights=weights,
        baseline=asdict(baseline),
        device=args.device,
        torch=torch.__version__,
        numpy=np.__version__,
        scope="Retrospective diagnostics; no causal validation or retraining.",
        interpretation=(
            "No counterfactual truth; a reliable minimum allocation " "is unidentified."
        ),
    )
    save_json(args.output / "provenance.json", provenance)
    save_json(args.output / "status.json", dict(stage="natural_accuracy"))
    natural = measured_errors(data, engines, args.device, args.output, args.batch_size)
    save_json(args.output / "status.json", dict(stage="interventions"))
    interventions = intervention_tests(
        data, engines, baseline, args.device, args.output
    )
    allocations = allocation_tests(data, baseline, args.output)
    create_report(data, baseline, natural, interventions, allocations, args.output)
    for path, expected in weights.items():
        require(digest(path) == expected, "Frozen weights changed during stress test.")
    for path, expected in provenance["sources"].items():
        require(digest(path) == expected, "Source changed during stress test.")
    save_json(
        args.output / "status.json",
        dict(stage="complete", elapsed_seconds=time.monotonic() - started),
    )
    print(
        json.dumps(
            dict(output=str(args.output), elapsed_seconds=time.monotonic() - started)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

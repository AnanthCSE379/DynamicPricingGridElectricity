"""Household-level analysis of paired frozen-model curtailment trajectories."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.analysis.consumer_recovery import (
    CAPS,
    WARMUP,
    RECOVERY,
    HOURS,
    cases,
)
from norway_energy_crisis.preprocess import PreparedData


COLORS = ("#b63a50", "#db8b20", "#2676b5", "#26906e")


def read(path):
    return json.loads(path.read_text())


def load(folder, engine, name):
    with np.load(folder / engine / f"{name}.npz") as saved:
        return {key: saved[key].copy() for key in saved.files}


def household_results(
    request, consumed, control, case, training_min, training_max, training_jump
):
    """Measure modeled return to the paired control, with near-zero safeguards."""
    end = WARMUP + case.cut_hours
    affected = np.any(
        consumed[:, :, WARMUP:end] < request[:, :, WARMUP:end] - 1e-8, axis=2
    )
    prediction = request[:, :, end : end + RECOVERY].astype(np.float64)
    reference = control[:, :, end : end + RECOVERY].astype(np.float64)
    difference = prediction - reference
    tolerance = np.maximum(0.1 * reference, 0.05)
    within = np.abs(difference) <= tolerance
    sustained = np.all(
        np.lib.stride_tricks.sliding_window_view(within, 3, axis=2), axis=3
    )
    ever = sustained.any(2)
    first = np.where(ever, sustained.argmax(2) + 3, np.nan)
    last_reference = reference[:, :, -24:].sum(2)
    last_prediction = prediction[:, :, -24:].sum(2)
    ratio = np.divide(
        last_prediction,
        last_reference,
        out=np.full_like(last_reference, np.nan),
        where=last_reference > 1e-6,
    )
    overshoot = np.any(difference > np.maximum(0.25 * reference, 0.25), axis=2)
    out_of_range = (prediction < training_min[None, :, None]) | (
        prediction > training_max[None, :, None]
    )
    jumps = np.abs(
        np.diff(consumed[:, :, end - 1 : end + RECOVERY].astype(float), axis=2)
    )
    jump_flags = jumps > training_jump[None, :, None]
    cut_feedback = consumed[:, :, WARMUP:end]
    below_support = np.any(cut_feedback < training_min[None, :, None], axis=2)
    lost = (request[:, :, WARMUP:end] - cut_feedback).sum(2, dtype=np.float64)
    surplus = np.maximum(difference, 0).sum(2)
    return dict(
        affected=affected,
        first_recovery_hours=first,
        day3_ratio=ratio,
        recovered_by_72h=ever,
        all_last24_within_band=np.all(within[:, :, -24:], axis=2),
        overshoot=overshoot,
        any_outside_training_range=np.any(out_of_range, axis=2),
        outside_training_range_fraction=out_of_range.mean(2),
        cut_below_training_minimum=below_support,
        recovery_jump_exceedance_fraction=jump_flags.mean(2),
        forced_lost_kwh=lost,
        modeled_surplus_kwh=surplus,
        post_release_request_difference_kwh=difference.sum(2),
    )


def summarize_values(values, mask):
    selected = np.asarray(values)[mask]
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return None
    return dict(
        mean=float(selected.mean()),
        p05=float(np.quantile(selected, 0.05)),
        median=float(np.median(selected)),
        p95=float(np.quantile(selected, 0.95)),
    )


def metrics(data, folder, manifest):
    training = np.asarray(data.demand[:, data.partitions == 0])
    minimum, maximum = training.min(1), training.max(1)
    usual_jump = np.quantile(np.abs(np.diff(training, axis=1)), 0.995, axis=1)
    results = []
    with (folder / "household_metrics.csv").open("w", newline="") as handle:
        writer = None
        for engine, plan in manifest["engines"].items():
            for case in cases():
                if case.cap_kw is None or case.name not in plan["cases"]:
                    continue
                saved = load(folder, engine, case.name)
                control = load(folder, engine, case.control)["requested"]
                request, consumed = saved["requested"], saved["consumed"]
                household = household_results(
                    request, consumed, control, case, minimum, maximum, usual_jump
                )
                exposed = household["affected"]
                end = WARMUP + case.cut_hours
                request_recovery = request[:, :, end : end + RECOVERY]
                control_recovery = control[:, :, end : end + RECOVERY]
                regions = {}
                for index, area in enumerate(data.state["areas"]):
                    region_mask = data.area_index == index
                    regions[area] = dict(
                        day3_aggregate_ratio=float(
                            request_recovery[:, region_mask, -24:].sum(dtype=float)
                            / control_recovery[:, region_mask, -24:].sum(dtype=float)
                        )
                    )
                row = dict(
                    engine=engine,
                    case=case.name,
                    cap_kw=case.cap_kw,
                    cut_hours=case.cut_hours,
                    high_price=case.high_price,
                    contexts=len(saved["origins"]),
                    household_contexts=int(exposed.size),
                    maximum_recovery_request_kwh=float(request_recovery.max()),
                    maximum_control_recovery_request_kwh=float(control_recovery.max()),
                    maximum_absolute_difference_from_control_kwh=float(
                        np.abs(request_recovery - control_recovery).max()
                    ),
                    first_recovery_hour_aggregate_ratio=float(
                        request_recovery[:, :, 0].sum(dtype=float)
                        / control_recovery[:, :, 0].sum(dtype=float)
                    ),
                    first_recovery_hour_mean_kwh=float(
                        request_recovery[:, :, 0].mean(dtype=float)
                    ),
                    last_restricted_hour_mean_consumed_kwh=float(
                        consumed[:, :, end - 1].mean(dtype=float)
                    ),
                    affected_household_contexts=int(exposed.sum()),
                    affected_percent=float(100 * exposed.mean()),
                    day3_aggregate_ratio=float(
                        request_recovery[:, :, -24:].sum(dtype=float)
                        / control_recovery[:, :, -24:].sum(dtype=float)
                    ),
                    exposed_day3_ratio=summarize_values(
                        household["day3_ratio"], exposed
                    ),
                    exposed_first_recovery_hours=summarize_values(
                        household["first_recovery_hours"], exposed
                    ),
                    exposed_recovered_by_72h_percent=(
                        float(100 * household["recovered_by_72h"][exposed].mean())
                        if exposed.any()
                        else None
                    ),
                    exposed_last24_in_band_percent=(
                        float(100 * household["all_last24_within_band"][exposed].mean())
                        if exposed.any()
                        else None
                    ),
                    exposed_still_below80_day3_percent=(
                        float(100 * (household["day3_ratio"][exposed] < 0.8).mean())
                        if exposed.any()
                        else None
                    ),
                    exposed_overshoot_percent=(
                        float(100 * household["overshoot"][exposed].mean())
                        if exposed.any()
                        else None
                    ),
                    exposed_cut_below_training_minimum_percent=(
                        float(
                            100
                            * household["cut_below_training_minimum"][exposed].mean()
                        )
                        if exposed.any()
                        else None
                    ),
                    exposed_outside_training_range_percent=(
                        float(
                            100
                            * household["any_outside_training_range"][exposed].mean()
                        )
                        if exposed.any()
                        else None
                    ),
                    exposed_jump_exceedance_fraction=summarize_values(
                        household["recovery_jump_exceedance_fraction"], exposed
                    ),
                    regions=regions,
                    day3_aggregate_ratio_by_context=[
                        float(a / b)
                        for a, b in zip(
                            request_recovery[:, :, -24:].sum(axis=(1, 2), dtype=float),
                            control_recovery[:, :, -24:].sum(axis=(1, 2), dtype=float),
                        )
                    ],
                    energy_ratio_by_recovery_day=[
                        float(
                            request_recovery[:, :, day * 24 : (day + 1) * 24].sum(
                                dtype=float
                            )
                            / control_recovery[:, :, day * 24 : (day + 1) * 24].sum(
                                dtype=float
                            )
                        )
                        for day in range(3)
                    ],
                    forced_lost_kwh=float(household["forced_lost_kwh"].sum()),
                    modeled_post_release_request_difference_kwh=float(
                        household["post_release_request_difference_kwh"].sum()
                    ),
                )
                if case.cut_hours == 1:
                    aggregate_ratio = request.sum(
                        axis=(0, 1), dtype=float
                    ) / control.sum(axis=(0, 1), dtype=float)
                    row["early_rebound_peak_ratio"] = float(
                        aggregate_ratio[WARMUP + 8 : WARMUP + 19].max()
                    )
                    echo = aggregate_ratio[WARMUP + 20 : WARMUP + 31]
                    row["next_day_dip_ratio"] = float(echo.min())
                    row["next_day_dip_hour_after_cut"] = int(echo.argmin() + 20)
                results.append(row)
                for day, origin in enumerate(saved["origins"]):
                    for household_index, household_id in enumerate(
                        data.preprocessor.ids
                    ):
                        record = dict(
                            engine=engine,
                            case=case.name,
                            origin=str(np.datetime64(int(data.times[origin]), "ns")),
                            household_id=household_id,
                            region=data.state["areas"][
                                int(data.area_index[household_index])
                            ],
                        )
                        for name, values in household.items():
                            value = values[day, household_index].item()
                            record[name] = (
                                None
                                if isinstance(value, float) and not np.isfinite(value)
                                else value
                            )
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(record))
                            writer.writeheader()
                        writer.writerow(record)
                print(f"Summarized {engine}/{case.name}", flush=True)
    price_effects = {}
    for engine, plan in manifest["engines"].items():
        if "price_only6" not in plan["cases"]:
            continue
        normal = load(folder, engine, "control")["requested"]
        pulse = load(folder, engine, "price_only6")["requested"]
        price_effects[engine] = dict(
            first_high_price_hour_ratio=float(
                pulse[:, :, WARMUP].sum(dtype=float)
                / normal[:, :, WARMUP].sum(dtype=float)
            ),
            first_normal_price_restored_hour_ratio=float(
                pulse[:, :, WARMUP + 6].sum(dtype=float)
                / normal[:, :, WARMUP + 6].sum(dtype=float)
            ),
            last24_ratio=float(
                pulse[:, :, -24:].sum(dtype=float) / normal[:, :, -24:].sum(dtype=float)
            ),
        )
    save_json(folder / "price_only_effects.json", price_effects)
    controls = {}
    for engine in manifest["engines"]:
        control = load(folder, engine, "control")["requested"]
        controls[engine] = dict(
            hourly_mean_by_day=[
                float(control[:, :, start : start + 24].mean(dtype=float))
                for start in (0, 24, 48, 72)
            ],
            last24_hourly_mean=float(control[:, :, -24:].mean(dtype=float)),
            minimum=float(control.min()),
            maximum=float(control.max()),
        )
    save_json(folder / "uncut_control_drift.json", controls)
    save_json(
        folder / "metrics.json",
        dict(
            results=results,
            recovery_definition=(
                "Three consecutive hours within max(10% of paired control, 0.05 kWh)."
            ),
            overshoot_definition=(
                "At least one post-release hour above control by max(25%, 0.25 kWh)."
            ),
            interpretation=(
                "Modeled comparisons, not real outage accuracy "
                "or independent household trials."
            ),
        ),
    )
    return results


def overview(folder):
    figure, axes = plt.subplots(2, 2, figsize=(14, 8), layout="constrained")
    control = load(folder, "ensemble", "control")
    x = np.arange(HOURS) - WARMUP
    for row, duration in enumerate((1, 6)):
        axes[row, 0].plot(
            x,
            control["consumed"][0].mean(0),
            color="black",
            linestyle="--",
            label="No cut",
        )
        for cap, color in zip(CAPS, COLORS):
            values = load(folder, "ensemble", f"cap{cap:g}_normal{duration}h")
            axes[row, 0].plot(
                x, values["consumed"][0].mean(0), color=color, label=f"{cap:g} kW cap"
            )
            ratio = values["requested"].sum(axis=(0, 1), dtype=float) / control[
                "requested"
            ].sum(axis=(0, 1), dtype=float)
            axes[row, 1].plot(x, 100 * ratio, color=color, label=f"{cap:g} kW")
        for axis in axes[row]:
            axis.axvspan(0, duration, color="grey", alpha=0.18)
            axis.axvline(duration, color="grey", linestyle=":")
            axis.set(xlabel="Hours from cut onset", xlim=(-24, 78))
            axis.legend(fontsize=8)
        axes[row, 0].set(
            title=f"{duration}-hour cut: first context, all 717 households",
            ylabel="Mean consumed power (kW)",
        )
        axes[row, 1].axhline(100, color="black", linestyle="--", alpha=0.5)
        axes[row, 1].set(
            title=f"{duration}-hour cut: all six contexts paired",
            ylabel="Predicted requests / uncut control (%)",
        )
    figure.savefig(folder / "normal_sequence_and_recovery.png", dpi=180)
    plt.close(figure)


def recovery_distribution(data, folder):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    training = np.asarray(data.demand[:, data.partitions == 0])
    minimum, maximum = training.min(1), training.max(1)
    jump = np.quantile(np.abs(np.diff(training, axis=1)), 0.995, axis=1)
    for axis, duration in zip(axes, (1, 6)):
        for engine, style in (("ensemble", "-"), ("reserved", "--")):
            control = load(folder, engine, "control")["requested"]
            for cap, color in zip(CAPS, COLORS):
                case = next(
                    case
                    for case in cases()
                    if case.name == f"cap{cap:g}_normal{duration}h"
                )
                values = load(folder, engine, case.name)
                result = household_results(
                    values["requested"],
                    values["consumed"],
                    control,
                    case,
                    minimum,
                    maximum,
                    jump,
                )
                times = result["first_recovery_hours"][result["affected"]]
                curve = [float(100 * np.mean(times <= hour)) for hour in range(1, 73)]
                axis.plot(
                    np.arange(1, 73),
                    curve,
                    color=color,
                    linestyle=style,
                    label=f"{cap:g}: {'42+7' if engine == 'ensemble' else '2026'}",
                )
        axis.set(
            xlabel="Hours after release",
            ylabel="Affected household-contexts (%)",
            title=f"First 3-hour return to control band: {duration}-hour cuts",
            ylim=(0, 102),
        )
        axis.legend(fontsize=8)
    figure.savefig(folder / "household_recovery_distribution.png", dpi=180)
    plt.close(figure)


def price_restore(folder):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
    control = load(folder, "ensemble", "control")["requested"].sum(
        axis=(0, 1), dtype=float
    )
    x = np.arange(HOURS) - WARMUP
    for axis, cap in zip(axes.ravel(), CAPS):
        for name, label, color, style in (
            ("control", "Normal price, no cut", "black", "--"),
            ("price_only6", "High price only, then normal", "#db8b20", "--"),
            (f"cap{cap:g}_normal6h", "Cut, price stays normal", "#2676b5", "-"),
            (f"cap{cap:g}_high6h", "Cut + high price, then normal", "#b63a50", "-"),
        ):
            values = load(folder, "ensemble", name)["requested"].sum(
                axis=(0, 1), dtype=float
            )
            axis.plot(
                x, 100 * values / control, color=color, linestyle=style, label=label
            )
        axis.axvspan(0, 6, color="grey", alpha=0.18)
        axis.axvline(6, color="grey", linestyle=":")
        axis.set(
            title=f"{cap:g} kW cap for six hours",
            xlim=(-6, 78),
            xlabel="Hours from cut onset; normal price restored at +6",
            ylabel="Predicted requests / normal uncut control (%)",
        )
        axis.legend(fontsize=8)
    figure.savefig(folder / "normal_price_restoration.png", dpi=180)
    plt.close(figure)


def household_examples(data, folder):
    training_mean = data.demand[:, data.partitions == 0].mean(1)
    selected = []
    for area in range(2):
        indices = np.flatnonzero(data.area_index == area)
        order = indices[np.argsort(training_mean[indices])]
        selected.extend(
            order[(np.array([0.25, 0.5, 0.75]) * (len(order) - 1)).astype(int)]
        )
    figure, axes = plt.subplots(2, 3, figsize=(15, 7), layout="constrained")
    control = load(folder, "ensemble", "control")["consumed"][0]
    normal = load(folder, "ensemble", "cap0.1_normal6h")["consumed"][0]
    high = load(folder, "ensemble", "cap0.1_high6h")["consumed"][0]
    x = np.arange(HOURS) - WARMUP
    for axis, index in zip(axes.ravel(), selected):
        axis.plot(x, control[index], color="black", linestyle="--", label="Uncut")
        axis.plot(x, normal[index], color="#2676b5", label="0.1 cap, normal price")
        axis.plot(x, high[index], color="#b63a50", label="0.1 cap + price pulse")
        axis.axvspan(0, 6, color="grey", alpha=0.18)
        axis.set(
            title=(
                f"{data.state['areas'][int(data.area_index[index])]} "
                f"household {data.preprocessor.ids[index]}"
            ),
            xlabel="Hours from cut onset",
            ylabel="Consumed power (kW)",
        )
        axis.legend(fontsize=7)
    figure.savefig(folder / "individual_households.png", dpi=180)
    plt.close(figure)
    save_json(
        folder / "example_households.json",
        dict(
            indices=[int(index) for index in selected],
            rule=(
                "25th/50th/75th training-mean-load household "
                "within each region; first context."
            ),
            selected_before_inspecting_recovery=True,
        ),
    )


def write_report(folder, provenance, results, verification):
    price = provenance["normal_price_nok_kwh"]
    lines = [
        "# Sudden curtailment and normal-price recovery",
        "",
        "## What was tested",
        "",
        f"All **717 households**, with normal price fixed at **{price:.5f} NOK/kWh**.",
        "Normal means the cohort-weighted training median, "
        "not a tariff label in the dataset.",
        "Start from a real 24-hour history and simulate another 24 hours normally.",
        "Then apply one-hour or six-hour caps of **0.05, 0.1, 0.75 or 1.25 kW**",
        "and follow at least 72 hours after release. "
        "Total simulated duration is 102 hours.",
        "Six starting contexts cover January, February and March, "
        "at midnight and 18:00.",
        "A restriction caps consumption at min(requested power, cap); "
        "it never increases",
        "a low-consuming household to the cap. The cut is an imposed overload-like",
        "scenario, not an observed grid failure or an electrically simulated feeder.",
        "",
        "Primary engines are the shared-history Transformer-42/7 ensemble "
        "and seed 2026.",
        "Individual seeds 42 and 7 repeat one-hour cuts on three contexts. Price-free",
        "seeds 42/7 check whether responses persist without any price input.",
        "",
        "The additional pricing experiment sets 2 NOK/kWh during a six-hour cut, then",
        "restores normal price. Its price-only control uses the same prices "
        "without a cut.",
        "This separates a price-history effect "
        "from the effect of restricted-load feedback.",
        "",
        "The diagnostic deliberately feeds restricted consumption "
        "into consumer history.",
        "The operational allocation simulator was not changed "
        "to use this feedback path.",
        "Future measured demand and temperature are never read. Origin temperature is",
        "held constant. Price schedules and calendar are declared experiment inputs.",
        "",
        "## How to read the results",
        "",
        "Every path has its own paired uncut control with identical starting inputs",
        "and identical prices. A ratio of 100% means matching that modeled control,",
        "not matching real household behavior. Day-three ratio uses the last 24 of",
        "the 72 hours after release. Contexts repeat the same households and are",
        "not independent consumer experiments. No significance claim is made.",
        "",
        "A household is affected only when its predicted request actually exceeds",
        "the cap during the restricted period. Recovery is the first three consecutive",
        "hours within max(10% of its paired control, 0.05 kWh). This is an explicit",
        "descriptive threshold. First recovery does not guarantee remaining recovered.",
        "",
        "## What happened in plain terms",
        "",
        "The model continued producing finite, nonnegative sequences. It did not",
        "reset to its uncut path when supply was restored. Lower caps and longer",
        "cuts caused larger, more persistent reductions relative to that path.",
        "There was a rise after release, then a new relative dip around the next",
        "day. This repeating pattern is visible even after a one-hour restriction.",
        "The effect remained in all response-model seeds; its magnitude differed.",
        "The following numbers describe the shared-history 42/7 ensemble:",
        "",
    ]
    for cap in CAPS:
        one = next(
            row
            for row in results
            if row["engine"] == "ensemble" and row["case"] == f"cap{cap:g}_normal1h"
        )
        six = next(
            row
            for row in results
            if row["engine"] == "ensemble" and row["case"] == f"cap{cap:g}_normal6h"
        )
        lines.append(
            f"- {cap:g} kW: day-three requests were "
            f"{100*one['day3_aggregate_ratio']:.1f}% of control after a one-hour cut "
            f"and {100*six['day3_aggregate_ratio']:.1f}% after a six-hour cut."
        )
    lines.extend(
        [
            "",
            "This is not a sequence of explosive or random numerical values. The",
            "problem is that a forced low consumption becomes evidence for continued",
            "low demand in later predictions. Smooth curves do not make that a valid",
            "model of real consumers returning from an outage.",
            "The price-only controls change the later trajectory "
            "far less than the cut.",
            "Returning to the declared normal price "
            "does not erase the low-load history.",
            "",
            "## Normal price throughout: aggregate results",
            "",
            "| Engine | Cut duration | Cap kW | Affected contexts | "
            "Day-1 requests/control | Day-3 requests/control |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        if row["engine"] in ("ensemble", "reserved") and not row["high_price"]:
            lines.append(
                f"| {row['engine']} | {row['cut_hours']}h | {row['cap_kw']:g} | "
                f"{row['affected_percent']:.1f}% | "
                f"{100*row['energy_ratio_by_recovery_day'][0]:.1f}% | "
                f"{100*row['day3_aggregate_ratio']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "![Whole-cohort sequence](normal_sequence_and_recovery.png)",
            "",
            "The left plots show the first starting context so a daily sequence",
            "is visible. The right plots combine all six paired contexts.",
            "",
            "Repeating dip after a one-hour cut, ensemble:",
            "",
            "| Cap kW | Highest request/control in hours 8–18 | "
            "Lowest in hours 20–30 | Hour of this later dip |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in results:
        if row["engine"] == "ensemble" and row["case"].endswith("normal1h"):
            lines.append(
                f"| {row['cap_kw']:g} | {100*row['early_rebound_peak_ratio']:.1f}% | "
                f"{100*row['next_day_dip_ratio']:.1f}% | "
                f"{row['next_day_dip_hour_after_cut']} |"
            )
    lines.extend(
        [
            "",
            "Although only the preceding 24 hours are visible at each step, an",
            "earlier cut can influence forecasts indefinitely because each forecast",
            "is fed into the next history. The observed daily dip is consistent with",
            "the network interpreting the cut as a consumption pattern. The precise",
            "internal cause was not identified by these experiments.",
            "",
            "Immediate return after releasing the one-hour restriction (ensemble):",
            "",
            "| Cap kW | Restricted-hour mean consumption kW | "
            "First unrestricted-hour mean request kW | First hour / uncut control |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in results:
        if row["engine"] == "ensemble" and row["case"].endswith("normal1h"):
            lines.append(
                f"| {row['cap_kw']:g} | "
                f"{row['last_restricted_hour_mean_consumed_kwh']:.3f} | "
                f"{row['first_recovery_hour_mean_kwh']:.3f} | "
                f"{100*row['first_recovery_hour_aggregate_ratio']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Household differences",
            "",
            "The table below includes only actually affected household-contexts.",
            "",
            "| Engine | Cut | Cap | First 3h return by 72h | "
            "Entire last24h in band | Still below 80% on day3 | "
            "Overshoot at any hour |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        if row["engine"] in ("ensemble", "reserved") and not row["high_price"]:
            lines.append(
                f"| {row['engine']} | {row['cut_hours']}h | {row['cap_kw']:g} | "
                f"{row['exposed_recovered_by_72h_percent']:.1f}% | "
                f"{row['exposed_last24_in_band_percent']:.1f}% | "
                f"{row['exposed_still_below80_day3_percent']:.1f}% | "
                f"{row['exposed_overshoot_percent']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "Overshoot means exceeding the paired control in at least one hour",
            "by more than max(25% of that control, 0.25 kWh). It is a numerical",
            "screen, not proof of unusual real behavior. Normal daily peaks can",
            "occur in both paths; the comparison always uses the same hour.",
            "",
            "![Household recovery distribution](household_recovery_distribution.png)",
            "",
            "![Individual household paths](individual_households.png)",
            "",
            "Examples are fixed by training consumption ranks (25th, 50th, 75th",
            "percentiles within each region), not chosen for dramatic outcomes.",
            "The CSV covers every household in every evaluated case, including",
            "unaffected households. Aggregate averages alone can hide persistent",
            "deficits for some households and overshoots for others.",
            "",
            "## Returning to normal price",
            "",
            "| Engine | Cap kW | Cut, normal price | Cut + price pulse, then normal |",
            "|---|---:|---:|---:|",
        ]
    )
    for engine in ("ensemble", "reserved"):
        for cap in CAPS:
            normal = next(
                row
                for row in results
                if row["engine"] == engine and row["case"] == f"cap{cap:g}_normal6h"
            )
            high = next(
                row
                for row in results
                if row["engine"] == engine and row["case"] == f"cap{cap:g}_high6h"
            )
            lines.append(
                f"| {engine} | {cap:g} | {100*normal['day3_aggregate_ratio']:.1f}% | "
                f"{100*high['day3_aggregate_ratio']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "Each percentage uses its matching price schedule's uncut control.",
            "The graph below instead shows all paths relative to the normal-price",
            "uncut path so the separate price-only effect remains visible.",
            "",
            "![Price restoration](normal_price_restoration.png)",
            "",
            "Restoring a price is not a signal telling this network that an outage",
            "ended. It sees prices, recent consumption, calendar, temperature and",
            "household descriptors; it has no outage indicator, unmet-load queue,",
            "appliance state or explicit requirement to replace lost energy.",
            "",
            "Price-only control (without any supply cut):",
            "",
            "| Engine | First high-price hour / normal | "
            "First restored-price hour / normal | Last24 / normal |",
            "|---|---:|---:|---:|",
        ]
    )
    for engine, effect in read(folder / "price_only_effects.json").items():
        lines.append(
            f"| {engine} | {100*effect['first_high_price_hour_ratio']:.2f}% | "
            f"{100*effect['first_normal_price_restored_hour_ratio']:.2f}% | "
            f"{100*effect['last24_ratio']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Input support and numerical screens",
            "",
            "| Engine | Cut | Cap | "
            "Affected cases with cut below household training minimum | "
            "Affected cases outside training range after release |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in results:
        if row["engine"] in ("ensemble", "reserved") and not row["high_price"]:
            lines.append(
                f"| {row['engine']} | {row['cut_hours']}h | {row['cap_kw']:g} | "
                f"{row['exposed_cut_below_training_minimum_percent']:.1f}% | "
                f"{row['exposed_outside_training_range_percent']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "These compare each household against its own training minimum and",
            "maximum. A scalar inside that range is not necessarily a familiar",
            "24-hour history or evidence of accurate counterfactual interpolation.",
            "Numerical stability and recovery toward a model control are different",
            "from verified behavior after real curtailment.",
            "",
            "The uncut control itself is an open-loop forecast and can drift.",
            "Its average hourly predictions by successive 24-hour blocks are:",
            "",
            "| Engine | Warmup day | Next day | Third day | Fourth day |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for engine, row in read(folder / "uncut_control_drift.json").items():
        means = " | ".join(f"{value:.3f}" for value in row["hourly_mean_by_day"])
        lines.append(f"| {engine} | {means} |")
    lines.extend(
        [
            "",
            "These are kWh per household-hour, numerically equal to average kW.",
            "They are not measured demand. Persistent deviation from a drifting",
            "control must not be described as actual energy lost by real consumers.",
            "Calendar also advances; a changing mean alone is not proof of error.",
            "",
            "## Individual response-seed sensitivity",
            "",
            "Individual seeds use the same three preselected contexts; their own",
            "uncut paths are their controls. The ensemble row above uses six contexts.",
            "",
            "| Seed | One-hour cap | Day-three modeled request/control |",
            "|---|---:|---:|",
        ]
    )
    for row in results:
        if row["engine"] in ("s42", "s7"):
            lines.append(
                f"| {row['engine']} | {row['cap_kw']:g} | "
                f"{100*row['day3_aggregate_ratio']:.1f}% |"
            )
    lines.extend(
        [
            "",
            "## Rechecks",
            "",
            "Timing, clamping and unaffected-household checks passed for "
            f"{len(verification['semantics'])} trajectories.",
            "The complete 717-household primary shock was repeated twice on all",
            "six contexts, using smaller inference batches than the main run.",
            "",
        ]
    )
    for attempt, row in verification["repeat"].items():
        lines.append(
            f"- Repeat {attempt}: maximum difference "
            f"{row['maximum_absolute_difference_kwh']:.8f} kWh."
        )
    lines.extend(
        [
            "",
            "An independent CPU path rebuilds raw histories through the existing",
            "Predictor interface, rather than reusing the new tensor recursion.",
            "It checks the uncut path, the 0.05 kW one-hour cut and the 1.25 kW",
            "six-hour cut with a high-price pulse, on households spanning both",
            "regions and training demand scales.",
            "",
        ]
    )
    for name, row in verification["independent_cpu"].items():
        lines.append(
            f"- {name}: {row['households']} households; maximum difference "
            f"{row['maximum_absolute_difference_kwh']:.8f} kWh."
        )
    if "software_checks" in verification:
        software = verification["software_checks"]
        analysis = verification["independent_analysis_checks"]
        scalar_checks = analysis["independent_scalar_recovery_checks"]
        lines.extend(
            [
                "",
                f"The repository check suite completed: {software['tests']} tests, "
                f"{software['skipped']} sandbox-related MPS unit-test skip.",
                "Main frozen-model simulations ran on MPS; "
                "the independent path used CPU.",
                f"A scalar time-loop separately checked {scalar_checks:,} "
                "household recovery records against the CSV.",
                "Independent aggregate energy reductions matched "
                "all 16 primary summaries.",
            ]
        )
    lines.extend(
        [
            "",
            "Price-free predictions were exactly invariant to the price pulse,",
            "both with and without the cut. They can still respond to the altered",
            "load history; price-free does not mean history-free.",
            "Frozen model/source fingerprints were checked before and after execution.",
            "These repetitions validate the calculations, not real outage behavior.",
            "",
            "## What this establishes",
            "",
            "This establishes what the frozen predictors do when consumption",
            "history is forcibly changed. It cannot establish whether real consumers",
            "would recover this way. Every counterfactual trajectory is unlabelled.",
            "The model was trained on observational consumption with rollout losses",
            "up to 24 hours; this longer test deliberately explores "
            "beyond that horizon.",
            "A return toward the uncut model path is not causal validation, proof",
            "of deferred load, or evidence that lost service was recovered.",
            "The follow-up months and seed 2026 were previously examined; this is",
            "a retrospective sensitivity study, not a fresh holdout experiment.",
            "Household descriptors come from a later survey and are assumed stable.",
            "Regional reanalysis weather is a proxy, not household measurements.",
            "",
            "## Files and reproduction",
            "",
            "- `metrics.json`: aggregate, regional and affected-household summaries.",
            "- `household_metrics.csv`: every household/context and recovery screen.",
            "- Engine subfolders: requested and consumed sequences "
            "in compressed arrays.",
            "- `verification.json`: repeated and independent implementation checks.",
            "- `provenance.json`: models, data, source fingerprints "
            "and exact settings.",
            "",
            "```sh",
            "python -m norway_energy_crisis.analysis.consumer_recovery --device mps",
            "python -m norway_energy_crisis.analysis.report_consumer_recovery",
            "```",
            "",
            "Use a new --output directory when rerunning simulations. No response",
            "or RL weights were retrained and no existing results were overwritten.",
            "",
        ]
    )
    (folder / "REPORT.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/consumer_recovery_v1"
    )
    args = parser.parse_args()
    verification = read(args.output / "verification.json")
    require(
        verification["passed"], "Independent verification must pass before reporting."
    )
    data = PreparedData(args.data)
    provenance = read(args.output / "provenance.json")
    require(
        digest(data.folder / "manifest.json") == provenance["prepared_manifest"],
        "Data changed.",
    )
    manifest = read(args.output / "trajectories.json")
    result = metrics(data, args.output, manifest)
    overview(args.output)
    recovery_distribution(data, args.output)
    price_restore(args.output)
    household_examples(data, args.output)
    write_report(args.output, provenance, result, verification)
    save_json(args.output / "report_provenance.json", dict(source=digest(__file__)))
    save_json(
        args.output / "status.json",
        dict(stage="complete", simulations_verified=True, report_complete=True),
    )
    print(args.output / "REPORT.md", flush=True)


if __name__ == "__main__":
    main()

"""Describe observed price/consumption associations without causal claims."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.preprocess import PreparedData


def within_groups(values, groups):
    """Remove group means, including calendar-day fixed effects."""
    values = np.asarray(values, dtype=np.float64)
    vector = values.ndim == 1
    values = values[:, None] if vector else values
    _, labels = np.unique(groups, return_inverse=True)
    counts = np.bincount(labels)
    totals = np.zeros((len(counts), values.shape[1]))
    np.add.at(totals, labels, values)
    result = values - totals[labels] / counts[labels, None]
    return result[:, 0] if vector else result


def adjusted_association(price, demand, controls, days, maxlags=168):
    """Partial slope via FWL and Bartlett/Newey-West hourly score covariance.

    Calendar-day and supplied nuisance controls are descriptive adjustments.
    The slope is not identified as a causal elasticity or intervention effect.
    """
    design = within_groups(controls, days)
    values = within_groups(np.column_stack((price, demand)), days)
    coefficients, _, rank, _ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coefficients
    x, y = residual.T
    denominator = x @ x
    require(denominator > 1e-12, "No independent price variation.")
    slope = float(x @ y / denominator)
    error = y - slope * x
    scores = x * error
    variance = scores @ scores
    maxlags = min(maxlags, len(x) - 1)
    for lag in range(1, maxlags + 1):
        variance += 2 * (1 - lag / (maxlags + 1)) * (scores[lag:] @ scores[:-lag])
    se = float(np.sqrt(max(variance, 0)) / denominator)
    return {
        "slope_kwh_per_nok_kwh": slope,
        "hac_standard_error": se,
        "approx_ci95": [slope - 1.96 * se, slope + 1.96 * se],
        "partial_correlation": float(np.corrcoef(x, y)[0, 1]),
        "remaining_price_std": float(x.std()),
        "nuisance_rank_after_day_demeaning": int(rank),
        "regional_hours": len(x),
        "hac_lags": maxlags,
    }


def controls_for(data, area, dates, indices):
    """Calendar and flexible regional temperature controls, not future values."""
    hour_of_week = np.asarray(dates.dayofweek * 24 + dates.hour)
    temperature = np.asarray(data.temperature[area, indices], dtype=float)
    center, scale = data.state["temperature_mean"], data.state["temperature_std"]
    t = (temperature - center) / scale
    knots = np.quantile(
        data.temperature[area, data.partitions == 0], [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    spline = [
        t,
        t * t,
        t * t * t,
        *[np.maximum(t - (k - center) / scale, 0) ** 3 for k in knots],
    ]
    return np.column_stack([np.eye(168)[hour_of_week], *spline])


def audit(output):
    output = Path(output)
    require(not output.exists(), f"Output exists: {output}")
    data = PreparedData(ROOT / "prepared")
    dates = pd.DatetimeIndex(data.times.astype("datetime64[ns]"))
    results, monthly, bands = {}, [], []
    splits = {
        "all": np.ones(len(dates), dtype=bool),
        "training": data.partitions == 0,
        "january": data.partitions == 1,
        "february_march": data.partitions == 2,
    }
    edges = np.asarray(data.state["price_band_edges"])
    for area, name in enumerate(data.state["areas"]):
        households = np.asarray(data.demand[data.area_index == area], dtype=float)
        demand = households.mean(0)
        price = np.asarray(data.prices[area], dtype=float)
        results[name] = {}
        for split, mask in splits.items():
            p, q = price[mask], demand[mask]
            indices = np.flatnonzero(mask)[24:]
            current_dates = dates[indices]
            day = current_dates.normalize().factorize()[0]
            controls = controls_for(data, area, current_dates, indices)
            # Every model uses the same hours, with observed lags wholly in split.
            basic = adjusted_association(price[indices], demand[indices], controls, day)
            with_lags = np.column_stack(
                (controls, demand[indices - 1], demand[indices - 24])
            )
            history = adjusted_association(
                price[indices], demand[indices], with_lags, day
            )
            centered_price = p - p.mean()
            covariance = households[:, mask] @ centered_price
            adjusted_price = within_groups(price[indices], day)
            z = within_groups(with_lags, day)
            adjusted_price -= z @ np.linalg.lstsq(z, adjusted_price, rcond=None)[0]
            household_partial_slopes = (
                households[:, indices]
                @ adjusted_price
                / (adjusted_price @ adjusted_price)
            )
            change_price = price[indices] - price[indices - 24]
            change_demand = demand[indices] - demand[indices - 24]
            results[name][split] = {
                "households": len(households),
                "regional_hours": len(p),
                "raw_correlation": float(np.corrcoef(p, q)[0, 1]),
                "raw_slope": float(
                    centered_price @ (q - q.mean()) / (centered_price @ centered_price)
                ),
                "mean_price": float(p.mean()),
                "mean_load_kwh": float(q.mean()),
                "households_positive_raw_association_fraction": float(
                    (covariance > 0).mean()
                ),
                "households_positive_partial_association_fraction": float(
                    (household_partial_slopes > 0).mean()
                ),
                "adjusted_calendar_temperature": basic,
                "adjusted_calendar_temperature_history": history,
                "same_hour_previous_day_change_correlation": float(
                    np.corrcoef(change_price, change_demand)[0, 1]
                ),
                "comparison_hours_after_history_requirement": len(indices),
            }
            require(
                np.isclose(
                    household_partial_slopes.mean(),
                    history["slope_kwh_per_nok_kwh"],
                    atol=1e-7,
                ),
                "Household and regional adjusted slopes disagree.",
            )
            for band in range(4):
                selected = np.searchsorted(edges, p, side="right") == band
                if selected.any():
                    bands.append(
                        {
                            "area": name,
                            "split": split,
                            "band": band,
                            "regional_hours": int(selected.sum()),
                            "mean_price": float(p[selected].mean()),
                            "mean_load_kwh": float(q[selected].mean()),
                            "mean_temperature_c": float(
                                data.temperature[area, mask][selected].mean()
                            ),
                        }
                    )
        frame = pd.DataFrame(
            {"price": price, "load": demand, "temperature": data.temperature[area]},
            index=dates,
        )
        month = frame.resample("MS").mean().reset_index(names="month")
        month["area"] = name
        monthly.append(month)
    output.mkdir(parents=True)
    monthly = pd.concat(monthly, ignore_index=True)
    monthly.to_csv(output / "monthly.csv", index=False)
    pd.DataFrame(bands).to_csv(output / "price_bands.csv", index=False)
    save_json(
        output / "results.json",
        {
            "regions": results,
            "price_band_edges": edges.tolist(),
            "prepared_manifest_sha256": digest(ROOT / "prepared/manifest.json"),
            "audit_source_sha256": digest(__file__),
            "scope": "Descriptive observational analysis; no causal identification.",
            "adjustments": (
                "Day/hour-of-week effects, temperature spline; then demand lags."
            ),
            "inference": (
                "Normal 95% HAC intervals, 168-hour Bartlett lags; no df correction."
            ),
        },
    )
    make_report(output, results, monthly)
    print(json.dumps(results, indent=2))


def make_report(output, results, monthly):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    labels = ["Training", "January", "Feb–Mar"]
    names = ["training", "january", "february_march"]
    for ax, (area, values) in zip(axes, results.items()):
        x = np.arange(3)
        ax.bar(x, [values[n]["raw_correlation"] for n in names], color="#247b91")
        ax.set_xticks(x, labels)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(
            title=area,
            ylabel="Observed price–consumption correlation",
            ylim=(-0.55, 0.55),
        )
        for i, n in enumerate(names):
            v = values[n]["raw_correlation"]
            ax.text(i, v + (0.025 if v >= 0 else -0.045), f"{v:+.3f}", ha="center")
    fig.suptitle("The historical relationship changes across periods")
    fig.tight_layout()
    fig.savefig(output / "historical_correlations.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for ax, area in zip(axes, results):
        frame = monthly[monthly.area.eq(area)]
        ax.plot(
            frame.month, frame.load, color="#247b91", marker="o", label="Consumption"
        )
        second = ax.twinx()
        second.plot(
            frame.month, frame.price, color="#d38629", marker="s", label="Price"
        )
        ax.set(ylabel="Mean household kWh/hour", title=area)
        second.set_ylabel("Mean price NOK/kWh")
        ax.legend(loc="upper left")
        second.legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output / "monthly_consumption_prices.png", dpi=170)
    plt.close(fig)
    rows = [
        "| Period | NO1 raw correlation | NO5 raw correlation |",
        "| --- | ---: | ---: |",
    ]
    for label, name in zip(labels, names):
        rows.append(
            f"| {label} | {results['NO1'][name]['raw_correlation']:+.4f} | "
            f"{results['NO5'][name]['raw_correlation']:+.4f} |"
        )
    adjusted = [
        "| Region | Calendar/weather slope | With history | 95% interval, history |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, regions in results.items():
        v = regions["training"]
        basic = v["adjusted_calendar_temperature"]["slope_kwh_per_nok_kwh"]
        h = v["adjusted_calendar_temperature_history"]
        adjusted.append(
            f"| {name} | {basic:+.5f} | {h['slope_kwh_per_nok_kwh']:+.5f} | "
            f"{h['approx_ci95']} |"
        )
    report = [
        "# Historical electricity prices and consumption",
        "",
        "Audited measured consumption for the 717 hourly-spot households: "
        "487 NO1 and 230 NO5, October 2020–March 2022. No model was retrained.",
        "",
        "## Recorded association",
        "",
        *rows,
        "",
        "![Historical correlations](historical_correlations.png)",
        "",
        "The training association is weak and differs by region. January is "
        "positive, while February-March is negative in both regions. Historical "
        "data do not support a universal higher-price/higher-consumption rule.",
        "",
        "## Training-period conditional checks",
        "",
        *adjusted,
        "",
        "Slopes are household kWh/hour per 1 NOK/kWh. These are descriptive "
        "partial associations, not causal effects or elasticities. Controls "
        "remove calendar-day means, hour-of-week effects and a flexible "
        "temperature function. The final check also includes regional mean "
        "consumption 1 and 24 hours earlier. Temperature spline knots and "
        "normalization use training values. Comparisons use the same hours.",
        "",
        "The lagged-demand adjustment concerns short-term residual association; "
        "it does not estimate total delayed price response. Price and consumption "
        "may share unmeasured causes, and wholesale prices may be related to "
        "broader demand. The data provide no randomized price intervention.",
        "",
        "Intervals use 168-hour Newey-West/Bartlett covariance and approximate "
        "normal inference. Each region-hour contributes one observation. "
        "Households sharing a price are not counted as independent price trials; "
        "the two regions are not treated as independent replicated experiments.",
        "",
        "## Monthly records",
        "",
        "![Monthly consumption and prices](monthly_consumption_prices.png)",
        "",
        "monthly.csv and price_bands.csv contain supporting aggregates. Both "
        "prices and consumption vary with time and weather; visual coincidence "
        "does not identify why they change.",
        "",
        "## Implication for the trained simulator",
        "",
        "The simulator's 99.7% positive endpoint response was measured by "
        "changing supplied prices at fixed constructed household histories. "
        "Historical correlations compare observed hours with varying histories. "
        "These are different statistics, not directly interchangeable percentages. "
        "The historical checks do not validate interpreting the model's positive "
        "price curve as a real consumer response. Conversely, these observational "
        "checks cannot establish the true causal direction either.",
        "",
        "The earlier suggestion that historical high-price/high-demand "
        "association explains the simulator should therefore be qualified: "
        "it appears in some periods but is not a consistent training-data pattern.",
    ]
    (output / "REPORT.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/historical_price_audit_v1"
    )
    audit(parser.parse_args().output)

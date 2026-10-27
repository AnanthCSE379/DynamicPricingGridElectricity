"""Shared reward mathematics; costs are proxies, not measured consumer welfare."""

import numpy as np

from norway_energy_crisis.clean_data import require

HORIZON = 48
NORMALIZATION = 7.0
REWARD_SCALE = 100.0
TOLERANCE = 0.05
VERSION = "regional_stress_smoothstep_5pct_fixed7_v1"


def bounded(deviation):
    """Return B(x) in [0, 1] for finite nonnegative deviations."""
    deviation = np.asarray(deviation, dtype=np.float64)
    require(
        np.isfinite(deviation).all() and (deviation >= 0).all(), "Invalid deviation."
    )
    return np.square(1 - 1 / (1 + deviation))


def state_weights(previous_load_ratio):
    """Increase grid priority from 1 to 5 using PRE-action regional demand.

    - Input: preceding regional power / capacity, dimensionless.
    - Output: one bounded grid weight per region.
    - Household energy and charge weights remain one; no moving denominator.
    """
    ratio = np.asarray(previous_load_ratio, dtype=np.float64)
    require(np.isfinite(ratio).all() and (ratio >= 0).all(), "Invalid load ratio.")
    danger = np.clip((ratio - 0.80) / 0.20, 0, 1)
    return 1 + 4 * danger**2 * (3 - 2 * danger)


def stress_cost(load_ratio):
    """Penalize proximity above 90% and overload above 100%, monotonically."""
    ratio = np.asarray(load_ratio, dtype=np.float64)
    require(np.isfinite(ratio).all() and (ratio >= 0).all(), "Invalid load ratio.")
    return 0.5 * (
        bounded(np.maximum(ratio - 0.90, 0) / 0.10)
        + bounded(np.maximum(ratio - 1, 0) / 0.10)
    )


def household_costs(
    energy_kwh,
    charge_nok,
    reference_energy_kwh,
    reference_charge_nok,
    training_median_price,
):
    """Score uncancelled household deviations over the complete episode.

    - Energy increases AND decreases beyond 5% incur a cost.
    - Charge increases beyond 5% incur a cost; savings get no positive reward.
    - Negative/zero reference charges use a fixed, positive denominator.
    """
    energy, charge, reference_energy, reference_charge = [
        np.asarray(value, dtype=np.float64)
        for value in (
            energy_kwh,
            charge_nok,
            reference_energy_kwh,
            reference_charge_nok,
        )
    ]
    require(
        all(
            np.isfinite(v).all()
            for v in (energy, charge, reference_energy, reference_charge)
        ),
        "Invalid accounts.",
    )
    require((energy >= 0).all() and (reference_energy >= 0).all(), "Negative energy.")
    require(training_median_price > 0, "Positive training median required.")
    energy_deviation = np.abs(energy - reference_energy) / np.maximum(
        reference_energy, 1e-6
    )
    charge_denominator = np.maximum.reduce(
        (
            np.abs(reference_charge),
            0.01 * training_median_price * reference_energy,
            np.full_like(energy, 1e-6),
        )
    )
    charge_increase = (charge - reference_charge) / charge_denominator
    return dict(
        energy=float(
            bounded(np.maximum(energy_deviation - TOLERANCE, 0) / TOLERANCE).mean()
        ),
        charge=float(
            bounded(np.maximum(charge_increase - TOLERANCE, 0) / TOLERANCE).mean()
        ),
    )


def score_trajectory(
    demand,
    prices,
    reference,
    market_prices,
    initial_demand,
    previous_prices,
    area_index,
    capacity_kw,
    training_median,
):
    """Independently score a full 48-hour trajectory and physical diagnostics.

    - Demand/reference: hours × households, kWh per one-hour interval.
    - Market prices: regions × hours; controlled price: hours, NOK/kWh.
    - Capacities: two independent regional kW limits; no allocation occurs.
    """
    demand, reference, prices, market_prices = [
        np.asarray(v, dtype=np.float64)
        for v in (demand, reference, prices, market_prices)
    ]
    require(
        demand.shape == reference.shape and demand.shape[0] == HORIZON,
        "Expected a complete 48-hour trajectory.",
    )
    require(
        demand.ndim == 2
        and prices.shape == (HORIZON,)
        and market_prices.shape == (2, HORIZON)
        and all(
            np.isfinite(v).all() for v in (demand, reference, prices, market_prices)
        )
        and (demand >= 0).all()
        and (reference >= 0).all(),
        "Invalid trajectory values or units/shapes.",
    )
    capacity = np.asarray(capacity_kw, dtype=float)
    require(
        capacity.shape == (2,) and np.isfinite(capacity).all() and (capacity > 0).all(),
        "Invalid capacities.",
    )
    regional = np.stack([demand[:, area_index == r].sum(1) for r in range(2)], 1)
    preceding = np.vstack(
        (
            [np.asarray(initial_demand)[area_index == r].sum() for r in range(2)],
            regional[:-1],
        )
    )
    ratios = regional / capacity
    weights = state_weights(preceding / capacity)
    unweighted = stress_cost(ratios)
    weighted_grid = float((weights * unweighted).mean())
    reference_energy = reference.sum(0)
    reference_charge = (reference * market_prices[area_index].T).sum(0)
    energy, charge = demand.sum(0), (demand * prices[:, None]).sum(0)
    costs = household_costs(
        energy, charge, reference_energy, reference_charge, training_median
    )
    household_energy_change = (energy - reference_energy) / np.maximum(
        reference_energy, 1e-6
    )
    costs["grid"] = weighted_grid
    cost = (weighted_grid + costs["energy"] + costs["charge"]) / NORMALIZATION
    require(-1e-12 <= cost <= 1 + 1e-12, "Cost outside its mathematical bound.")
    entry_jump = np.abs(prices[0] - np.asarray(previous_prices)[area_index])
    jumps = np.abs(np.diff(prices))
    stats = dict(
        overload_region_hours=int((ratios > 1).sum()),
        overload_fraction=float((ratios > 1).mean()),
        overload_kwh=float(np.maximum(regional - capacity, 0).sum()),
        max_load_ratio=float(ratios.max()),
        unweighted_grid=float(unweighted.mean()),
        mean_grid_weight=float(weights.mean()),
        peak_kw=float(regional.sum(1).max()),
        energy_change_fraction=float(
            (energy.sum() - reference_energy.sum()) / max(reference_energy.sum(), 1e-6)
        ),
        household_mean_abs_energy_change=float(
            (
                np.abs(energy - reference_energy) / np.maximum(reference_energy, 1e-6)
            ).mean()
        ),
        charge_change_nok=float(charge.sum() - reference_charge.sum()),
        charge_nok=float(charge.sum()),
        energy_kwh=float(energy.sum()),
        internal_switches=int((jumps > 1e-8).sum()),
        mean_jump_nok_kwh=float((entry_jump.mean() + jumps.sum()) / HORIZON),
        max_jump_nok_kwh=float(max(entry_jump.max(), jumps.max(initial=0))),
        high_price_fraction=float((prices > 1.20 + 1e-8).mean()),
        household_energy_reduced_over_5pct=float(
            (household_energy_change < -0.05).mean()
        ),
        household_energy_increased_over_5pct=float(
            (household_energy_change > 0.05).mean()
        ),
    )
    for percentile in (5, 50, 95):
        stats[f"household_energy_change_p{percentile}"] = float(
            np.percentile(household_energy_change, percentile)
        )
    for region in range(2):
        stats[f"region_{region}_peak_kw"] = float(regional[:, region].max())
        stats[f"region_{region}_overload_hours"] = int((ratios[:, region] > 1).sum())
    return dict(
        cost=float(cost),
        terms=costs,
        stats=stats,
        regional_kw=regional.tolist(),
        load_ratios=ratios.tolist(),
        grid_weights=weights.tolist(),
        unweighted_stress=unweighted.tolist(),
    )

"""Predeclared probes expose incentives without tuning weights to favor DQN."""

import numpy as np
import pandas as pd

from .environment import StressPriceEnv
from .objective import household_costs, score_trajectory, state_weights
from .evaluation import read_json
from .io import save_json


def mathematical_checks():
    """Verify suppression, inflation and cross-household cancellation are penalized."""
    reference = np.ones(2)
    examples = {
        name: household_costs(energy, np.ones(2), reference, np.ones(2), 0.5)
        for name, energy in {
            "unchanged": [1, 1],
            "suppressed": [0.5, 0.5],
            "inflated": [1.5, 1.5],
            "cancelled_aggregate": [2, 0],
        }.items()
    }
    assert examples["unchanged"]["energy"] == 0
    assert all(
        examples[k]["energy"] > 0
        for k in ("suppressed", "inflated", "cancelled_aggregate")
    )
    # Raising danger never weakens the fixed 1/7 coefficients of either guard.
    weights = state_weights([0.5, 0.8, 0.9, 1, 2]).tolist()
    assert weights[0] == 1 and weights[-1] == 5
    return dict(
        examples=examples,
        weights=weights,
        consumer_coefficient=1 / 7,
        note="No positive cycling/progress bonuses; all costs are nonnegative.",
    )


def price_support(data, grid):
    """Count distinct region-hours near actions, never duplicate household exposures."""
    times = pd.DatetimeIndex(data.times)
    training = data.partitions == 0
    result = {}
    for region, name in enumerate(data.state["areas"]):
        rows = []
        for price in grid:
            near = abs(np.asarray(data.prices[region]) - price) <= 0.05 + 1e-8
            rows.append(
                dict(
                    price_nok_kwh=float(price),
                    region_hours=int((near & training).sum()),
                    fraction=float(near[training].mean()),
                    by_quarter={
                        str(q): dict(
                            hours=int((near & training & (times.quarter == q)).sum()),
                            total_hours=int((training & (times.quarter == q)).sum()),
                        )
                        for q in (1, 2, 3, 4)
                    },
                )
            )
        result[name] = rows
    return dict(
        radius_nok_kwh=0.05,
        unit="distinct training region-hours",
        regions=result,
        note="Sparse actions remain legal; support is not causal identification.",
    )


def simulator_audit(data, providers, engines, scenarios, cache, output):
    """Compare fixed schedules on training/validation only, prior to timed fits."""
    path = output / "audit.json"
    if path.exists():
        return read_json(path)
    results = {"mathematical_checks": mathematical_checks(), "engines": {}}
    patterns = {
        "minimum": np.zeros(48, int),
        "maximum": np.full(48, 20),
        "alternating_extremes": np.tile([0, 20], 24),
        "first_six_high": np.r_[np.full(6, 20), np.zeros(42, int)],
        "last_six_high": np.r_[np.zeros(42, int), np.full(6, 20)],
        "last_six_low": np.r_[np.full(42, 20), np.zeros(6, int)],
    }
    for family, engine in engines.items():
        rows = []
        curves = []
        for split, provider in providers.items():
            env = StressPriceEnv(provider, engine, scenarios, cache)
            indices = (
                [0]
                if split == "train"
                else sorted(
                    set([0, len(provider.episodes) // 2, len(provider.episodes) - 1])
                )
            )
            for index in indices:
                for name, actions in patterns.items():
                    observation, _ = env.reset(
                        options={"episode_index": index, "scenario_index": 0}
                    )
                    if name == "minimum":
                        predicted = engine.candidates(
                            env.history, env.grid, env.episode.future_calendar[0]
                        )
                        curves.append(
                            dict(
                                split=split,
                                origin=env.episode.timestamp,
                                prices=env.grid.tolist(),
                                household_median_kwh=np.median(
                                    predicted, axis=1
                                ).tolist(),
                                regional_kwh=[
                                    [float(q[m].sum()) for m in env.area_masks]
                                    for q in predicted
                                ],
                                high_minus_low_positive_fraction=float(
                                    (predicted[-1] > predicted[0]).mean()
                                ),
                            )
                        )
                    for action in actions:
                        observation, _, _, _, _ = env.step(int(action))
                    for scenario in scenarios:
                        scored = score_trajectory(
                            env.demands,
                            env.prices,
                            env.reference,
                            env.market_prices,
                            env.episode.demand_history[:, -1],
                            env.episode.previous_prices,
                            env.area_index,
                            scenario.capacity_kw,
                            env.training_median,
                        )
                        rows.append(
                            dict(
                                split=split,
                                origin=env.episode.timestamp,
                                scenario=scenario.name,
                                schedule=name,
                                **scored
                            )
                        )
        results["engines"][family] = dict(
            schedules=rows,
            price_curves=curves,
            interpretation=(
                "Descriptive incentives and simulator behavior; "
                "not proof an RL policy exploited them."
            ),
        )
    results["price_support"] = price_support(data, np.r_[0.05, np.arange(1, 21) / 10])
    save_json(path, results)
    return results

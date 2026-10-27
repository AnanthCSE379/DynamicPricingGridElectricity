"""Learned-request simulation with independent regional allocation budgets."""

import copy
from dataclasses import replace
import hashlib

import gymnasium as gym
import numpy as np
import pandas as pd

from norway_energy_crisis.clean_data import require
from norway_energy_crisis.dqn.environment import (
    ReferenceCache,
    append_history,
    price_grid,
    price_transitions,
)
from .allocation import allocate, jain, service


def arrival_order(household_count, origin, hour, area, seed):
    """Reproducible synthetic order, independent of policy and Python hash salts."""
    identity = f"{origin}:{hour}:{area}:{seed}".encode()
    value = int.from_bytes(hashlib.sha256(identity).digest()[:8], "little")
    return np.random.default_rng(value).permutation(household_count)


class CapacityPriceEnv(gym.Env):
    """Prices affect requests; allocation never changes later consumer requests.

    Inputs come from safe EpisodeProvider snapshots. No measured future demand or
    temperature is retained. Reference schedules are model forecasts, not targets.
    """

    metadata = {"render_modes": []}
    feature_order = dict(
        household=(
            "24 normalized demand values",
            "24 reference demand values",
            "saved static descriptors",
            "six service accounts",
        ),
        accounts=(
            "requested",
            "delivered",
            "flexible_requested",
            "flexible_delivered",
            "essential_unserved",
            "daily_reference",
        ),
        temporal=(
            "NO1 demand",
            "NO5 demand",
            "NO1 price",
            "NO1 temperature",
            "NO5 price",
            "NO5 temperature",
            "saved six calendar features",
        ),
        known=(
            "NO1 price",
            "NO5 price",
            "saved six calendar features",
            "NO1 capacity",
            "NO5 capacity",
        ),
        context=(
            "elapsed",
            "remaining",
            "alpha",
            "preceding NO1 price",
            "preceding NO5 price",
            "NO1 capacity",
            "NO5 capacity",
            "essential unserved/reference energy",
        ),
    )

    def __init__(
        self,
        provider,
        engine,
        scenarios,
        cache_folder,
        method="fair",
        support=None,
        order_seed=0,
        minimum_supply=None,
    ):
        super().__init__()
        require(len(scenarios) > 0, "At least one capacity scenario required.")
        self.provider, self.engine = provider, engine
        self.scenarios = tuple(scenarios)
        self.method, self.order_seed = method, order_seed
        self.support = support
        self.minimum_supply = minimum_supply
        if minimum_supply is not None:
            minimum_supply.validate_provider(provider)
        self.grid = price_grid(2.0)
        self.reference_cache = ReferenceCache(provider, engine, cache_folder)
        self.area_index = provider.area_index.copy()
        self.area_masks = [self.area_index == area for area in range(2)]
        require(
            all(mask.any() for mask in self.area_masks), "Both regions are required."
        )
        self.first_in_area = [int(np.flatnonzero(mask)[0]) for mask in self.area_masks]
        scales = provider.preprocessor.scales
        self.regional_scale = np.array([scales[mask].sum() for mask in self.area_masks])
        households, static_size = provider.preprocessor.static.shape
        # Six accounts: requested, delivered, flexible requested/served, essential
        # unserved, and the full-day reference. All reuse saved demand scales.
        shapes = dict(
            demand=(households, 24),
            reference=(households, 24),
            static=(households, static_size),
            accounts=(households, 6),
            temporal=(24, 12),
            known=(24, 10),
            context=(8,),
        )
        self.observation_space = gym.spaces.Dict(
            {
                name: gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)
                for name, shape in shapes.items()
            }
        )
        self.action_space = gym.spaces.Discrete(len(self.grid))
        self.done = True

    def action_masks(self):
        """Every bounded price is legal; overload is not an invalid price action."""
        return np.ones(len(self.grid), dtype=bool)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.index = int(
            options.get(
                "episode_index", self.np_random.integers(len(self.provider.episodes))
            )
        )
        self.scenario_index = int(
            options.get("scenario_index", self.np_random.integers(len(self.scenarios)))
        )
        require(
            0 <= self.index < len(self.provider.episodes)
            and 0 <= self.scenario_index < len(self.scenarios),
            "Invalid episode/scenario index.",
        )
        self.scenario = self.scenarios[self.scenario_index]
        if "alpha" in options:
            self.scenario = replace(self.scenario, alpha=float(options["alpha"]))
        self.episode = self.provider.episodes[self.index]
        self.history = self.provider.features(self.episode)
        self.reference = np.array(
            self.reference_cache.get(self.index), dtype=np.float64, copy=True
        )
        self.reference_energy = self.reference.sum(0)
        households = self.provider.households
        self.requested_energy = np.zeros(households, dtype=np.float64)
        self.delivered_energy = np.zeros(households, dtype=np.float64)
        self.flexible_requests = np.zeros(households, dtype=np.float64)
        self.flexible_delivered = np.zeros(households, dtype=np.float64)
        self.essential_unserved = np.zeros(households, dtype=np.float64)
        self.charges = np.zeros(households, dtype=np.float64)
        self.previous_regional = self.episode.prices_history[:, -1].copy()
        self.prices, self.hourly = [], []
        self.raw_predictions = []
        self.floor_bindings = 0
        self.hour, self.done = 0, False
        self.trace = None
        return self.observation(), dict(
            origin=self.episode.timestamp, scenario=self.scenario.name
        )

    def observation(self):
        preprocessor = self.provider.preprocessor
        scales = preprocessor.scales.astype(np.float64)
        regional = (
            self.history[self.first_in_area, :, 1:3].transpose(1, 0, 2).reshape(24, 4)
        )
        aggregate = np.stack(
            [
                (self.history[mask, :, 0] * scales[mask, None]).sum(0)
                / self.regional_scale[area]
                for area, mask in enumerate(self.area_masks)
            ],
            axis=1,
        )
        temporal = np.concatenate((aggregate, regional, self.history[0, :, 3:]), axis=1)
        capacities = (
            np.broadcast_to(self.scenario.capacity_kw, (24, 2)) / self.regional_scale
        )
        known = np.concatenate(
            (
                preprocessor.price(self.episode.future_prices).T,
                self.episode.future_calendar,
                capacities,
            ),
            axis=1,
        )
        accounts = np.stack(
            (
                self.requested_energy,
                self.delivered_energy,
                self.flexible_requests,
                self.flexible_delivered,
                self.essential_unserved,
                self.reference_energy,
            ),
            axis=1,
        ) / (24 * scales[:, None])
        result = dict(
            demand=self.history[:, :, 0],
            reference=self.reference.T / scales[:, None],
            static=preprocessor.static,
            accounts=accounts,
            temporal=temporal,
            known=known,
            context=np.r_[
                self.hour / 24,
                (24 - self.hour) / 24,
                self.scenario.alpha,
                preprocessor.price(self.previous_regional),
                np.asarray(self.scenario.capacity_kw) / self.regional_scale,
                self.essential_unserved.sum() / max(self.reference_energy.sum(), 1e-6),
            ],
        )
        result = {
            name: np.array(values, dtype=np.float32, copy=True)
            for name, values in result.items()
        }
        require(
            all(np.isfinite(values).all() for values in result.values()),
            "Nonfinite observation.",
        )
        return result

    def attach_trace(self, prices, raw_predictions):
        """Reuse a fingerprinted model-generated schedule, never measured targets."""
        prices = np.asarray(prices, dtype=np.float64)
        predictions = np.asarray(raw_predictions, dtype=np.float32)
        require(
            self.hour == 0
            and prices.shape == (24,)
            and predictions.shape == self.reference.shape
            and np.isfinite(predictions).all()
            and (predictions >= 0).all(),
            "Invalid model-generated trace.",
        )
        self.trace = (prices.copy(), predictions.copy())

    def market_action(self):
        weights = np.bincount(self.area_index, minlength=2) / self.provider.households
        price = float(weights @ self.episode.future_prices[:, self.hour])
        return int(np.argmin(np.abs(self.grid - price)))

    def rule_action(self):
        market = float(self.grid[self.market_action()])
        prediction = self.engine.predict(
            self.history, market, self.episode.future_calendar[self.hour]
        )
        requested = np.maximum(
            prediction, self.scenario.alpha * self.reference[self.hour]
        )
        ratios = [
            float(requested[mask].sum()) / max(self.scenario.budget_kwh[area], 1e-12)
            for area, mask in enumerate(self.area_masks)
        ]
        price = np.clip(market + max(max(ratios) - 1.0, 0.0) * 1.95, 0.05, 2.0)
        return int(np.argmin(np.abs(self.grid - price)))

    def step(self, action):
        require(
            not self.done and self.action_space.contains(action),
            "Invalid action or finished day.",
        )
        price = float(self.grid[int(action)])
        if self.trace is not None:
            require(
                abs(self.trace[0][self.hour] - price) < 1e-9,
                "Trace and action schedule disagree.",
            )
            predicted = self.trace[1][self.hour]
        elif self.engine.is_price_free:
            # Price-free checkpoints ignore past/candidate prices. By induction,
            # requests equal the reference under every schedule (alpha <= 1).
            predicted = self.reference[self.hour]
        else:
            predicted = self.engine.predict(
                self.history, price, self.episode.future_calendar[self.hour]
            )
        essential = self.scenario.alpha * self.reference[self.hour]
        self.raw_predictions.append(np.asarray(predicted, dtype=np.float32).copy())
        requested = np.maximum(np.asarray(predicted, dtype=np.float64), essential)
        self.floor_bindings += int(np.count_nonzero(predicted < essential))
        delivered = np.zeros_like(requested)
        essential_shortage = np.zeros_like(requested)
        flexible_served = np.zeros_like(requested)
        infeasible = []
        for area, mask in enumerate(self.area_masks):
            allocator = allocate
            extra = {}
            if self.minimum_supply is not None:
                allocator = self.minimum_supply.allocate
                extra = dict(
                    household_indices=np.flatnonzero(mask),
                    interval_hours=self.scenario.interval_hours,
                )
            outcome = allocator(
                requested[mask],
                essential[mask],
                self.scenario.budget_kwh[area],
                self.flexible_requests[mask],
                self.flexible_delivered[mask],
                self.method,
                arrival_order(
                    int(mask.sum()),
                    self.episode.origin,
                    self.hour,
                    area,
                    self.order_seed,
                ),
                **extra,
            )
            delivered[mask] = outcome.delivered
            essential_shortage[mask] = outcome.essential_unserved
            flexible_served[mask] = outcome.flexible_delivered
            infeasible.append(outcome.essential_infeasible)
        self.requested_energy += requested
        self.delivered_energy += delivered
        self.flexible_requests += requested - essential
        self.flexible_delivered += flexible_served
        self.essential_unserved += essential_shortage
        self.charges += delivered * price
        # Requests, not curtailed deliveries, determine subsequent model histories.
        self.history = append_history(
            self.history,
            requested,
            price,
            self.episode.future_calendar[self.hour],
            self.provider.preprocessor,
        )
        self.hourly.append(
            dict(
                reference=[
                    float(self.reference[self.hour, mask].sum())
                    for mask in self.area_masks
                ],
                requested=[float(requested[mask].sum()) for mask in self.area_masks],
                delivered=[float(delivered[mask].sum()) for mask in self.area_masks],
                essential=[float(essential[mask].sum()) for mask in self.area_masks],
                essential_unserved=[
                    float(essential_shortage[mask].sum()) for mask in self.area_masks
                ],
                essential_infeasible=infeasible,
            )
        )
        self.prices.append(price)
        self.previous_regional = np.full(2, price)
        self.hour += 1
        self.done = self.hour == 24
        info = self.daily_result() if self.done else dict(price_nok_kwh=price)
        reward = -100 * info["loss"] if self.done else 0.0
        return self.observation(), reward, self.done, False, info

    def daily_result(self):
        require(self.done, "Daily metrics require a completed episode.")
        metrics = service(self.reference_energy, self.delivered_energy)
        ratios = metrics.pop("service_ratios")
        active = self.flexible_requests > 1e-6
        metrics.update(
            flexible_eligible_households=int(active.sum()),
            flexible_fairness=jain(
                self.flexible_delivered[active] / self.flexible_requests[active]
            ),
            forced_unserved_kwh=float(
                (self.requested_energy - self.delivered_energy).sum()
            ),
            essential_unserved_kwh=float(self.essential_unserved.sum()),
            requested_change_kwh=float(
                (self.requested_energy - self.reference_energy).sum()
            ),
            requested_change_fraction=float(
                (self.requested_energy.sum() - self.reference_energy.sum())
                / max(self.reference_energy.sum(), 1e-6)
            ),
            requested_household_abs_change_kwh=float(
                np.abs(self.requested_energy - self.reference_energy).mean()
            ),
            requested_kwh=float(self.requested_energy.sum()),
            delivered_kwh=float(self.delivered_energy.sum()),
            reference_kwh=float(self.reference_energy.sum()),
            charge_nok=float(self.charges.sum()),
            reference_charge_nok=float(
                (self.reference * self.episode.future_prices[self.area_index].T).sum()
            ),
            floor_binding_fraction=self.floor_bindings
            / (24 * self.provider.households),
            service_ratio_quantiles=(
                np.quantile(ratios, [0, 0.1, 0.5, 0.9, 1]).tolist() if ratios else []
            ),
        )
        for area, mask in enumerate(self.area_masks):
            regional = service(self.reference_energy[mask], self.delivered_energy[mask])
            regional.pop("service_ratios")
            metrics[f"region_{area}"] = dict(
                **regional,
                forced_unserved_kwh=float(
                    (self.requested_energy[mask] - self.delivered_energy[mask]).sum()
                ),
                essential_unserved_kwh=float(self.essential_unserved[mask].sum()),
                charge_nok=float(self.charges[mask].sum()),
            )
        requested = np.asarray([row["requested"] for row in self.hourly])
        delivered = np.asarray([row["delivered"] for row in self.hourly])
        budgets = self.scenario.budget_kwh
        metrics.update(
            preallocation_overload_region_hours=int((requested > budgets + 1e-8).sum()),
            postallocation_violation_region_hours=int(
                (delivered > budgets + 1e-8).sum()
            ),
            essential_infeasible_region_hours=int(
                np.asarray([row["essential_infeasible"] for row in self.hourly]).sum()
            ),
            essential_infeasible_hours=int(
                np.any(
                    [row["essential_infeasible"] for row in self.hourly], axis=1
                ).sum()
            ),
        )
        _, switching = price_transitions(
            self.prices, self.episode.prices_history[:, -1], self.area_index
        )
        metrics.update(switching)
        if self.support is not None:
            quarter = (pd.Timestamp(self.episode.timestamp).month - 1) // 3 + 1
            grid = np.asarray(self.support["grid"])
            indices = np.abs(np.asarray(self.prices)[:, None] - grid).argmin(1)
            fractions = [
                np.mean(np.asarray(periods[f"Q{quarter}"]["percentage"])[indices] < 1.0)
                for periods in self.support["regions"].values()
            ]
            weights = np.bincount(self.area_index, minlength=2) / len(self.area_index)
            metrics["sparse_price_fraction"] = float(weights @ fractions)
        return dict(
            origin=self.episode.timestamp,
            origin_index=self.episode.origin,
            scenario=self.scenario.name,
            alpha=self.scenario.alpha,
            method=self.method,
            capacity_kw=list(self.scenario.capacity_kw),
            prices=list(self.prices),
            hourly=copy.deepcopy(self.hourly),
            minimum_supply_signature=(
                self.minimum_supply.signature if self.minimum_supply else None
            ),
            **metrics,
        )

    def snapshot(self):
        keys = (
            "index",
            "scenario_index",
            "hour",
            "history",
            "requested_energy",
            "delivered_energy",
            "flexible_requests",
            "flexible_delivered",
            "essential_unserved",
            "charges",
            "previous_regional",
            "prices",
            "raw_predictions",
            "hourly",
            "floor_bindings",
            "done",
        )
        return dict(
            values=copy.deepcopy({name: getattr(self, name) for name in keys}),
            minimum_supply_signature=(
                self.minimum_supply.signature if self.minimum_supply else None
            ),
            alpha=self.scenario.alpha,
            rng=copy.deepcopy(self.np_random.bit_generator.state),
            action_rng=copy.deepcopy(self.action_space.np_random.bit_generator.state),
        )

    def restore(self, state):
        require(
            state.get("minimum_supply_signature")
            == (self.minimum_supply.signature if self.minimum_supply else None),
            "Snapshot minimum-supply baseline mismatch.",
        )
        self.reset(
            options=dict(
                episode_index=state["values"]["index"],
                scenario_index=state["values"]["scenario_index"],
                alpha=state["alpha"],
            )
        )
        for name, value in state["values"].items():
            setattr(self, name, copy.deepcopy(value))
        self.np_random.bit_generator.state = state["rng"]
        self.action_space.np_random.bit_generator.state = state["action_rng"]

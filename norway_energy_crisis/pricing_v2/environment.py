"""48-hour price decisions with daily market-information release and no allocation."""

import copy
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from norway_energy_crisis.clean_data import require, digest
from norway_energy_crisis.preprocess import calendar
from norway_energy_crisis.response_v2.data import origins_for
from norway_energy_crisis.dqn.environment import (
    append_history,
    atomic_npz,
    fingerprint,
    price_grid,
)
from .objective import (
    HORIZON,
    NORMALIZATION,
    REWARD_SCALE,
    household_costs,
    score_trajectory,
    state_weights,
    stress_cost,
)


@dataclass(frozen=True)
class Episode:
    """Only starting measurements and deterministic calendar, never future labels."""

    origin: int
    timestamp: str
    history: np.ndarray
    demand_history: np.ndarray
    previous_prices: np.ndarray
    future_calendar: np.ndarray


class EpisodeProvider:
    """Copy safe snapshots; release regional prices in 24-hour chunks.

    - Demand/weather are read only before the episode origin.
    - Calendar is deterministic; future measured weather is never retained.
    - The environment requests day two only after completing hour 24.
    """

    def __init__(self, data, split, selection_file=None):
        require(split in ("train", "validation", "test"), "Unknown partition.")
        if split == "test":
            import json

            require(
                selection_file is not None and Path(selection_file).exists(),
                "Freeze selection before follow-up testing.",
            )
            require(
                json.loads(Path(selection_file).read_text()).get("frozen_before_test")
                is True,
                "Selection not frozen.",
            )
        self.preprocessor = data.preprocessor
        self.households, self.area_index = data.households, data.area_index.copy()
        self.manifest_hash = digest(data.folder / "manifest.json")
        self.split = split
        self.episodes, self._price_days = [], []
        origins = origins_for(
            data.partitions, ("train", "validation", "test").index(split), HORIZON, True
        )
        for origin in origins:
            past = np.arange(origin - 24, origin)
            history_demand = np.array(data.demand[:, past], copy=True)
            history_prices = np.array(data.prices[:, past], copy=True)
            history_temperature = np.array(data.temperature[:, past], copy=True)
            features = self.preprocessor.history_features(
                history_demand,
                history_prices[self.area_index],
                history_temperature[self.area_index],
                data.times[past],
                np.arange(self.households),
            )
            future_calendar = calendar(data.times[origin : origin + HORIZON])
            self.episodes.append(
                Episode(
                    int(origin),
                    str(np.datetime64(int(data.times[origin]), "ns")),
                    features,
                    history_demand,
                    history_prices[:, -1],
                    future_calendar,
                )
            )
            self._price_days.append(
                tuple(
                    np.array(data.prices[:, origin + d : origin + d + 24], copy=True)
                    for d in (0, 24)
                )
            )
        require(self.episodes, "No eligible episodes.")
        self.episodes = tuple(self.episodes)

    def release_prices(self, index, day):
        require(day in (0, 1), "Only two market days are supported.")
        return self._price_days[index][day].copy()

    def features(self, episode):
        return episode.history.copy()


class ReferenceCache:
    (
        "Cache independent reference days; controlled histories never "
        "enter this branch."
    )

    def __init__(self, provider, engine, folder):
        self.provider, self.engine = provider, engine
        identity = fingerprint(
            dict(
                engine=engine.signature,
                data=provider.manifest_hash,
                split=provider.split,
                version="reference48_daily_v1",
            )
        )
        self.folder = Path(folder) / identity
        self.memory = {}

    def get(self, index, day):
        key = index, day
        if key in self.memory:
            return self.memory[key]
        episode = self.provider.episodes[index]
        path = self.folder / f"{episode.origin}_day{day}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                demand, final_history = saved["demand"], saved["final_history"]
        else:
            history = (
                episode.history.copy() if day == 0 else self.get(index, 0)[1].copy()
            )
            market = self.provider.release_prices(index, day)
            values = []
            for t in range(24):
                price = market[self.provider.area_index, t]
                target_calendar = episode.future_calendar[day * 24 + t]
                demand = self.engine.predict(history, price, target_calendar)
                values.append(demand)
                history = append_history(
                    history, demand, price, target_calendar, self.provider.preprocessor
                )
            demand, final_history = np.asarray(values), history
            atomic_npz(path, demand=demand, final_history=final_history)
        require(
            demand.shape == (24, self.provider.households)
            and np.isfinite(demand).all()
            and (demand >= 0).all(),
            "Invalid reference cache.",
        )
        self.memory[key] = demand, final_history
        return demand, final_history


class StressPriceEnv(gym.Env):
    """Price consumption; every household's modeled consumption is preserved.

    - 48 actions, no truncation, allocation, floor, backlog or recovery period.
    - Weights use preceding demand; costs use this action's resulting demand.
    - Only released market/reference information enters policy observations.
    """

    metadata = {"render_modes": []}
    allocation_enabled = False

    def __init__(self, provider, engine, scenarios, cache_folder):
        super().__init__()
        self.provider, self.engine, self.scenarios = provider, engine, tuple(scenarios)
        require(
            self.scenarios
            and all(
                np.isfinite(s.capacity_kw).all() and min(s.capacity_kw) > 0
                for s in self.scenarios
            ),
            "Positive regional capacities required.",
        )
        self.grid = price_grid(2.0)
        self.reference_cache = ReferenceCache(provider, engine, cache_folder)
        self.area_index = provider.area_index
        self.area_masks = [self.area_index == r for r in range(2)]
        self.first_in_area = [int(np.flatnonzero(mask)[0]) for mask in self.area_masks]
        self.regional_scale = np.array(
            [provider.preprocessor.scales[m].sum() for m in self.area_masks]
        )
        self.training_median = float(provider.preprocessor.state["price_band_edges"][1])
        shapes = dict(
            demand=(provider.households, 24),
            regional=(2, 24, 2),
            calendar=(24, 6),
            known_prices=(2, 48),
            known_calendar=(48, 6),
            known_reference=(2, 48),
            known_mask=(48,),
            accounts=(4, provider.households),
            context=(13,),
        )
        self.feature_shapes = shapes
        self.observation_space = gym.spaces.Dict(
            {
                name: gym.spaces.Box(
                    -np.inf, np.inf, shape=(int(np.prod(shape)),), dtype=np.float32
                )
                for name, shape in shapes.items()
            }
        )
        self.action_space = gym.spaces.Discrete(len(self.grid))
        self.done = True

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
            "Invalid episode/scenario.",
        )
        self.episode = self.provider.episodes[self.index]
        self.scenario = self.scenarios[self.scenario_index]
        self.history = self.episode.history.copy()
        self.reference = np.zeros((48, self.provider.households), dtype=np.float32)
        self.market_prices = np.zeros((2, 48), dtype=np.float64)
        self.known_hours = 0
        self._release(0)
        self.energy = np.zeros(self.provider.households, dtype=np.float64)
        self.charge = self.energy.copy()
        self.previous_regional = self.episode.previous_prices.copy()
        self.demands, self.prices, self.weights, self.stresses = [], [], [], []
        self.hour, self.done = 0, False
        return self.observation(), dict(
            origin=self.episode.timestamp, scenario=self.scenario.name
        )

    def _release(self, day):
        """Reveal one market day and its independent modeled reference together."""
        start = day * 24
        self.market_prices[:, start : start + 24] = self.provider.release_prices(
            self.index, day
        )
        self.reference[start : start + 24] = self.reference_cache.get(self.index, day)[
            0
        ]
        self.known_hours = start + 24

    def regional_power(self, household_demand):
        return np.array(
            [
                np.asarray(household_demand, dtype=np.float64)[m].sum()
                for m in self.area_masks
            ]
        )

    def preceding_demand(self):
        return self.demands[-1] if self.demands else self.episode.demand_history[:, -1]

    def observation(self):
        prep = self.provider.preprocessor
        scale = prep.scales.astype(np.float64)
        known = np.arange(48) < self.known_hours
        prices = np.zeros_like(self.market_prices, dtype=np.float32)
        prices[:, known] = prep.price(self.market_prices[:, known])
        reference_energy = self.reference[: self.known_hours].astype(np.float64).sum(0)
        reference_charge = (
            self.reference[: self.known_hours].astype(np.float64)
            * self.market_prices[self.area_index, : self.known_hours].T
        ).sum(0)
        preceding = self.regional_power(self.preceding_demand())
        ratios = preceding / np.asarray(self.scenario.capacity_kw)
        values = dict(
            demand=self.history[:, :, 0],
            regional=self.history[self.first_in_area, :, 1:3],
            calendar=self.history[0, :, 3:],
            known_prices=prices,
            known_calendar=self.episode.future_calendar,
            known_reference=np.stack(
                [
                    self.reference[:, m].sum(1) / self.regional_scale[r]
                    for r, m in enumerate(self.area_masks)
                ]
            ),
            known_mask=known.astype(np.float32),
            accounts=np.stack(
                (
                    self.energy / (48 * scale),
                    self.charge / (48 * scale * 2),
                    reference_energy / (48 * scale),
                    reference_charge / (48 * scale * 2),
                )
            ),
            context=np.r_[
                self.hour / 48,
                (48 - self.hour) / 48,
                prep.price(self.previous_regional),
                np.asarray(self.scenario.capacity_kw) / self.regional_scale,
                ratios,
                state_weights(ratios),
                self.known_hours / 48,
                float(self.scenario.quantile),
                1.0,
            ],
        )
        observation = {
            k: np.asarray(v, dtype=np.float32).ravel().copy() for k, v in values.items()
        }
        require(
            all(np.isfinite(v).all() for v in observation.values()),
            "Nonfinite policy observation.",
        )
        return observation

    def action_masks(self):
        return np.ones(len(self.grid), dtype=bool)

    def market_action(self):
        weights = np.bincount(self.area_index, minlength=2) / self.provider.households
        market = weights @ self.market_prices[:, self.hour]
        return int(np.argmin(np.abs(self.grid - market)))

    def rule_action(self):
        price = self.grid[self.market_action()]
        predicted = self.engine.predict(
            self.history, price, self.episode.future_calendar[self.hour]
        )
        ratio = self.regional_power(predicted) / np.asarray(self.scenario.capacity_kw)
        chosen = np.clip(price + max(float(ratio.max()) - 1, 0) * 1.95, 0.05, 2)
        return int(np.argmin(abs(self.grid - chosen)))

    def _grid_components(self, demand, weights):
        ratios = self.regional_power(demand) / np.asarray(self.scenario.capacity_kw)
        costs = stress_cost(ratios)
        components = {
            f"grid_{r}": -REWARD_SCALE
            * float(weights[r] * costs[r])
            / (NORMALIZATION * 48 * 2)
            for r in range(2)
        }
        return components, ratios, costs

    def greedy_action(self):
        """One-hour simulator comparator; terminal guards apply only at hour 48."""
        predictions = self.engine.candidates(
            self.history, self.grid, self.episode.future_calendar[self.hour]
        )
        weights = state_weights(
            self.regional_power(self.preceding_demand())
            / np.asarray(self.scenario.capacity_kw)
        )
        rewards = []
        for demand, price in zip(predictions, self.grid):
            components, _, _ = self._grid_components(demand, weights)
            reward = sum(components.values())
            if self.hour == 47:
                costs = household_costs(
                    self.energy + demand,
                    self.charge + demand * price,
                    self.reference.sum(0, dtype=np.float64),
                    (
                        self.reference.astype(np.float64)
                        * self.market_prices[self.area_index].T
                    ).sum(0),
                    self.training_median,
                )
                reward -= (
                    REWARD_SCALE * (costs["energy"] + costs["charge"]) / NORMALIZATION
                )
            rewards.append(reward)
        return int(np.argmax(rewards))

    def step(self, action):
        require(
            not self.done and self.action_space.contains(action),
            "Invalid action or finished episode.",
        )
        price = float(self.grid[int(action)])
        before_ratio = self.regional_power(self.preceding_demand()) / np.asarray(
            self.scenario.capacity_kw
        )
        weights = state_weights(before_ratio)
        demand = self.engine.predict(
            self.history, price, self.episode.future_calendar[self.hour]
        )
        components, ratios, costs = self._grid_components(demand, weights)
        components.update(energy=0.0, charge=0.0)
        reference_step = self.reference[self.hour].copy()
        self.energy += np.asarray(demand, dtype=np.float64)
        self.charge += np.asarray(demand, dtype=np.float64) * price
        self.history = append_history(
            self.history,
            demand,
            price,
            self.episode.future_calendar[self.hour],
            self.provider.preprocessor,
        )
        self.demands.append(np.asarray(demand).copy())
        self.prices.append(price)
        self.weights.append(weights)
        self.stresses.append(costs)
        self.previous_regional = np.full(2, price)
        self.hour += 1
        self.done = self.hour == 48
        if self.hour == 24:
            self._release(1)
        if self.done:
            guards = household_costs(
                self.energy,
                self.charge,
                self.reference.sum(0, dtype=np.float64),
                (
                    self.reference.astype(np.float64)
                    * self.market_prices[self.area_index].T
                ).sum(0),
                self.training_median,
            )
            components.update(
                {
                    name: -REWARD_SCALE * value / NORMALIZATION
                    for name, value in guards.items()
                }
            )
        reward = float(sum(components.values()))
        info = dict(
            price_nok_kwh=price,
            reward_components=components,
            telemetry=dict(
                arrays=dict(
                    requested=demand,
                    raw=demand,
                    delivered=demand,
                    essential=np.zeros_like(demand),
                    reference=reference_step,
                ),
                diagnostics={
                    **{f"load_ratio_{r}": float(ratios[r]) for r in range(2)},
                    **{f"grid_weight_{r}": float(weights[r]) for r in range(2)},
                    **{f"stress_cost_{r}": float(costs[r]) for r in range(2)},
                    **{f"previous_ratio_{r}": float(before_ratio[r]) for r in range(2)},
                    "known_hours": self.known_hours,
                },
            ),
        )
        if self.hour == 24:
            info["telemetry_context"] = self.inspection_context()[1]
        if self.done:
            info.update(self.daily_result())
        return self.observation(), reward, self.done, False, info

    def daily_result(self):
        require(self.done, "Complete the episode before scoring.")
        return dict(
            origin=self.episode.timestamp,
            scenario=self.scenario.name,
            prices=list(self.prices),
            **score_trajectory(
                self.demands,
                self.prices,
                self.reference,
                self.market_prices,
                self.episode.demand_history[:, -1],
                self.episode.previous_prices,
                self.area_index,
                self.scenario.capacity_kw,
                self.training_median,
            ),
        )

    def inspection_context(self):
        """Explicit observation-only telemetry; scenario does NOT imply allocation."""
        metadata = dict(
            origin=self.episode.timestamp,
            objective="state-dependent grid stress",
            reward_timing="Hourly regional stress; terminal 48-hour energy/charge",
            scenario=self.scenario.name,
            alpha=None,
            allocation_enabled=False,
            interval_hours=1.0,
            source="Frozen-model consumption; reference is modeled, not measured.",
            regions=list(self.provider.preprocessor.state["areas"]),
            model_engine=self.engine.signature,
            horizon=48,
        )
        arrays = dict(
            household_ids=np.asarray(self.provider.preprocessor.ids).astype(str),
            regions=self.area_index,
            initial_history=self.episode.demand_history,
            reference=self.reference.copy(),
            market_prices=self.market_prices.copy(),
            grid=self.grid,
            reference_known=np.arange(48) < self.known_hours,
            capacity_kw=np.asarray(self.scenario.capacity_kw),
        )
        return metadata, arrays

    def snapshot(self):
        keys = (
            "index",
            "scenario_index",
            "hour",
            "history",
            "reference",
            "market_prices",
            "known_hours",
            "energy",
            "charge",
            "previous_regional",
            "demands",
            "prices",
            "weights",
            "stresses",
            "done",
        )
        return dict(
            values=copy.deepcopy({k: getattr(self, k) for k in keys}),
            rng=copy.deepcopy(self.np_random.bit_generator.state),
            action_rng=copy.deepcopy(self.action_space.np_random.bit_generator.state),
        )

    def restore(self, state):
        self.reset(
            options={
                "episode_index": state["values"]["index"],
                "scenario_index": state["values"]["scenario_index"],
            }
        )
        for name, value in state["values"].items():
            setattr(self, name, copy.deepcopy(value))
        self.np_random.bit_generator.state = state["rng"]
        self.action_space.np_random.bit_generator.state = state["action_rng"]

"""Hourly tariff simulation using historical measurements and known schedules."""

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from norway_energy_crisis.clean_data import digest, require
from norway_energy_crisis.preprocess import calendar
from norway_energy_crisis.train_response import load_model

TERM_NAMES = ("demand", "energy", "jump", "charge")
WEIGHTS = np.array([1.0, 1.0, 0.1, 1.0])
REWARD_SCALE = 100.0
JUMP_RANGE = 2.95


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def price_grid(cap):
    """Return canonical discrete NOK/kWh actions, sharing a prefix across caps."""
    require(cap in (2.0, 3.0), "Price cap must be 2.00 or 3.00 NOK/kWh.")
    # Integer ore avoid accumulating floating-point increments.
    return np.r_[5, np.arange(10, int(cap * 100) + 1, 10)].astype(np.float64) / 100


def bounded(value):
    """Apply B(x) = (x / (1 + x))**2 to finite, nonnegative deviations."""
    value = np.asarray(value, dtype=np.float64)
    require(
        np.isfinite(value).all() and (value >= 0).all(), "Invalid bounded-cost input."
    )
    return np.square(value / (1.0 + value))


def price_transitions(prices, previous_regional, area_index):
    """Return per-hour normalized jump costs and separate switching diagnostics.

    The midnight transition starts from each household's regional preceding price.
    Later transitions use the shared action price. Both caps use the same 2.95
    NOK/kWh jump range; switch counts never contribute directly to the objective.
    """
    prices = np.asarray(prices, dtype=np.float64)
    prior = np.asarray(previous_regional, dtype=np.float64)[area_index]
    entry = prices[0] - prior
    changes = np.diff(prices)
    jumps = np.r_[
        (np.minimum(np.abs(entry) / JUMP_RANGE, 1.0) ** 2).mean(),
        np.minimum(np.abs(changes) / JUMP_RANGE, 1.0) ** 2,
    ]
    entry_switch = float(np.mean(np.abs(entry) > 1e-8))
    internal_switches = int(np.count_nonzero(np.abs(changes) > 1e-8))
    boundaries = np.r_[0, np.flatnonzero(np.abs(changes) > 1e-8) + 1, len(prices)]
    return jumps, dict(
        entry_switch_fraction=entry_switch,
        internal_switches=internal_switches,
        switching_fraction=(entry_switch + internal_switches) / 24,
        run_durations_hours=np.diff(boundaries).tolist(),
        mean_jump_nok_kwh=float((np.abs(entry).mean() + np.abs(changes).sum()) / 24),
        max_jump_nok_kwh=float(
            max(np.abs(entry).max(), np.abs(changes).max(initial=0))
        ),
    )


@dataclass(frozen=True)
class Episode:
    """Immutable input snapshot containing history and known schedules only."""

    origin: int
    timestamp: str
    demand_history: np.ndarray  # H x 24, measured historical kWh only
    prices_history: np.ndarray  # two regions x 24
    temperature_history: np.ndarray  # two regions x 24; no future weather
    history_times: np.ndarray
    future_prices: np.ndarray  # two regions x 24; assumed known day-ahead
    future_calendar: np.ndarray  # 24 x 6, deterministic
    terminal_calendar: np.ndarray


class EpisodeProvider:
    """Materialize safe snapshots without retaining the demand/weather panel.

    Only historical consumption and temperature are copied. Future day-ahead
    prices and deterministic calendar features are allowed; future measurements
    and the supervised PreparedData.batch helper are deliberately excluded.
    """

    def __init__(self, data, split, selection_file=None):
        require(split in ("train", "validation", "test"), "Unknown split.")
        if split == "test":
            require(
                selection_file is not None and Path(selection_file).is_file(),
                "Freeze selection before testing.",
            )
            require(
                json.loads(Path(selection_file).read_text()).get("frozen_before_test")
                is True,
                "Selection is not frozen.",
            )
        self.preprocessor = data.preprocessor
        self.area_index = np.array(data.area_index, copy=True)
        self.households = data.households
        self.manifest_hash = digest(data.folder / "manifest.json")
        self.split = split
        partition = ("train", "validation", "test").index(split)
        origins = [int(origin) for origin in data.origins[split] if origin % 24 == 0]
        require(origins, f"No eligible days in {split}.")
        episodes = []
        for origin in origins:
            require(
                np.all(data.partitions[origin - 24 : origin + 24] == partition),
                "History/episode crosses a partition.",
            )
            history_indices = np.arange(origin - 24, origin)
            target_indices = np.arange(origin, origin + 24)
            arrays = [
                np.array(data.demand[:, history_indices], copy=True),
                np.array(data.prices[:, history_indices], copy=True),
                np.array(data.temperature[:, history_indices], copy=True),
                np.array(data.times[history_indices], copy=True),
                np.array(data.prices[:, target_indices], copy=True),
                calendar(data.times[target_indices]),
                calendar(np.array([data.times[origin] + 24 * 3600 * 10**9]))[0],
            ]
            for array in arrays:
                require(np.isfinite(array).all(), "Invalid episode input.")
                array.flags.writeable = False
            episodes.append(
                Episode(
                    origin, str(np.datetime64(int(data.times[origin]), "ns")), *arrays
                )
            )
        self.episodes = tuple(episodes)

    def features(self, episode):
        """Encode one snapshot in the saved household and feature order."""
        household_count = self.households
        return self.preprocessor.history_features(
            episode.demand_history,
            episode.prices_history[self.area_index],
            episode.temperature_history[self.area_index],
            episode.history_times,
            np.arange(household_count),
        )


class FrozenResponse:
    """Run frozen checkpoints on a shared controlled household history.

    Average members' household predictions before appending demand to history.
    Reserved dynamics require frozen policy selection; no member updates weights.
    """

    def __init__(
        self, paths, data, device="mps", label="ensemble", selection_file=None
    ):
        require(device in ("cpu", "mps"), "Explicit cpu or mps device required.")
        if device == "mps":
            require(
                torch.backends.mps.is_available(),
                "MPS is unavailable. GPU access is required; no CPU fallback.",
            )
        if label == "reserved":
            require(
                selection_file is not None and Path(selection_file).is_file(),
                "Reserved response is sealed until selection.",
            )
            require(
                json.loads(Path(selection_file).read_text()).get("frozen_before_test")
                is True,
                "Selection is not frozen.",
            )
        self.device = torch.device(device)
        self.models = []
        self.preprocessor = data.preprocessor
        self.scales = self.preprocessor.scales
        self.label = label
        hashes = []
        for path in paths:
            model, bundle = load_model(path)
            require(
                bundle["prepared_manifest_sha256"]
                == digest(data.folder / "manifest.json"),
                "Response/data mismatch.",
            )
            require(
                bundle["preprocessing"] == data.state
                and model.config["lookback"] == 24,
                "Response preprocessing/history mismatch.",
            )
            model.to(self.device).eval().requires_grad_(False)
            self.models.append(model)
            hashes.append(digest(path))
        require(bool(self.models), "At least one response model is required.")
        require(
            len({response_model.use_price for response_model in self.models}) == 1,
            "Do not mix price and price-free models.",
        )
        self.is_price_free = not self.models[0].use_price
        self.signature = fingerprint(
            dict(
                models=hashes,
                data=digest(data.folder / "manifest.json"),
                recursion="shared_mean_v1",
                temperature="persist_origin",
                source=digest(__file__),
            )
        )
        self.static = torch.as_tensor(self.preprocessor.static, device=self.device)

    @torch.inference_mode()
    def predict(self, history, raw_prices, target_calendar):
        """Return kWh predictions for one episode or a batch of candidate rollouts.

        History has shape (households, 24, 9) or (batch, households, 24, 9).
        Raw NOK/kWh prices may be scalar, household-sized or batch-by-household.
        All members receive identical inputs; their normalized predictions are averaged.
        """
        history = np.asarray(history, dtype=np.float32)
        single = history.ndim == 3
        if single:
            history = history[None]
        batch_size, household_count = history.shape[:2]
        require(
            history.shape[2:] == (24, 9) and household_count == len(self.scales),
            "Invalid response history.",
        )
        prices = np.broadcast_to(
            np.asarray(raw_prices, dtype=np.float32), (batch_size, household_count)
        )
        calendar_features = np.asarray(target_calendar, dtype=np.float32)
        if calendar_features.shape == (6,):
            calendar_features = np.broadcast_to(calendar_features, (batch_size, 6))
        require(calendar_features.shape == (batch_size, 6), "Invalid target calendar.")
        history_tensor = torch.as_tensor(
            np.array(history.reshape(batch_size * household_count, 24, 9), copy=True),
            device=self.device,
        )
        price_tensor = torch.as_tensor(
            self.preprocessor.price(prices).ravel().copy(), device=self.device
        )
        calendar_tensor = torch.as_tensor(
            np.repeat(calendar_features, household_count, axis=0).copy(),
            device=self.device,
        )
        static = self.static.repeat(batch_size, 1)
        values = torch.stack(
            [
                model(history_tensor, static, price_tensor, calendar_tensor)
                for model in self.models
            ]
        ).mean(0)
        result = (
            values.cpu().numpy().reshape(batch_size, household_count)
            * self.scales[None]
        )
        require(
            np.isfinite(result).all() and (result >= 0).all(),
            "Invalid simulator prediction; refusing to clip.",
        )
        return result[0] if single else result

    @torch.inference_mode()
    def predict_candidates(self, history, prices, target_calendar):
        """Score all shared candidate prices while encoding household history once.

        Return (actions, households) kWh predictions. The prediction head and
        persistence correction match ResponseModel.forward, including price-free inputs.
        """
        prices = np.asarray(prices, dtype=np.float32)
        household_count, action_count = len(self.scales), len(prices)
        history_tensor = torch.as_tensor(
            np.array(history, dtype=np.float32, copy=True), device=self.device
        )
        calendar_tensor = torch.as_tensor(
            np.array(target_calendar, dtype=np.float32), device=self.device
        )
        price_tensor = torch.as_tensor(
            self.preprocessor.price(prices), device=self.device
        )
        values = []
        for model in self.models:
            encoded = model.encode_history(history_tensor)
            values.append(
                model.predict_encoded(
                    encoded[None].expand(action_count, -1, -1),
                    self.static[None].expand(action_count, -1, -1),
                    price_tensor[:, None].expand(-1, household_count),
                    calendar_tensor[None, None].expand(
                        action_count, household_count, -1
                    ),
                    history_tensor[None, :, -1, 0],
                )
            )
        result = torch.stack(values).mean(0).cpu().numpy() * self.scales[None]
        require(
            np.isfinite(result).all() and (result >= 0).all(),
            "Invalid candidate prediction.",
        )
        return result


def append_history(history, predicted, raw_prices, target_calendar, preprocessor):
    """Append predicted demand and chosen prices, retaining the last temperature.

    History stays in the saved normalized feature space; predictions arrive in
    kWh and candidate prices in NOK/kWh. Supports single and batched rollouts.
    """
    single = history.ndim == 3
    batched_history = history[None] if single else history
    batch_size, household_count = batched_history.shape[:2]
    demand = (
        np.broadcast_to(predicted, (batch_size, household_count))
        / preprocessor.scales[None]
    )
    price = preprocessor.price(
        np.broadcast_to(raw_prices, (batch_size, household_count))
    )
    calendar_features = np.asarray(target_calendar)
    if calendar_features.shape == (6,):
        calendar_features = np.broadcast_to(calendar_features, (batch_size, 6))
    next_row = np.concatenate(
        (
            demand[:, :, None],
            price[:, :, None],
            batched_history[:, :, -1, 2, None],
            np.broadcast_to(
                calendar_features[:, None], (batch_size, household_count, 6)
            ),
        ),
        axis=2,
    ).astype(np.float32)
    result = np.concatenate((batched_history[:, :, 1:], next_row[:, :, None]), axis=2)
    return result[0] if single else result


class ReferenceCache:
    """Cache recorded-price rollouts keyed by response/data/simulation fingerprints."""

    def __init__(self, provider, engine, folder):
        self.provider, self.engine = provider, engine
        self.folder = Path(folder) / engine.signature / provider.split
        self.memory = {}

    def get(self, index):
        """Return the fixed 24-hour reference demand for an enumerated episode."""
        episode = self.provider.episodes[index]
        if index in self.memory:
            return self.memory[index]
        path = self.folder / f"{episode.origin}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                demand = saved["demand"]
        else:
            history = self.provider.features(episode)
            demand = []
            for hour in range(24):
                price = episode.future_prices[self.provider.area_index, hour]
                predicted = self.engine.predict(
                    history, price, episode.future_calendar[hour]
                )
                demand.append(predicted)
                history = append_history(
                    history,
                    predicted,
                    price,
                    episode.future_calendar[hour],
                    self.provider.preprocessor,
                )
            demand = np.asarray(demand, dtype=np.float32)
            atomic_npz(path, demand=demand)
        require(
            demand.shape == (24, self.provider.households)
            and np.isfinite(demand).all()
            and (demand >= 0).all(),
            "Invalid cached reference.",
        )
        self.memory[index] = demand
        return demand


def score(demand, prices, reference, episode, area_index, training_median):
    """Score a complete day using demand, energy, jump and charge-increase terms.

    Demand/reference have shape (24, households) in kWh; prices are 24 shared
    NOK/kWh actions. Reference energy charges use each region's recorded prices.
    The charge denominator handles negative and zero reference charges safely.
    Return the weighted cost in [0, 1], its four components and physical diagnostics.
    """
    demand, reference = np.asarray(demand, dtype=np.float64), np.asarray(
        reference, dtype=np.float64
    )
    prices = np.asarray(prices, dtype=np.float64)
    require(
        demand.shape == reference.shape
        and demand.shape[0] == 24
        and prices.shape == (24,),
        "Invalid episode shape.",
    )
    require(
        np.isfinite(demand).all()
        and np.isfinite(reference).all()
        and np.isfinite(prices).all()
        and (demand >= 0).all()
        and (reference >= 0).all(),
        "Invalid scoring input.",
    )
    reference_energy = reference.sum(0)
    reference_charge = (reference * episode.future_prices[area_index].T).sum(0)
    target = reference_energy.sum() / 24
    charge_scale = np.maximum.reduce(
        (
            np.abs(reference_charge),
            0.01 * training_median * reference_energy,
            np.full_like(reference_energy, 1e-6),
        )
    )
    jumps, switching = price_transitions(
        prices, episode.prices_history[:, -1], area_index
    )
    aggregate_demand = demand.sum(1)
    reference_aggregate = reference.sum(1)
    energy_delta = (demand.sum(0) - reference_energy) / np.maximum(
        reference_energy, 1e-6
    )
    charge_delta = (demand * prices[:, None]).sum(0) - reference_charge
    demand_deviation = np.maximum(
        np.abs(aggregate_demand - target) / max(target, 1e-6) - 0.10, 0
    )
    charge_increase = np.maximum(charge_delta / charge_scale, 0)
    terms = dict(
        demand=float(bounded(demand_deviation).mean()),
        energy=float(bounded(np.abs(energy_delta)).mean()),
        jump=float(jumps.mean()),
        charge=float(bounded(charge_increase).mean()),
    )
    cost = float(np.dot(WEIGHTS, [terms[term] for term in TERM_NAMES]) / WEIGHTS.sum())
    stats = dict(
        **switching,
        peak_kwh=float(aggregate_demand.max()),
        reference_peak_kwh=float(reference_aggregate.max()),
        peak_reduction_pct=float(
            100 * (1 - aggregate_demand.max() / max(reference_aggregate.max(), 1e-6))
        ),
        energy_change_pct=float(100 * (demand.sum() / max(reference.sum(), 1e-6) - 1)),
        energy_household_mean_abs_change_pct=float(100 * np.abs(energy_delta).mean()),
        charge_change_nok=float(charge_delta.sum()),
        households_charge_increased_fraction=float(np.mean(charge_delta > 1e-8)),
        above_1_20_fraction=float(np.mean(prices > 1.20 + 1e-8)),
        above_2_00_fraction=float(np.mean(prices > 2.0 + 1e-8)),
        above_2_50_fraction=float(np.mean(prices > 2.50 + 1e-8)),
    )
    return dict(cost=cost, terms=terms, stats=stats)


class PriceEnv(gym.Env):
    """Choose one shared price each hour for a 24-decision demand-shaping episode.

    Observations include household history, known schedules and cumulative
    accounts. Demand/jump costs arrive hourly; energy/charge costs arrive at the
    last step. With gamma=1, total reward is exactly -100 times the daily cost.
    """

    metadata = {"render_modes": []}

    def __init__(self, provider, engine, cache_folder, cap):
        super().__init__()
        self.provider, self.engine = provider, engine
        self.grid = price_grid(cap)
        self.cap = cap
        self.reference_cache = ReferenceCache(provider, engine, cache_folder)
        self.training_median = float(provider.preprocessor.state["price_band_edges"][1])
        household_count = provider.households
        shapes = dict(
            demand=(household_count, 24),
            regional=(2, 24, 2),
            calendar=(24, 6),
            known_prices=(2, 24),
            known_calendar=(24, 6),
            accounts=(4, household_count),
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
        self.first_in_area = [
            int(np.flatnonzero(provider.area_index == area)[0]) for area in range(2)
        ]

    def observation(self):
        """Flatten factored context into float32 arrays using saved normalization.

        Demand scaling and regional standardization are reused unchanged. Energy
        accounts use 24-hour demand scales; charge accounts also use the common
        3 NOK/kWh maximum so observations remain comparable across action caps.
        """
        preprocessor = self.provider.preprocessor
        scales = preprocessor.scales.astype(np.float64)
        current_calendar = (
            self.episode.future_calendar[self.hour]
            if self.hour < 24
            else self.episode.terminal_calendar
        )
        result = dict(
            demand=self.history[:, :, 0],
            regional=self.history[self.first_in_area, :, 1:3],
            calendar=self.history[0, :, 3:],
            known_prices=preprocessor.price(self.episode.future_prices),
            known_calendar=self.episode.future_calendar,
            accounts=np.stack(
                (
                    self.energy / (24 * scales),
                    self.charge / (24 * scales * 3),
                    self.energy0 / (24 * scales),
                    self.charge0 / (24 * scales * 3),
                )
            ),
            context=np.r_[
                self.hour / 24,
                (24 - self.hour) / 24,
                preprocessor.price(self.previous_regional),
                self.target / scales.sum(),
                self.demand_sum / 24,
                self.jump_sum / 24,
                current_calendar,
            ],
        )
        result = {
            name: np.array(values, dtype=np.float32, copy=True).ravel()
            for name, values in result.items()
        }
        require(
            all(np.isfinite(values).all() for values in result.values()),
            "Nonfinite policy observation.",
        )
        return result

    def reset(self, *, seed=None, options=None):
        """Start a sampled or indexed day from the reference's starting history."""
        super().reset(seed=seed)
        self.done = True
        index = (options or {}).get("episode_index")
        if index is None:
            index = int(self.np_random.integers(len(self.provider.episodes)))
        require(
            isinstance(index, (int, np.integer))
            and 0 <= index < len(self.provider.episodes),
            "Invalid episode index.",
        )
        self.index, self.hour = int(index), 0
        self.episode = self.provider.episodes[self.index]
        self.history = self.provider.features(self.episode)
        self.reference = self.reference_cache.get(self.index)
        self.energy0 = self.reference.astype(np.float64).sum(0)
        self.charge0 = (
            self.reference.astype(np.float64)
            * self.episode.future_prices[self.provider.area_index].astype(np.float64).T
        ).sum(0)
        self.target = self.energy0.sum() / 24
        self.denom = np.maximum.reduce(
            (
                np.abs(self.charge0),
                0.01 * self.training_median * self.energy0,
                np.full_like(self.energy0, 1e-6),
            )
        )
        self.energy = np.zeros(self.provider.households, dtype=np.float64)
        self.charge = self.energy.copy()
        self.previous_regional = self.episode.prices_history[:, -1].copy()
        self.demand_sum = self.jump_sum = 0.0
        self.demands, self.prices = [], []
        self.done = False
        return self.observation(), dict(origin=self.episode.timestamp)

    def incremental_cost(self, demands, prices):
        """Return this step's weighted cost and unweighted demand/jump terms.

        Accept one household trajectory or an action-by-household matrix for greedy
        selection. Daily energy and charge penalties are added only at hour 23;
        there is no invented intermediate spending target or switch-count penalty.
        """
        demands = np.asarray(demands, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        demand_deviation = np.maximum(
            np.abs(demands.sum(-1) - self.target) / max(self.target, 1e-6) - 0.10, 0
        )
        demand_cost = bounded(demand_deviation)
        previous = self.previous_regional[self.provider.area_index]
        jump_cost = np.mean(
            np.minimum(np.abs(prices[..., None] - previous) / JUMP_RANGE, 1.0) ** 2,
            axis=-1,
        )
        energy_cost = np.zeros_like(demand_cost)
        charge_cost = np.zeros_like(demand_cost)
        if self.hour == 23:
            energy_cost = bounded(
                np.abs(self.energy + demands - self.energy0)
                / np.maximum(self.energy0, 1e-6)
            ).mean(-1)
            charge_cost = bounded(
                np.maximum(
                    (self.charge + demands * prices[..., None] - self.charge0)
                    / self.denom,
                    0,
                )
            ).mean(-1)
        return (
            ((demand_cost + 0.1 * jump_cost) / 24 + energy_cost + charge_cost) / 3.1,
            demand_cost,
            jump_cost,
        )

    def greedy_action(self):
        """Minimize immediate cost, including daily terms at the last hour."""
        require(not self.done, "Reset before choosing an action.")
        candidates = self.engine.predict_candidates(
            self.history, self.grid, self.episode.future_calendar[self.hour]
        )
        return int(np.argmin(self.incremental_cost(candidates, self.grid)[0]))

    def step(self, action):
        """Predict demand, advance history/accounts and return the hourly reward."""
        require(not self.done, "Reset before stepping.")
        require(self.action_space.contains(action), "Invalid discrete action.")
        price = float(self.grid[int(action)])
        predicted = self.engine.predict(
            self.history, price, self.episode.future_calendar[self.hour]
        )
        increment, demand_cost, jump_cost = self.incremental_cost(
            predicted, np.array(price)
        )
        self.energy += predicted.astype(np.float64)
        self.charge += predicted.astype(np.float64) * price
        self.history = append_history(
            self.history,
            predicted,
            price,
            self.episode.future_calendar[self.hour],
            self.provider.preprocessor,
        )
        self.demand_sum += float(demand_cost)
        self.jump_sum += float(jump_cost)
        self.demands.append(predicted.copy())
        self.prices.append(price)
        self.previous_regional = np.full(2, price)
        self.hour += 1
        self.done = self.hour == 24
        info = dict(price_nok_kwh=price)
        if self.done:
            info.update(
                score(
                    self.demands,
                    self.prices,
                    self.reference,
                    self.episode,
                    self.provider.area_index,
                    self.training_median,
                )
            )
        return (
            self.observation(),
            -REWARD_SCALE * float(increment),
            self.done,
            False,
            info,
        )

    def snapshot(self):
        """Copy mutable episode state and both RNGs for a local training checkpoint."""
        keys = (
            "index",
            "hour",
            "history",
            "energy",
            "charge",
            "previous_regional",
            "demand_sum",
            "jump_sum",
            "demands",
            "prices",
            "done",
        )
        return dict(
            values=copy.deepcopy({name: getattr(self, name) for name in keys}),
            rng=copy.deepcopy(self.np_random.bit_generator.state),
            action_rng=copy.deepcopy(self.action_space.np_random.bit_generator.state),
        )

    def restore(self, state):
        """Rebuild fixed day inputs, then restore mutable episode state and RNGs."""
        self.reset(options={"episode_index": state["values"]["index"]})
        for name, value in state["values"].items():
            setattr(self, name, copy.deepcopy(value))
        self.np_random.bit_generator.state = state["rng"]
        self.action_space.np_random.bit_generator.state = state["action_rng"]

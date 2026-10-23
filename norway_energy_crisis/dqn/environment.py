"""Hourly tariff simulation. No future consumption or measured weather enters an episode."""

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

TERM_NAMES = ('demand', 'energy', 'jump', 'charge')
WEIGHTS = np.array([1., 1., .1, 1.])
REWARD_SCALE = 100.
JUMP_RANGE = 2.95


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def price_grid(cap):
    require(cap in (2., 3.), 'Price cap must be 2.00 or 3.00 NOK/kWh.')
    # Integer ore avoid accumulating floating-point increments.
    return np.r_[5, np.arange(10, int(cap * 100) + 1, 10)].astype(np.float64) / 100


def bounded(value):
    value = np.asarray(value, dtype=np.float64)
    require(np.isfinite(value).all() and (value >= 0).all(), 'Invalid bounded-cost input.')
    return np.square(value / (1. + value))


def price_transitions(prices, previous_regional, area_index):
    prices = np.asarray(prices, dtype=np.float64)
    prior = np.asarray(previous_regional, dtype=np.float64)[area_index]
    entry = prices[0] - prior
    changes = np.diff(prices)
    jumps = np.r_[(np.minimum(np.abs(entry) / JUMP_RANGE, 1.) ** 2).mean(),
                  np.minimum(np.abs(changes) / JUMP_RANGE, 1.) ** 2]
    entry_switch = float(np.mean(np.abs(entry) > 1e-8))
    internal_switches = int(np.count_nonzero(np.abs(changes) > 1e-8))
    boundaries = np.r_[0, np.flatnonzero(np.abs(changes) > 1e-8) + 1, len(prices)]
    return jumps, dict(entry_switch_fraction=entry_switch, internal_switches=internal_switches,
                       switching_fraction=(entry_switch + internal_switches) / 24,
                       run_durations_hours=np.diff(boundaries).tolist(),
                       mean_jump_nok_kwh=float((np.abs(entry).mean() + np.abs(changes).sum()) / 24),
                       max_jump_nok_kwh=float(max(np.abs(entry).max(), np.abs(changes).max(initial=0))))


@dataclass(frozen=True)
class Episode:
    origin: int
    timestamp: str
    demand_history: np.ndarray       # H x 24, measured historical kWh only
    prices_history: np.ndarray       # two regions x 24
    temperature_history: np.ndarray # two regions x 24; no future weather
    history_times: np.ndarray
    future_prices: np.ndarray        # two regions x 24; assumed known day-ahead
    future_calendar: np.ndarray      # 24 x 6, deterministic
    terminal_calendar: np.ndarray


class EpisodeProvider:
    """Materialize safe snapshots; never retain the full demand/weather panel."""
    def __init__(self, data, split, selection_file=None):
        require(split in ('train', 'validation', 'test'), 'Unknown split.')
        if split == 'test':
            require(selection_file is not None and Path(selection_file).is_file(), 'Freeze selection before testing.')
            require(json.loads(Path(selection_file).read_text()).get('frozen_before_test') is True,
                    'Selection is not frozen.')
        self.preprocessor = data.preprocessor
        self.area_index = np.array(data.area_index, copy=True)
        self.households = data.households
        self.manifest_hash = digest(data.folder / 'manifest.json')
        self.split = split
        partition = ('train', 'validation', 'test').index(split)
        origins = [int(i) for i in data.origins[split] if i % 24 == 0]
        require(origins, f'No eligible days in {split}.')
        episodes = []
        for i in origins:
            require(np.all(data.partitions[i-24:i+24] == partition), 'History/episode crosses a partition.')
            hist = np.arange(i-24, i)
            future = np.arange(i, i+24)
            arrays = [np.array(data.demand[:, hist], copy=True), np.array(data.prices[:, hist], copy=True),
                      np.array(data.temperature[:, hist], copy=True), np.array(data.times[hist], copy=True),
                      np.array(data.prices[:, future], copy=True), calendar(data.times[future]),
                      calendar(np.array([data.times[i] + 24 * 3600 * 10**9]))[0]]
            for array in arrays:
                require(np.isfinite(array).all(), 'Invalid episode input.')
                array.flags.writeable = False
            episodes.append(Episode(i, str(np.datetime64(int(data.times[i]), 'ns')), *arrays))
        self.episodes = tuple(episodes)

    def features(self, episode):
        h = self.households
        return self.preprocessor.history_features(episode.demand_history,
                    episode.prices_history[self.area_index], episode.temperature_history[self.area_index],
                    episode.history_times, np.arange(h))


class FrozenResponse:
    """One or more frozen checkpoints, averaged before updating a shared history."""
    def __init__(self, paths, data, device='mps', label='ensemble', selection_file=None):
        require(device in ('cpu', 'mps'), 'Explicit cpu or mps device required.')
        if device == 'mps':
            require(torch.backends.mps.is_available(), 'MPS is unavailable. GPU access is required; no CPU fallback.')
        if label == 'reserved':
            require(selection_file is not None and Path(selection_file).is_file(), 'Reserved response is sealed until selection.')
            require(json.loads(Path(selection_file).read_text()).get('frozen_before_test') is True, 'Selection is not frozen.')
        self.device = torch.device(device)
        self.models = []
        self.preprocessor = data.preprocessor
        self.scales = self.preprocessor.scales
        self.label = label
        hashes = []
        for path in paths:
            model, bundle = load_model(path)
            require(bundle['prepared_manifest_sha256'] == digest(data.folder / 'manifest.json'), 'Response/data mismatch.')
            require(bundle['preprocessing'] == data.state and model.config['lookback'] == 24,
                    'Response preprocessing/history mismatch.')
            model.to(self.device).eval().requires_grad_(False)
            self.models.append(model)
            hashes.append(digest(path))
        require(bool(self.models), 'At least one response model is required.')
        require(len({m.use_price for m in self.models}) == 1, 'Do not mix price and price-free models.')
        self.is_price_free = not self.models[0].use_price
        self.signature = fingerprint(dict(models=hashes, data=digest(data.folder / 'manifest.json'),
                                          recursion='shared_mean_v1', temperature='persist_origin',
                                          source=digest(__file__)))
        self.static = torch.as_tensor(self.preprocessor.static, device=self.device)

    @torch.inference_mode()
    def predict(self, history, raw_prices, target_calendar):
        """history Hx24x9 or BxHx24x9; candidate prices scalar, H, or BxH."""
        history = np.asarray(history, dtype=np.float32)
        single = history.ndim == 3
        if single:
            history = history[None]
        b, h = history.shape[:2]
        require(history.shape[2:] == (24, 9) and h == len(self.scales), 'Invalid response history.')
        prices = np.broadcast_to(np.asarray(raw_prices, dtype=np.float32), (b, h))
        cal = np.asarray(target_calendar, dtype=np.float32)
        if cal.shape == (6,):
            cal = np.broadcast_to(cal, (b, 6))
        require(cal.shape == (b, 6), 'Invalid target calendar.')
        x = torch.as_tensor(np.array(history.reshape(b*h, 24, 9), copy=True), device=self.device)
        p = torch.as_tensor(self.preprocessor.price(prices).ravel().copy(), device=self.device)
        c = torch.as_tensor(np.repeat(cal, h, axis=0).copy(), device=self.device)
        static = self.static.repeat(b, 1)
        values = torch.stack([m(x, static, p, c) for m in self.models]).mean(0)
        result = values.cpu().numpy().reshape(b, h) * self.scales[None]
        require(np.isfinite(result).all() and (result >= 0).all(), 'Invalid simulator prediction; refusing to clip.')
        return result[0] if single else result

    @torch.inference_mode()
    def predict_candidates(self, history, prices, target_calendar):
        """Encode once per model; evaluate all candidate-price heads without mutation."""
        prices = np.asarray(prices, dtype=np.float32)
        h, a = len(self.scales), len(prices)
        x = torch.as_tensor(np.array(history, dtype=np.float32, copy=True), device=self.device)
        c = torch.as_tensor(np.array(target_calendar, dtype=np.float32), device=self.device)
        p = torch.as_tensor(self.preprocessor.price(prices), device=self.device)
        values = []
        for m in self.models:
            inputs = x.clone() if not m.use_price else x
            if not m.use_price:
                inputs[:, :, 1] = 0
            if m.config['architecture'] == 'Transformer':
                encoded = m.encoder(m.projection(inputs) + m.position[None, :24])[:, -1]
            else:
                encoded = m.encoder(inputs)[0][:, -1]
            next_price = p if m.use_price else torch.zeros_like(p)
            head = torch.cat((encoded[None].expand(a, -1, -1), self.static[None].expand(a, -1, -1),
                              next_price[:, None, None].expand(-1, h, -1), c[None, None].expand(a, h, -1)), dim=2)
            anchor = inputs[:, -1, 0].clamp_min(1e-6)
            latent = anchor + torch.log(-torch.expm1(-anchor))
            values.append(torch.nn.functional.softplus(latent[None] + m.head(head)[:, :, 0]))
        result = torch.stack(values).mean(0).cpu().numpy() * self.scales[None]
        require(np.isfinite(result).all() and (result >= 0).all(), 'Invalid candidate prediction.')
        return result


def append_history(history, predicted, raw_prices, target_calendar, preprocessor):
    single = history.ndim == 3
    x = history[None] if single else history
    b, h = x.shape[:2]
    demand = np.broadcast_to(predicted, (b, h)) / preprocessor.scales[None]
    price = preprocessor.price(np.broadcast_to(raw_prices, (b, h)))
    cal = np.asarray(target_calendar)
    if cal.shape == (6,):
        cal = np.broadcast_to(cal, (b, 6))
    row = np.concatenate((demand[:, :, None], price[:, :, None], x[:, :, -1, 2, None],
                          np.broadcast_to(cal[:, None], (b, h, 6))), axis=2).astype(np.float32)
    result = np.concatenate((x[:, :, 1:], row[:, :, None]), axis=2)
    return result[0] if single else result


class ReferenceCache:
    def __init__(self, provider, engine, folder):
        self.provider, self.engine = provider, engine
        self.folder = Path(folder) / engine.signature / provider.split
        self.memory = {}

    def get(self, index):
        episode = self.provider.episodes[index]
        if index in self.memory:
            return self.memory[index]
        path = self.folder / f'{episode.origin}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                demand = saved['demand']
        else:
            history = self.provider.features(episode)
            demand = []
            for t in range(24):
                price = episode.future_prices[self.provider.area_index, t]
                predicted = self.engine.predict(history, price, episode.future_calendar[t])
                demand.append(predicted)
                history = append_history(history, predicted, price, episode.future_calendar[t], self.provider.preprocessor)
            demand = np.asarray(demand, dtype=np.float32)
            atomic_npz(path, demand=demand)
        require(demand.shape == (24, self.provider.households) and np.isfinite(demand).all()
                and (demand >= 0).all(), 'Invalid cached reference.')
        self.memory[index] = demand
        return demand


def score(demand, prices, reference, episode, area_index, training_median):
    demand, reference = np.asarray(demand, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    require(demand.shape == reference.shape and demand.shape[0] == 24 and prices.shape == (24,), 'Invalid episode shape.')
    require(np.isfinite(demand).all() and np.isfinite(reference).all() and np.isfinite(prices).all()
            and (demand >= 0).all() and (reference >= 0).all(), 'Invalid scoring input.')
    energy0 = reference.sum(0)
    charge0 = (reference * episode.future_prices[area_index].T).sum(0)
    target = energy0.sum() / 24
    denom = np.maximum.reduce((np.abs(charge0), .01 * training_median * energy0, np.full_like(energy0, 1e-6)))
    jumps, switching = price_transitions(prices, episode.prices_history[:, -1], area_index)
    parts = dict(demand=float(bounded(np.maximum(np.abs(demand.sum(1)-target) / max(target, 1e-6)-.10, 0)).mean()),
                 energy=float(bounded(np.abs(demand.sum(0)-energy0) / np.maximum(energy0, 1e-6)).mean()),
                 jump=float(jumps.mean()),
                 charge=float(bounded(np.maximum(((demand*prices[:, None]).sum(0)-charge0)/denom, 0)).mean()))
    cost = float(np.dot(WEIGHTS, [parts[k] for k in TERM_NAMES]) / WEIGHTS.sum())
    q, q0 = demand.sum(1), reference.sum(1)
    energy_delta = (demand.sum(0)-energy0) / np.maximum(energy0, 1e-6)
    charge_delta = (demand*prices[:, None]).sum(0)-charge0
    stats = dict(**switching, peak_kwh=float(q.max()), reference_peak_kwh=float(q0.max()),
                 peak_reduction_pct=float(100*(1-q.max()/max(q0.max(), 1e-6))),
                 energy_change_pct=float(100*(demand.sum()/max(reference.sum(), 1e-6)-1)),
                 energy_household_mean_abs_change_pct=float(100*np.abs(energy_delta).mean()),
                 charge_change_nok=float(charge_delta.sum()),
                 households_charge_increased_fraction=float(np.mean(charge_delta > 1e-8)),
                 above_1_20_fraction=float(np.mean(prices > 1.20+1e-8)),
                 above_2_00_fraction=float(np.mean(prices > 2.+1e-8)),
                 above_2_50_fraction=float(np.mean(prices > 2.50+1e-8)))
    return dict(cost=cost, terms=parts, stats=stats)


class PriceEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, provider, engine, cache_folder, cap):
        super().__init__()
        self.provider, self.engine = provider, engine
        self.grid = price_grid(cap)
        self.cap = cap
        self.reference_cache = ReferenceCache(provider, engine, cache_folder)
        self.training_median = float(provider.preprocessor.state['price_band_edges'][1])
        h = provider.households
        shapes = dict(demand=(h, 24), regional=(2, 24, 2), calendar=(24, 6),
                      known_prices=(2, 24), known_calendar=(24, 6), accounts=(4, h), context=(13,))
        self.feature_shapes = shapes
        self.observation_space = gym.spaces.Dict({k:gym.spaces.Box(-np.inf, np.inf, shape=(int(np.prod(v)),), dtype=np.float32)
                                                 for k, v in shapes.items()})
        self.action_space = gym.spaces.Discrete(len(self.grid))
        self.done = True
        self.first_in_area = [int(np.flatnonzero(provider.area_index == a)[0]) for a in range(2)]

    def observation(self):
        p = self.provider.preprocessor
        scales = p.scales.astype(np.float64)
        current_cal = self.episode.future_calendar[self.hour] if self.hour < 24 else self.episode.terminal_calendar
        result = dict(demand=self.history[:, :, 0], regional=self.history[self.first_in_area, :, 1:3],
                      calendar=self.history[0, :, 3:], known_prices=p.price(self.episode.future_prices),
                      known_calendar=self.episode.future_calendar,
                      accounts=np.stack((self.energy/(24*scales), self.charge/(24*scales*3),
                                         self.energy0/(24*scales), self.charge0/(24*scales*3))),
                      context=np.r_[self.hour/24, (24-self.hour)/24, p.price(self.previous_regional),
                                    self.target/scales.sum(), self.demand_sum/24, self.jump_sum/24, current_cal])
        result = {k:np.array(v, dtype=np.float32, copy=True).ravel() for k, v in result.items()}
        require(all(np.isfinite(v).all() for v in result.values()), 'Nonfinite policy observation.')
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = True
        index = (options or {}).get('episode_index')
        if index is None:
            index = int(self.np_random.integers(len(self.provider.episodes)))
        require(isinstance(index, (int, np.integer)) and 0 <= index < len(self.provider.episodes), 'Invalid episode index.')
        self.index, self.hour = int(index), 0
        self.episode = self.provider.episodes[self.index]
        self.history = self.provider.features(self.episode)
        self.reference = self.reference_cache.get(self.index)
        self.energy0 = self.reference.astype(np.float64).sum(0)
        self.charge0 = (self.reference.astype(np.float64) *
                        self.episode.future_prices[self.provider.area_index].astype(np.float64).T).sum(0)
        self.target = self.energy0.sum()/24
        self.denom = np.maximum.reduce((np.abs(self.charge0), .01*self.training_median*self.energy0,
                                       np.full_like(self.energy0, 1e-6)))
        self.energy = np.zeros(self.provider.households, dtype=np.float64)
        self.charge = self.energy.copy()
        self.previous_regional = self.episode.prices_history[:, -1].copy()
        self.demand_sum = self.jump_sum = 0.
        self.demands, self.prices = [], []
        self.done = False
        return self.observation(), dict(origin=self.episode.timestamp)

    def incremental_cost(self, demands, prices):
        demands = np.asarray(demands, dtype=np.float64)
        prices = np.asarray(prices, dtype=np.float64)
        d = bounded(np.maximum(np.abs(demands.sum(-1)-self.target)/max(self.target, 1e-6)-.10, 0))
        previous = self.previous_regional[self.provider.area_index]
        j = np.mean(np.minimum(np.abs(prices[..., None]-previous)/JUMP_RANGE, 1.)**2, axis=-1)
        energy = np.zeros_like(d)
        charge = np.zeros_like(d)
        if self.hour == 23:
            energy = bounded(np.abs(self.energy+demands-self.energy0)/np.maximum(self.energy0, 1e-6)).mean(-1)
            charge = bounded(np.maximum((self.charge+demands*prices[..., None]-self.charge0)/self.denom, 0)).mean(-1)
        return ((d+.1*j)/24+energy+charge)/3.1, d, j

    def greedy_action(self):
        require(not self.done, 'Reset before choosing an action.')
        candidates = self.engine.predict_candidates(self.history, self.grid, self.episode.future_calendar[self.hour])
        return int(np.argmin(self.incremental_cost(candidates, self.grid)[0]))

    def step(self, action):
        require(not self.done, 'Reset before stepping.')
        require(self.action_space.contains(action), 'Invalid discrete action.')
        price = float(self.grid[int(action)])
        predicted = self.engine.predict(self.history, price, self.episode.future_calendar[self.hour])
        increment, d, j = self.incremental_cost(predicted, np.array(price))
        self.energy += predicted.astype(np.float64)
        self.charge += predicted.astype(np.float64)*price
        self.history = append_history(self.history, predicted, price, self.episode.future_calendar[self.hour], self.provider.preprocessor)
        self.demand_sum += float(d)
        self.jump_sum += float(j)
        self.demands.append(predicted.copy()); self.prices.append(price)
        self.previous_regional = np.full(2, price)
        self.hour += 1
        self.done = self.hour == 24
        info = dict(price_nok_kwh=price)
        if self.done:
            info.update(score(self.demands, self.prices, self.reference, self.episode,
                              self.provider.area_index, self.training_median))
        return self.observation(), -REWARD_SCALE*float(increment), self.done, False, info

    def snapshot(self):
        keys = ('index', 'hour', 'history', 'energy', 'charge', 'previous_regional', 'demand_sum', 'jump_sum',
                'demands', 'prices', 'done')
        import copy
        return dict(values=copy.deepcopy({k:getattr(self, k) for k in keys}),
                    rng=copy.deepcopy(self.np_random.bit_generator.state),
                    action_rng=copy.deepcopy(self.action_space.np_random.bit_generator.state))

    def restore(self, state):
        self.reset(options={'episode_index':state['values']['index']})
        import copy
        for k, v in state['values'].items():
            setattr(self, k, copy.deepcopy(v))
        self.np_random.bit_generator.state = state['rng']
        self.action_space.np_random.bit_generator.state = state['action_rng']

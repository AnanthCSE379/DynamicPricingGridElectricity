"""Shared-tariff Gymnasium environment with the five bounded soft objectives.

One episode = 12 two-hour decisions over 24 hours, ending after the last decision.
All consumption is kWh/half-hour; all prices are pence/kWh. Reference demand
is a frozen Normal-price rollout from the same initial history, not future
measurements. This is a learned simulator, not a causal customer model.
"""

import numpy as np
import pandas as pd
from gymnasium import Env
from gymnasium.spaces import Box, Dict, Discrete

PRICES = np.array([3.99, 11.76, 67.20])
DEFAULT_WEIGHTS = dict(demand=1.0, energy=1.0, switch=0.1, jump=0.1, bill=1.0)


def time_features(timestamps):
    """Return [noon_peak, sine_phase], both in [0,1], for local clock time.

    noon_peak = (1-cos(theta))/2: midnight=0, noon=1, next midnight=0.
    sine_phase = (1+sin(theta))/2 distinguishes the two halves of the day.
    theta = 2*pi*hour/24. No fitted scaler or future data is involved.
    """
    dates = pd.DatetimeIndex(timestamps)
    if dates.hasnans:
        raise ValueError("Time features require valid timestamps.")
    hours = dates.hour + dates.minute / 60 + dates.second / 3600
    phase = 2 * np.pi * np.asarray(hours) / 24
    return np.column_stack(((1 - np.cos(phase)) / 2,
                            (1 + np.sin(phase)) / 2)).astype(np.float32)


def _weights(weights):
    values = dict(DEFAULT_WEIGHTS if weights is None else weights)
    if (set(values) != set(DEFAULT_WEIGHTS)
            or any(not np.isfinite(v) or v < 0 for v in values.values())
            or values['demand'] <= 0):
        raise ValueError("Require all five nonnegative finite weights; demand must be positive.")
    return values


def _price(value):
    matches = np.flatnonzero(np.isclose(PRICES, value, rtol=0, atol=1e-5))
    if len(matches) != 1:
        raise ValueError("Price must be 3.99, 11.76 or 67.20 p/kWh.")
    return float(PRICES[matches[0]])


def score_episode(demand, reference, prices, previous_price, *,
                  weights=None, tolerance=0.10,
                  bill_allowance=0.0):
    """Score time x household arrays (48,H) and a shared price vector (48,).

    B(x)=(x/(1+x))**2. Q=sum across households; T=mean(Q0).
    demand = mean B(max(abs(Q-T)/T-tolerance, 0))
    energy = mean_households B(abs(E-E0)/E0), over all 48 intervals
    switch = mean(delta_price != 0), across 12 tariff transitions (entry plus 11 internal changes)
    jump   = mean((delta_price/(67.20-3.99))**2)
    bill   = mean_households B(max((bill-bill0)/bill0-allowance, 0))
    cost   = sum(weight*term)/sum(weights), in [0,1].
    Reference uses Normal for all 48 intervals. No forced terminal price change.
    """
    weights = _weights(weights)
    if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
        raise ValueError("tolerance must lie in [0,1).")
    if not np.isfinite(bill_allowance) or bill_allowance < 0:
        raise ValueError("bill_allowance must be finite and nonnegative.")
    demand, reference, prices = map(lambda x: np.asarray(x, dtype=np.float64),
                                    (demand, reference, prices))
    if (demand.ndim != 2 or demand.shape[0] != 48 or demand.shape[1] == 0
            or reference.shape != demand.shape or prices.shape != (48,)):
        raise ValueError("Require demand/reference (48,households), prices (48,).")
    if any(not np.isfinite(x).all() for x in (demand, reference, prices)):
        raise ValueError("All scoring inputs must be finite.")
    if (demand < 0).any() or (reference < 0).any():
        raise ValueError("Consumption cannot be negative.")
    prices = np.array([_price(p) for p in prices])
    previous_price = _price(previous_price)
    if not np.array_equal(prices, np.repeat(prices[::4], 4)):
        raise ValueError("Each control tariff must last four half-hours.")
    reference_prices = np.full(48, PRICES[1])
    target = reference.sum(axis=1).mean()
    energy0 = reference.sum(axis=0)
    bill0 = (reference * reference_prices[:, None]).sum(axis=0)
    if target <= 1e-8 or (energy0 <= 1e-8).any() or (bill0 <= 1e-8).any():
        raise ValueError("Reference target, household energies and bills must be positive.")
    def bounded(error):
        return np.square(error / (1 + error))
    changes = np.diff(np.r_[previous_price, prices[::4]])
    parts = {
        'demand': bounded(np.maximum(np.abs(demand.sum(axis=1) - target)
                                     / target - tolerance, 0)).mean(),
        'energy': bounded(np.abs(demand.sum(axis=0) - energy0) / energy0).mean(),
        'switch': np.mean(changes != 0),
        'jump': np.mean(np.square(changes / np.ptp(PRICES))),
        'bill': bounded(np.maximum((demand * prices[:, None]).sum(axis=0)
                                   / bill0 - 1 - bill_allowance, 0)).mean(),
    }
    parts = {key: float(value) for key, value in parts.items()}
    return float(sum(weights[k] * v for k, v in parts.items()) / sum(weights.values())), parts


class DataPreprocessor:
    """Single input contract shared by DQN observations and the frozen GRU.

    Supply means/maxima fitted on TRAINING data, in fixed household order.
    Raw histories retain NaN/None; only model features fill them with training
    means, and append availability flags (1=available, 0=missing). Predictions
    subsequently inserted into history are available inputs, not measurements.
    No statistics are fitted here and no missing prices/timestamps are guessed.
    Canonical units are kWh per half-hour and pence/kWh. Optional kW input is
    converted to half-hour energy. Values above training maxima are not clipped.
    """

    def __init__(self, maxima, means=None, household_ids=None):
        self.maxima = np.array(maxima, dtype=np.float32, copy=True)
        if (self.maxima.ndim != 1 or not self.maxima.size
                or not np.isfinite(self.maxima).all() or (self.maxima <= 0).any()):
            raise ValueError("Require positive finite training maxima, one per household.")
        self.means = None if means is None else np.array(means, dtype=np.float32, copy=True)
        if self.means is not None and (self.means.shape != self.maxima.shape
                or not np.isfinite(self.means).all() or (self.means < 0).any()):
            raise ValueError("Require finite nonnegative training means matching maxima.")
        self.household_ids = None if household_ids is None else tuple(household_ids)
        if self.household_ids is not None and (len(self.household_ids) != self.maxima.size
                or len(set(self.household_ids)) != self.maxima.size):
            raise ValueError("Provide unique household IDs matching training statistics.")

    def history(self, values, *, unit='kwh', household_ids=None):
        """Validate/copy (...,10,H) input; None/NaN remain missing internally."""
        if household_ids is not None:
            if self.household_ids is None or tuple(household_ids) != self.household_ids:
                raise ValueError("Household order must exactly match the saved model.")
        data = np.array(values, dtype=np.float64, copy=True)
        if data.ndim not in (2, 3) or data.shape[-2:] != (10, len(self.maxima)):
            raise ValueError("History must have shape (10,H) or (episodes,10,H).")
        if data.size == 0 or np.isinf(data).any() or (data < 0).any():
            raise ValueError("Use nonnegative readings and None/NaN for missing values.")
        if unit == 'kw':
            data *= 0.5
        elif unit != 'kwh':
            raise ValueError("Demand unit must be 'kwh' per half-hour or average 'kw'.")
        if np.isnan(data).any() and self.means is None:
            raise ValueError("Missing demand requires training-only household means.")
        return data

    def history_features(self, values):
        """Float32 (...,10,2H): scaled demand followed by availability flags."""
        data = self.history(values)
        available = np.isfinite(data)
        filled = np.where(available, data, self.means if self.means is not None else 0)
        # Cast before scaling to match the response model's training pipeline.
        with np.errstate(over='ignore', invalid='ignore'):
            scaled = filled.astype(np.float32) / self.maxima
        if not np.isfinite(scaled).all():
            raise ValueError("Demand cannot be represented as finite float32 model inputs.")
        return np.concatenate((scaled, available.astype(np.float32)), axis=-1)

    @staticmethod
    def price(value, *, unit='pence'):
        """Return canonical tariff and its normalized float32 model feature."""
        if value is None or not np.isscalar(value):
            raise ValueError("Price must be one known numeric tariff.")
        value = float(value)
        if unit == 'gbp':
            value *= 100
        elif unit != 'pence':
            raise ValueError("Price unit must be 'pence' or 'gbp' per kWh.")
        canonical = _price(value)
        return canonical, np.float32((canonical - PRICES[0]) / np.ptp(PRICES))

    @staticmethod
    def time(timestamps):
        return time_features(timestamps)


class PriceEnv(Env):
    """DQN selects one shared tariff: 0=Low, 1=Normal, 2=High.

    response(history_kwh, next_price) must deterministically return H predictions
    from a (10,H) history, without modifying its inputs or holding hidden state.
    histories has shape (episodes,10,H), with nonnegative readings; None/NaN denote missing values.
    starts gives the first CONTROL interval timestamp of each episode.
    The response adapter supplies its shared preprocessor automatically. For other
    predictors, provide demand_scale and training-only household_means. Raw NaN
    histories reach the predictor so it can preserve missingness flags.
    Only training-partition histories should be supplied to a training environment.

    Observation includes scaled history, current price, preprocessed time pair,
    remaining decisions, fixed reference target, cumulative energy/bills, and
    reference totals, and running demand/switch/jump penalties. History is visible
    because it drives the response model.
    Terminal reward is -cost; gamma=1 preserves the exact episodic objective.
    """
    metadata = {'render_modes': []}

    def __init__(self, response, histories, starts, demand_scale=None, *,
                 household_means=None, household_ids=None, demand_unit='kwh',
                 previous_prices=11.76, weights=None,
                 tolerance=0.10, bill_allowance=0.0):
        super().__init__()
        self.response = response
        self.preprocessor = getattr(response, 'preprocessor', None)
        if self.preprocessor is None:
            if demand_scale is None:
                raise ValueError("Provide training demand_scale for this response function.")
            shape = np.asarray(histories).shape
            if len(shape) != 3:
                raise ValueError('Environment histories must have shape (episodes,10,H).')
            h = shape[-1]
            scales = np.broadcast_to(demand_scale, (h,))
            self.preprocessor = DataPreprocessor(scales, household_means, household_ids)
        else:
            if demand_scale is not None and not np.array_equal(
                    np.broadcast_to(np.asarray(demand_scale, dtype=np.float32), self.preprocessor.maxima.shape),
                    self.preprocessor.maxima):
                raise ValueError("Demand scales must match the frozen response model.")
            if household_means is not None and not np.array_equal(
                    np.asarray(household_means, dtype=np.float32), self.preprocessor.means):
                raise ValueError("Household means must match the frozen response model.")
        self.histories = self.preprocessor.history(histories, unit=demand_unit,
                                                   household_ids=household_ids)
        if self.histories.ndim != 3:
            raise ValueError("Environment histories must have shape (episodes,10,H).")
        n, _, h = self.histories.shape
        self.starts = pd.DatetimeIndex(starts)
        if len(self.starts) != n or self.starts.hasnans:
            raise ValueError("Provide one valid control start timestamp per episode.")
        self.scale = self.preprocessor.maxima.astype(np.float64)
        self.previous_prices = np.broadcast_to(previous_prices, (n,)).copy()
        self.previous_prices = np.array([self.preprocessor.price(p)[0] for p in self.previous_prices])
        self.weights = _weights(weights)
        if not np.isfinite(tolerance) or not 0 <= tolerance < 1:
            raise ValueError("tolerance must lie in [0,1).")
        if not np.isfinite(bill_allowance) or bill_allowance < 0:
            raise ValueError("bill_allowance must be finite and nonnegative.")
        self.tolerance, self.bill_allowance = tolerance, bill_allowance
        # Preprocess all decision times, including the terminal observation.
        self.times = np.stack([self.preprocessor.time(pd.date_range(t, periods=13, freq='2h'))
                               for t in self.starts])
        self.action_space = Discrete(3)
        self.observation_space = Dict({
            'history': Box(0, np.inf, shape=(20 * h,), dtype=np.float32),
            'context': Box(0, np.inf, shape=(8,), dtype=np.float32),
            'accounts': Box(0, np.inf, shape=(4 * h,), dtype=np.float32),
        })
        self.done = True

    def _advance(self, history, price):
        # Own this array even if the predictor reuses an output buffer.
        demand = np.array(self.response(history.copy(), price), dtype=np.float64, copy=True)
        if demand.shape != self.scale.shape or not np.isfinite(demand).all() or (demand < 0).any():
            raise ValueError("Response must return one finite nonnegative prediction per household.")
        return np.concatenate((history[1:], demand[None]), axis=0), demand

    def _observation(self):
        energy = np.sum(self.demands, axis=0) if self.demands else np.zeros_like(self.scale)
        bill = (np.asarray(self.demands) * np.asarray(self.prices)[:, None]).sum(axis=0) if self.demands else np.zeros_like(self.scale)
        # Running penalties make the eventual terminal reward observable too.
        control = np.asarray(self.demands)
        demand_cost = 0.0
        if len(control):
            error = np.maximum(np.abs(control.sum(axis=1) - self.target) / self.target
                               - self.tolerance, 0)
            demand_cost = np.square(error / (1 + error)).sum() / 48
        price_path = [self.initial_price, *self.prices[::4]]
        changes = np.diff(price_path)
        switch_cost = np.count_nonzero(changes) / 12
        jump_cost = np.square(changes / np.ptp(PRICES)).sum() / 12
        return {
            'history': self.preprocessor.history_features(self.history).ravel(),
            'context': np.asarray([self.preprocessor.price(self.current_price)[1],
                                   *self.times[self.episode, self.block],
                                   (12 - self.block) / 12,
                                   self.target / self.scale.sum(),
                                   demand_cost, switch_cost, jump_cost], dtype=np.float32),
            'accounts': np.stack((energy / (48 * self.scale),
                                  bill / (48 * self.scale * PRICES[-1]),
                                  self.energy0 / (48 * self.scale),
                                  self.bill0 / (48 * self.scale * PRICES[-1]))).astype(np.float32).ravel(),
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.done = True  # A failed reset must not leave an old episode step-able.
        index = (options or {}).get('episode_index')
        if index is None:
            index = int(self.np_random.integers(len(self.histories)))
        if not isinstance(index, (int, np.integer)) or not 0 <= index < len(self.histories):
            raise ValueError("Invalid episode_index.")
        self.episode, self.block = int(index), 0
        self.history = self.histories[index].copy()
        self.current_price = self.initial_price = self.previous_prices[index]
        history = self.history.copy()
        reference = []
        reference_prices = np.full(48, PRICES[1])
        for price in reference_prices:
            history, demand = self._advance(history, float(price))
            reference.append(demand)
        self.reference = np.asarray(reference)
        self.target = self.reference.sum(axis=1).mean()
        self.energy0 = self.reference.sum(axis=0)
        self.bill0 = (self.reference * reference_prices[:, None]).sum(axis=0)
        if self.target <= 1e-8 or (self.energy0 <= 1e-8).any() or (self.bill0 <= 1e-8).any():
            raise ValueError("Reference target, energies and bills must be positive.")
        self.demands, self.prices = [], []
        self.done = False
        return self._observation(), {}

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a new episode.")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0, 1 or 2.")
        price = float(PRICES[int(action)])
        for _ in range(4):
            self.history, demand = self._advance(self.history, price)
            self.demands.append(demand)
            self.prices.append(price)
        self.current_price = price
        self.block += 1
        self.done = self.block == 12
        reward, info = 0.0, {'price_pence_per_kwh': price}
        if self.done:
            cost, parts = score_episode(self.demands, self.reference, self.prices,
                                        self.initial_price,
                                        weights=self.weights, tolerance=self.tolerance,
                                        bill_allowance=self.bill_allowance)
            reward = -cost
            info.update(cost=cost, metrics=parts)
        return self._observation(), reward, self.done, False, info

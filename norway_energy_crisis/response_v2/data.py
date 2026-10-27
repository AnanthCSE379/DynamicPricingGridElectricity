"""Training-only features, measured labels, and safe forecast inputs."""

from dataclasses import dataclass, fields
from pathlib import Path
import json

import numpy as np
import torch

from norway_energy_crisis.clean_data import digest, require, save_json
from norway_energy_crisis.preprocess import Preprocessor, calendar, encode_static

FEATURES = [
    "demand_scaled",
    "price_standardized",
    "temperature_standardized",
    "hour_noon_peak",
    "hour_sine_phase",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
    "demand_log1p",
    "price_change_standardized",
    "price_change_available",
]
VERSION = "norway_response_v2"


def origins_for(partitions, split, horizon=72, daily=True):
    """Require the full observed history and forecast to stay inside one split."""
    require(1 <= horizon <= 120, "Unsupported horizon.")
    result = [
        t
        for t in range(24, len(partitions) - horizon + 1)
        if (not daily or t % 24 == 0)
        and np.all(partitions[t - 24 : t + horizon] == split)
    ]
    return np.asarray(result, dtype=np.int64)


@dataclass(frozen=True)
class ForecastInputs:
    """Permitted model inputs only; measured targets are deliberately absent."""

    history: torch.Tensor
    static: torch.Tensor
    scales: torch.Tensor
    prices: torch.Tensor
    calendar: torch.Tensor

    def to(self, device):
        return ForecastInputs(
            **{f.name: getattr(self, f.name).to(device) for f in fields(self)}
        )

    def slice(self, start, stop):
        return ForecastInputs(
            **{f.name: getattr(self, f.name)[start:stop] for f in fields(self)}
        )


class FeatureBuilder:
    """Apply the checkpoint's raw-unit contract without fitting at inference."""

    def __init__(self, state):
        require(state["version"] == VERSION, "Unknown feature schema.")
        require(state["history_features"] == FEATURES, "Feature order changed.")
        self.state = state
        self.base = Preprocessor(state)
        self.scales = self.base.scales
        self.static = self.base.static
        self.ids = self.base.ids
        self.lookup = {value: index for index, value in enumerate(self.ids)}

    def history(self, demand, prices, temperature, times, households):
        x = self.base.history_features(demand, prices, temperature, times, households)
        require(x.shape[1] == 24, "Exactly 24 history hours are required.")
        stamps = np.asarray(times, dtype="datetime64[ns]")
        if stamps.ndim == 1:
            stamps = np.broadcast_to(stamps, x.shape[:2])
        require(
            stamps.shape == x.shape[:2]
            and np.all(np.diff(stamps.astype(np.int64), axis=1) == 3600 * 10**9),
            "History timestamps must be consecutive dataset hours.",
        )
        changes = np.zeros(x.shape[:2], dtype=np.float32)
        changes[:, 1:] = np.diff(x[:, :, 1], axis=1)
        available = np.ones_like(changes)
        available[:, 0] = 0
        return np.concatenate(
            (x, np.log1p(x[:, :, :1]), changes[..., None], available[..., None]),
            axis=-1,
        ).astype(np.float32)

    def inputs(self, demand, prices, temperature, times, household_ids, future_prices):
        require(len(set(household_ids)) == len(household_ids), "Duplicate IDs.")
        require(all(i in self.lookup for i in household_ids), "Unknown household ID.")
        indices = np.array([self.lookup[i] for i in household_ids])
        history = self.history(demand, prices, temperature, times, indices)
        future_prices = np.asarray(future_prices, dtype=np.float32)
        require(
            future_prices.ndim == 2 and len(future_prices) == len(indices),
            "Future prices must be household by horizon.",
        )
        require(1 <= future_prices.shape[1] <= 120, "Unsupported forecast length.")
        stamps = np.asarray(times, dtype="datetime64[ns]")
        if stamps.ndim == 1:
            stamps = np.broadcast_to(stamps, history.shape[:2])
        future_times = stamps[:, -1, None] + np.arange(1, future_prices.shape[1] + 1)[
            None
        ] * np.timedelta64(1, "h")
        return tensor_inputs(
            history,
            self.static[indices],
            self.scales[indices],
            self.base.price(future_prices),
            calendar(future_times),
        )


def tensor_inputs(history, static, scales, prices, target_calendar):
    values = [history, static, scales, prices, target_calendar]
    require(all(np.isfinite(v).all() for v in values), "Nonfinite forecast input.")
    return ForecastInputs(
        *(torch.tensor(np.asarray(value), dtype=torch.float32) for value in values)
    )


class ConsumerData:
    """Keep all metered households, with a distinct hourly-spot fine-tuning cohort."""

    def __init__(self, cleaned, prepared, output=None):
        self.cleaned, self.prepared = Path(cleaned), Path(prepared)
        manifest = json.loads((self.cleaned / "manifest.json").read_text())
        prepared_manifest = json.loads((self.prepared / "manifest.json").read_text())
        require(
            prepared_manifest["cleaned_manifest_sha256"]
            == digest(self.cleaned / "manifest.json"),
            "Cleaned/prepared provenance mismatch.",
        )
        for folder, description in (
            (self.cleaned, manifest),
            (self.prepared, prepared_manifest),
        ):
            for name, expected in description["output_hashes"].items():
                require(digest(folder / name) == expected, f"Data changed: {name}")
        for name in ("demand", "prices", "temperature", "times", "area_index"):
            setattr(self, name, np.load(self.cleaned / f"{name}.npy", mmap_mode="r"))
        self.times = self.times.astype("datetime64[ns]")
        self.partitions = np.load(self.prepared / "partitions.npy")
        self.primary = np.flatnonzero(np.load(self.cleaned / "cohort_mask.npy"))
        self.all_households = np.arange(len(self.demand))
        self.ids = np.load(self.cleaned / "household_ids.npy").tolist()
        old = json.loads((self.prepared / "preprocessing.json").read_text())
        require(
            [self.ids[i] for i in self.primary] == old["household_ids"], "ID order."
        )
        train = self.partitions == 0
        scales = np.quantile(self.demand[:, train], 0.95, axis=1).astype(np.float32)
        scales = np.where(scales > 0, scales, 1).astype(np.float32)
        require(np.array_equal(scales[self.primary], old["demand_scales"]), "Scales.")
        static, _ = encode_static(
            np.load(self.cleaned / "static_codes.npy"),
            self.area_index,
            old["static_categories"],
        )
        self.state = {
            **old,
            "version": VERSION,
            "household_ids": self.ids,
            "primary_household_ids": [self.ids[i] for i in self.primary],
            "demand_scales": scales.tolist(),
            "static_encoded": static.tolist(),
            "history_features": FEATURES,
            "candidate_features": ["price_standardized", "price_change_standardized"],
            "head_target": "next-hour household kWh, returned nonnegative",
        }
        self.features = FeatureBuilder(self.state)
        self.origins = {
            split: origins_for(self.partitions, number, 72, split != "train")
            for number, split in enumerate(("train", "validation", "test"))
        }
        require(all(len(v) for v in self.origins.values()), "Empty partition.")
        self.fingerprint = {
            "cleaned": digest(self.cleaned / "manifest.json"),
            "prepared": digest(self.prepared / "manifest.json"),
        }
        if output is not None:
            output = Path(output)
            output.mkdir(parents=True, exist_ok=True)
            for name, value in (
                ("preprocessing.json", self.state),
                ("manifest.json", self.fingerprint),
                ("origins.json", {k: v.tolist() for k, v in self.origins.items()}),
            ):
                path = output / name
                if path.exists():
                    require(
                        json.loads(path.read_text()) == value, "Prepared v2 changed."
                    )
                else:
                    save_json(path, value)

    def validate_indices(self, households, origins, horizon):
        require(1 <= horizon <= 120, "Forecast horizon must be 1 to 120 hours.")
        h, t = np.asarray(households, dtype=int), np.asarray(origins, dtype=int)
        require(h.shape == t.shape and h.ndim == 1, "Index shapes.")
        require(
            ((h >= 0) & (h < len(self.ids))).all()
            and ((t >= 24) & (t + horizon <= len(self.times))).all(),
            "Index bounds.",
        )
        parts = self.partitions[t[:, None] + np.arange(-24, horizon)]
        require(np.all(parts == parts[:, :1]), "Window crosses partition.")
        return h, t

    def inputs(self, households, origins, horizon):
        """Read history and recorded scenario prices, never future measured loads.

        Longer price paths condition retrospective rollouts; they are not a claim
        that several days of market prices were known at the forecast origin.
        """
        h, t = self.validate_indices(households, origins, horizon)
        past = t[:, None] + np.arange(-24, 0)
        area = self.area_index[h]
        history = self.features.history(
            self.demand[h[:, None], past],
            self.prices[area[:, None], past],
            self.temperature[area[:, None], past],
            self.times[past],
            h,
        )
        future = t[:, None] + np.arange(horizon)
        return tensor_inputs(
            history,
            self.features.static[h],
            self.features.scales[h],
            self.features.base.price(self.prices[area[:, None], future]),
            calendar(self.times[future]),
        )

    def targets(self, households, origins, horizon):
        """Supervised labels; never pass this result to the forecast function."""
        h, t = self.validate_indices(households, origins, horizon)
        return torch.tensor(
            np.asarray(self.demand[h[:, None], t[:, None] + np.arange(horizon)]),
            dtype=torch.float32,
        )


class TransitionSampler:
    """Cover observed rare transitions with importance-corrected sampling.

    The target remains uniform household/origin MSE. Proposal strata concern
    observed training consumption, never fabricated recovery or price classes.
    """

    def __init__(self, data, households, seed):
        self.households = np.asarray(households)
        self.origins = data.origins["train"]
        self.rng = np.random.default_rng(seed)
        self.size = len(households) * len(self.origins)
        previous = np.asarray(data.demand[self.households[:, None], self.origins - 1])
        current = np.asarray(data.demand[self.households[:, None], self.origins])
        zero = (previous == 0) & (current > 0)
        low = (previous > 0) & (previous <= 0.1) & (current > 0.1)
        self.pools = [np.flatnonzero(zero), np.flatnonzero(low)]
        self.flags = [zero.ravel(), low.ravel()]
        self.mass = np.array([0.05 if zero.any() else 0, 0.15 if low.any() else 0])
        self.uniform_mass = 1 - self.mass.sum()
        self.counts = {
            "zero_to_positive": int(zero.sum()),
            "low_to_high": int(low.sum()),
        }

    def sample(self, size):
        keys = self.rng.integers(self.size, size=size)
        choices = self.rng.random(size)
        end = self.uniform_mass
        for pool, mass in zip(self.pools, self.mass):
            selected = (choices >= end) & (choices < end + mass)
            if selected.any():
                keys[selected] = self.rng.choice(pool, size=int(selected.sum()))
            end += mass
        probability = np.full(size, self.uniform_mass / self.size)
        for pool, flag, mass in zip(self.pools, self.flags, self.mass):
            if len(pool):
                probability += mass / len(pool) * flag[keys]
        weights = (1 / self.size / probability).astype(np.float32)
        h = self.households[keys // len(self.origins)]
        t = self.origins[keys % len(self.origins)]
        return h, t, torch.from_numpy(weights)


def support_report(data):
    """Describe training support; these scores are not calibrated error bounds."""
    training = data.partitions == 0
    grid = np.array([0.05, *np.arange(0.1, 2.01, 0.1)])
    months = (data.times.astype("datetime64[M]").astype(int) % 12) + 1
    seasons = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "autumn": [9, 10, 11],
    }
    counts = {}
    for i, area in enumerate(data.state["areas"]):
        counts[area] = {}
        for season, members in {"all": list(range(1, 13)), **seasons}.items():
            selected = training & np.isin(months, members)
            prices = np.asarray(data.prices[i, selected])
            counts[area][season] = {
                "regional_hours": len(prices),
                "within_0_05_counts": [
                    int((np.abs(prices - p) <= 0.05).sum()) for p in grid
                ],
                "within_0_05_fraction": [
                    float((np.abs(prices - p) <= 0.05).mean()) for p in grid
                ],
            }
    rng = np.random.default_rng(42)

    def summaries(inputs):
        h = inputs.history.numpy()
        return np.column_stack(
            (
                h[:, -1, 0],
                h[:, :, 0].mean(1),
                h[:, :, 0].std(1),
                np.abs(np.diff(h[:, :, 0], axis=1)).max(1),
                (h[:, :, 0] == 0).mean(1),
                h[:, -1, 1],
                h[:, :, 1].mean(1),
                np.abs(h[:, :, 10]).max(1),
                h[:, -1, 2],
            )
        )

    h = rng.choice(data.primary, size=4096)
    t = rng.choice(data.origins["train"], size=4096)
    reference = summaries(data.inputs(h, t, 1))
    mean, std = reference.mean(0), reference.std(0)
    std = np.where(std > 1e-6, std, 1)
    reference = (reference - mean) / std
    chosen = data.origins["validation"][
        :: max(len(data.origins["validation"]) // 6, 1)
    ][:6]
    h = np.tile(data.primary, len(chosen))
    t = np.repeat(chosen, len(data.primary))
    query = (summaries(data.inputs(h, t, 1)) - mean) / std
    distances = []
    for start in range(0, len(query), 128):
        delta = query[start : start + 128, None] - reference[None]
        distances.extend(np.sqrt((delta**2).sum(-1).min(1)).tolist())
    return {
        "grid": grid.tolist(),
        "price_support": counts,
        "summary_names": [
            "last_load",
            "mean_load",
            "load_std",
            "largest_load_change",
            "zero_fraction",
            "last_price",
            "mean_price",
            "largest_price_change",
            "temperature",
        ],
        "training_summary_mean": mean.tolist(),
        "training_summary_std": std.tolist(),
        "reference_windows": len(reference),
        "validation_nearest_summary_distance_quantiles": np.quantile(
            distances, [0, 0.25, 0.5, 0.75, 0.95, 1]
        ).tolist(),
        "scope": "Training-fitted descriptive support; not reliability guarantees.",
    }

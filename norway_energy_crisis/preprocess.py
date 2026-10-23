"""Training-only preprocessing shared by Norwegian response training and inference."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json

CALENDAR_NAMES = [
    "hour_noon_peak",
    "hour_sine_phase",
    "weekday_sin",
    "weekday_cos",
    "year_sin",
    "year_cos",
]
HISTORY_NAMES = [
    "demand_scaled",
    "price_standardized",
    "temperature_standardized",
    *CALENDAR_NAMES,
]
SPLITS = {
    "train": ("2020-10-01", "2022-01-01"),
    "validation": ("2022-01-01", "2022-02-01"),
    "test": ("2022-02-01", "2022-04-01"),
}


def calendar(values):
    """Return six calendar features, preserving the input timestamp shape.

    The first hour feature peaks at noon: (1 - cos(hour phase)) / 2.
    The second is (1 + sin(hour phase)) / 2. Weekday and annual cycles use
    ordinary sine/cosine pairs. These are local dataset hours, not UTC hours.
    """
    values = np.asarray(values)
    shape = values.shape
    timestamps = pd.DatetimeIndex(pd.to_datetime(values.ravel()))
    require(not timestamps.hasnans, "Invalid calendar timestamp.")
    hour_phase = 2 * np.pi * timestamps.hour.to_numpy() / 24
    weekday_phase = 2 * np.pi * timestamps.dayofweek.to_numpy() / 7
    year_phase = (
        2
        * np.pi
        * (timestamps.dayofyear.to_numpy() - 1 + timestamps.hour.to_numpy() / 24)
        / np.where(timestamps.is_leap_year, 366, 365)
    )
    return (
        np.column_stack(
            (
                (1 - np.cos(hour_phase)) / 2,
                (1 + np.sin(hour_phase)) / 2,
                np.sin(weekday_phase),
                np.cos(weekday_phase),
                np.sin(year_phase),
                np.cos(year_phase),
            )
        )
        .astype(np.float32)
        .reshape(*shape, 6)
    )


def eligible_origins(partitions, split_id, daily=False):
    """Find origins whose full 24-hour history and targets stay in one split.

    Apply the same eligibility rule to both history lengths so their evaluations
    use identical days. Daily origins are midnight offsets in this hourly panel.
    """
    result = [
        origin
        for origin in range(24, len(partitions) - 23)
        if (not daily or origin % 24 == 0)
        and np.all(partitions[origin - 24 : origin + 24] == split_id)
    ]
    return np.asarray(result, dtype=np.int64)


def encode_static(codes, area_index, categories=None):
    """One-hot encode region and survey descriptors, adding an unknown category.

    When categories are supplied, reuse them unchanged; unseen and missing values
    go to the explicit unknown column instead of being dropped or imputed.
    """
    codes = np.column_stack((area_index, codes))
    if categories is None:
        categories = [
            sorted(
                float(column_values)
                for column_values in np.unique(codes[:, column])
                if np.isfinite(column_values)
            )
            for column in range(codes.shape[1])
        ]
    blocks = []
    for column, values in enumerate(categories):
        column_values = codes[:, column]
        block = np.column_stack(
            [column_values == value for value in values]
            + [~np.isin(column_values, values)]
        )
        blocks.append(block.astype(np.float32))
    return np.concatenate(blocks, axis=1), categories


class Preprocessor:
    """Apply saved training statistics to response-model inputs.

    Feature order, household order and scaling are shared by training, inference
    and DQN simulation. Prices and demand are never clipped to the training range.
    """

    def __init__(self, state):
        self.state = state
        self.scales = np.asarray(state["demand_scales"], dtype=np.float32)
        self.ids = list(state["household_ids"])
        self.static = np.asarray(state["static_encoded"], dtype=np.float32)

    def history_features(self, demand, prices, temperature, times, household_indices):
        """Build float32 (batch, history, 9) inputs from raw household histories.

        Use kWh demand, NOK/kWh prices and Celsius temperature. Household indices
        select fitted demand scales; shared or per-household timestamps are accepted.
        """
        demand, prices, temperature = [
            np.asarray(values, dtype=np.float32)
            for values in (demand, prices, temperature)
        ]
        require(
            demand.ndim == 2
            and prices.shape == demand.shape
            and temperature.shape == demand.shape,
            "Demand, price and temperature must have matching (batch,history) shapes.",
        )
        indices = np.asarray(household_indices, dtype=int)
        require(
            indices.shape == (len(demand),)
            and ((indices >= 0) & (indices < len(self.ids))).all(),
            "Invalid household indices.",
        )
        require(
            np.isfinite(demand).all()
            and (demand >= 0).all()
            and np.isfinite(prices).all()
            and np.isfinite(temperature).all(),
            "Invalid history values.",
        )
        calendar_features = calendar(times)
        if calendar_features.ndim == 2:
            calendar_features = np.broadcast_to(calendar_features, (*demand.shape, 6))
        require(
            calendar_features.shape == (*demand.shape, 6),
            "Timestamp/history shape mismatch.",
        )
        return np.concatenate(
            (
                (demand / self.scales[indices, None])[..., None],
                ((prices - self.state["price_mean"]) / self.state["price_std"])[
                    ..., None
                ],
                (
                    (temperature - self.state["temperature_mean"])
                    / self.state["temperature_std"]
                )[..., None],
                calendar_features,
            ),
            axis=-1,
        ).astype(np.float32)

    def price(self, prices):
        """Standardize continuous NOK/kWh prices using saved training statistics."""
        prices = np.asarray(prices, dtype=np.float32)
        require(
            np.isfinite(prices).all(),
            "Candidate price must be finite; negative prices are allowed.",
        )
        return (prices - self.state["price_mean"]) / self.state["price_std"]


class PreparedData:
    """Read verified prepared arrays without loading the full panel into memory."""

    def __init__(self, folder, verify=True):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "manifest.json").read_text())
        if verify:
            for name, expected in self.manifest["output_hashes"].items():
                require(
                    digest(self.folder / name) == expected,
                    f"Prepared file changed: {name}",
                )
        self.state = json.loads((self.folder / "preprocessing.json").read_text())
        self.preprocessor = Preprocessor(self.state)
        for name in [
            "demand",
            "prices",
            "temperature",
            "times",
            "area_index",
            "static",
            "partitions",
        ]:
            setattr(
                self,
                name,
                np.load(self.folder / f"{name}.npy", mmap_mode="r", allow_pickle=False),
            )
        self.origins = {
            name: np.load(self.folder / f"{name}_origins.npy", allow_pickle=False)
            for name in SPLITS
        }
        self.households = len(self.preprocessor.ids)

    def batch(self, households, origins, lookback, horizon=24):
        """Build supervised response-training examples and measured targets.

        Each example is one household at one origin. This helper reads future demand
        for labels; the DQN environment must use EpisodeProvider snapshots instead.
        """
        require(
            lookback in (5, 24) and 1 <= horizon <= 24,
            "Use 5h/24h history and at most 24 forecast hours.",
        )
        households, origins = np.asarray(households, dtype=int), np.asarray(
            origins, dtype=int
        )
        require(
            households.shape == origins.shape and households.ndim == 1,
            "Batch indices must match.",
        )
        history_indices = origins[:, None] + np.arange(-lookback, 0)
        target_indices = origins[:, None] + np.arange(horizon)
        areas = self.area_index[households]
        history = self.preprocessor.history_features(
            self.demand[households[:, None], history_indices],
            self.prices[areas[:, None], history_indices],
            self.temperature[areas[:, None], history_indices],
            self.times[history_indices],
            households,
        )
        # The original previous day is also available for the seasonal baseline.
        previous_day = self.demand[
            households[:, None], origins[:, None] + np.arange(-24, 0)
        ]
        result = dict(
            history=history,
            static=np.asarray(self.static[households]),
            prices=self.preprocessor.price(self.prices[areas[:, None], target_indices]),
            calendar=calendar(self.times[target_indices]),
            target=np.asarray(self.demand[households[:, None], target_indices]),
            scales=self.preprocessor.scales[households],
            yesterday=np.asarray(previous_day),
            household_indices=households,
            area_index=areas,
            raw_prices=np.asarray(self.prices[areas[:, None], target_indices]),
        )
        return {
            key: torch.from_numpy(np.array(value, copy=True))
            for key, value in result.items()
        }


def preprocess(cleaned=ROOT / "cleaned", output=ROOT / "prepared"):
    """Fit the shared preprocessing contract on the training partition only.

    Select hourly-spot households, encode their assumed-stable descriptors and
    save separate chronological origins. Validation/test values cannot influence
    demand scales, regional price/temperature statistics or reporting price bands.
    """
    cleaned, output = Path(cleaned), Path(output)
    require(not output.exists(), f"Output exists: {output}; choose a new directory.")
    manifest = json.loads((cleaned / "manifest.json").read_text())
    for name, expected in manifest["output_hashes"].items():
        require(digest(cleaned / name) == expected, f"Cleaned file changed: {name}")
    cohort_mask = np.load(cleaned / "cohort_mask.npy")
    household_ids = np.load(cleaned / "household_ids.npy")[cohort_mask]
    demand = np.load(cleaned / "demand.npy", mmap_mode="r")[cohort_mask]
    area_index = np.load(cleaned / "area_index.npy")[cohort_mask]
    prices = np.load(cleaned / "prices.npy")
    temperature = np.load(cleaned / "temperature.npy")
    times = np.load(cleaned / "times.npy")
    dates = pd.DatetimeIndex(times)
    partitions = np.full(len(times), 255, dtype=np.uint8)
    for split_id, (_, (start, end)) in enumerate(SPLITS.items()):
        partitions[(dates >= start) & (dates < end)] = split_id
    require((partitions != 255).all(), "Unexpected time outside configured splits.")
    training_mask = partitions == 0
    require(
        np.isfinite(demand).all() and np.isfinite(temperature).all(),
        "Missing data blocks preprocessing.",
    )
    scales = np.quantile(demand[:, training_mask], 0.95, axis=1).astype(np.float32)
    scales = np.where(scales > 0, scales, 1.0).astype(np.float32)
    static, categories = encode_static(
        np.load(cleaned / "static_codes.npy")[cohort_mask], area_index
    )
    state = dict(
        version=1,
        household_ids=household_ids.tolist(),
        demand_scales=scales.tolist(),
        price_mean=float(prices[:, training_mask].mean()),
        price_std=float(prices[:, training_mask].std()),
        temperature_mean=float(temperature[:, training_mask].mean()),
        temperature_std=float(temperature[:, training_mask].std()),
        demand_scaling="household training-only 95th percentile; no clipping",
        static_categories=categories,
        static_encoded=static.tolist(),
        static_questions=["price_area", *manifest["static_questions"]],
        static_labels=manifest["category_labels"],
        history_features=HISTORY_NAMES,
        calendar_features=CALENDAR_NAMES,
        areas=manifest["areas"],
        assumptions=manifest["assumptions"],
        forecast_temperature=(
            "Persist last known temperature at each forecast origin; never use future"
            " measured temperature."
        ),
    )
    require(
        state["price_std"] > 0 and state["temperature_std"] > 0,
        "Training price/temperature has zero variance.",
    )
    # Reporting bands describe training support without constraining prices.
    state["price_band_edges"] = np.quantile(
        prices[:, training_mask], [0.25, 0.5, 0.75]
    ).tolist()
    arrays = dict(
        demand=demand,
        prices=prices.astype(np.float32),
        temperature=temperature,
        times=times,
        area_index=area_index,
        static=static,
        partitions=partitions,
    )
    origins = {
        name: eligible_origins(partitions, split_id, daily=(name != "train"))
        for split_id, name in enumerate(SPLITS)
    }
    require(
        all(len(values) > 0 for values in origins.values()),
        "A split has no usable 24h origins.",
    )
    output.mkdir(parents=True)
    for name, value in arrays.items():
        np.save(output / f"{name}.npy", value, allow_pickle=False)
    for name, value in origins.items():
        np.save(output / f"{name}_origins.npy", value, allow_pickle=False)
    save_json(output / "preprocessing.json", state)
    audit = dict(
        state="complete",
        households=len(household_ids),
        split_dates=SPLITS,
        origins={name: len(value) for name, value in origins.items()},
        split_rule=(
            "All 24 history hours and all 24 target hours inside one split; common"
            " origins for 5h/24h models."
        ),
        cleaned_manifest_sha256=digest(cleaned / "manifest.json"),
    )
    audit["output_hashes"] = {
        path.name: digest(path)
        for path in output.iterdir()
        if path.suffix in (".npy", ".json")
    }
    save_json(output / "manifest.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=ROOT / "cleaned")
    parser.add_argument("--output", type=Path, default=ROOT / "prepared")
    args = parser.parse_args()
    preprocess(args.cleaned, args.output)

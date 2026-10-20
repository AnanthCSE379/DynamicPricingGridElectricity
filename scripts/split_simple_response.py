"""Read original LCL ToU observations and create season-stratified 80/10/10 pairs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SEASONS = ("winter", "spring", "summer", "autumn")
PARTITIONS = ("train", "validation", "test")
PRICES = (3.99, 11.76, 67.20)  # Official LCL tariff amounts, pence per kWh.
PAIR_TYPE = np.dtype([
    ("current_kwh", "f4"), ("next_kwh", "f4"), ("price_id", "u1"),
    ("season_id", "u1"), ("household_index", "i4"), ("start", "i4"),
])


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def allocate(total, proportions):
    """Round proportional counts while preserving the requested total exactly."""
    expected = total * np.asarray(proportions, dtype=float) / np.sum(proportions)
    counts = np.floor(expected).astype(int)
    remaining = total - int(counts.sum())
    counts[np.argsort(-(expected - counts), kind="stable")[:remaining]] += 1
    return counts


def seasonal_split(times, price_ids=None, seed=20261017):
    """Shared time blocks: 80% training and 10% each validation/test per season.

    Allocate held-out time across all three months of a season. Within each
    month, use contiguous blocks and rotate their order to spread event exposure.
    Never independently split households or adjacent demand pairs.
    """
    times = pd.DatetimeIndex(times)
    season_ids = np.array([0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 0])[times.month - 1]
    labels = np.full(len(times), -1, dtype=np.int8)
    for season_id in range(4):
        positions = np.flatnonzero(season_ids == season_id)
        months = times[positions].to_period("M")
        groups = [positions[months == month] for month in np.unique(months)]
        lengths = [len(group) for group in groups]
        counts = allocate(len(positions), [0.8, 0.1, 0.1])
        validation_counts = allocate(int(counts[1]), lengths)
        test_counts = allocate(int(counts[2]), lengths)
        for group, validation_count, test_count in zip(groups, validation_counts, test_counts):
            sizes = [len(group) - validation_count - test_count, validation_count, test_count]
            # January train/validation/test; February test/train/validation; etc.
            order = [(0, 1, 2), (2, 0, 1), (1, 2, 0)][(times[group[0]].month - 1) % 3]
            offset = 0
            for partition_id in order:
                labels[group[offset:offset + sizes[partition_id]]] = partition_id
                offset += sizes[partition_id]
    if (labels < 0).any():
        raise ValueError("Every timestamp must belong to one partition.")
    if price_ids is not None:
        # Block placement can use the known tariff calendar, never consumption.
        # Require repeated dates for every seasonal tariff in every partition.
        base_labels = labels.copy()
        random = np.random.default_rng(seed)
        month_groups = [np.flatnonzero(times.to_period("M") == month) for month in np.unique(times.to_period("M"))]
        date_ids = times.normalize().asi8
        for attempt in range(10_000):
            if attempt:
                for group in month_groups:
                    labels[group] = np.roll(base_labels[group], int(random.integers(len(group))))
            valid = True
            for partition_id in range(3):
                for season_id in range(4):
                    for price_id in range(3):
                        selected = (labels == partition_id) & (season_ids == season_id) & (price_ids == price_id)
                        # At least two event dates and two hours of each tariff.
                        if selected.sum() < 4 or len(np.unique(date_ids[selected])) < 2:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                break
        else:
            raise ValueError("Cannot place 80/10/10 time blocks with repeated seasonal tariff exposure.")
    return labels, season_ids.astype(np.uint8)


def household_groups(path, chunk_rows):
    """Stream the provided household-sorted raw CSV without a household cap."""
    carry = None
    completed = set()
    for chunk in pd.read_csv(path, dtype=str, chunksize=chunk_rows):
        chunk.columns = chunk.columns.str.strip()
        chunk = chunk[chunk.stdorToU.str.strip().eq("ToU")].copy()
        if chunk.empty:
            continue
        chunk["LCLid"] = chunk.LCLid.str.strip()
        if chunk.LCLid.isna().any() or chunk.LCLid.eq("").any():
            raise ValueError("Missing household ID in raw input.")
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        identifiers = chunk.LCLid.to_numpy()
        boundaries = np.r_[0, np.flatnonzero(identifiers[1:] != identifiers[:-1]) + 1, len(chunk)]
        for left, right in zip(boundaries[:-2], boundaries[1:-1]):
            group = chunk.iloc[left:right]
            household_id = str(group.LCLid.iloc[0])
            if household_id in completed:
                raise ValueError("Raw CSV must be grouped by household; a finished ID reappeared.")
            completed.add(household_id)
            yield household_id, group
        carry = chunk.iloc[boundaries[-2]:].copy()
    if carry is not None:
        household_id = str(carry.LCLid.iloc[0])
        if household_id in completed:
            raise ValueError("A finished household reappeared.")
        yield household_id, carry


def clean_household(group, times):
    """Keep measured readings; collapse identical duplicates and exclude conflicts."""
    dates = pd.to_datetime(group.DateTime.str.slice(0, 19), errors="coerce")
    demand = pd.to_numeric(group["KWH/hh (per half hour)"].str.strip(), errors="coerce")
    valid = dates.notna() & np.isfinite(demand) & demand.ge(0)
    valid &= dates.dt.minute.isin([0, 30]) & dates.dt.second.eq(0)
    cleaned = pd.DataFrame({"time": dates[valid], "demand": demand[valid]})
    duplicate_rows = int(cleaned.time.duplicated().sum())
    values_per_time = cleaned.groupby("time").demand.nunique()
    conflicts = values_per_time.index[values_per_time > 1]
    cleaned = cleaned[~cleaned.time.isin(conflicts)].drop_duplicates("time")
    positions = times.get_indexer(cleaned.time)
    inside = positions >= 0
    values = np.full(len(times), np.nan, dtype=np.float32)
    values[positions[inside]] = cleaned.demand.to_numpy()[inside]
    return values, {"rows": len(group), "invalid_rows": int((~valid).sum()),
                    "duplicate_rows": duplicate_rows, "conflicting_times": len(conflicts),
                    "outside_tariff_coverage": int((~inside).sum())}


def valid_starts(values, labels, partition_id):
    """Pair t with t+1 only when both measurements and split membership are valid."""
    return np.flatnonzero(np.isfinite(values[:-1]) & np.isfinite(values[1:]) &
                          (labels[:-1] == partition_id) & (labels[1:] == partition_id))


def prepare(raw_file, tariff_file, folder, chunk_rows=500_000):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("Choose an empty preparation folder; existing data will not be overwritten.")
    save_json(folder / "status.json", {"state": "reading_raw"})
    tariffs = pd.read_excel(tariff_file).sort_values("TariffDateTime").reset_index(drop=True)
    times = pd.DatetimeIndex(tariffs.TariffDateTime)
    if times.empty or times.hasnans or times.has_duplicates or times.tz is not None:
        raise ValueError("Require valid, unique native LCL tariff timestamps.")
    if not np.all(np.diff(times.asi8) == 30 * 60 * 1_000_000_000):
        raise ValueError("Tariff workbook must be continuous at half-hour resolution.")
    price_ids = tariffs.Tariff.map({"Low": 0, "Normal": 1, "High": 2})
    if price_ids.isna().any():
        raise ValueError("Unknown tariff; never fill a missing price with Normal.")
    price_ids = price_ids.to_numpy(dtype=np.uint8)
    labels, season_ids = seasonal_split(times, price_ids)
    pd.DataFrame({"timestamp": times, "partition": np.array(PARTITIONS)[labels],
                  "season": np.array(SEASONS)[season_ids], "price_pence_per_kwh": np.array(PRICES)[price_ids]}
                 ).to_csv(folder / "calendar_split.csv", index=False)
    households, panels = [], []
    totals = {key: 0 for key in ("rows", "invalid_rows", "duplicate_rows", "conflicting_times", "outside_tariff_coverage")}
    empty_households = 0
    for household_id, group in household_groups(raw_file, chunk_rows):
        values, counts = clean_household(group, times)
        for key, count in counts.items():
            totals[key] += count
        if not any(len(valid_starts(values, labels, partition_id)) for partition_id in range(3)):
            empty_households += 1
            continue
        households.append(household_id)
        panels.append(values)
        if len(households) % 100 == 0:
            print(f"Raw input: {totals['rows']:,} rows read; {len(households):,} households retained.", flush=True)
    if not panels:
        raise ValueError("No valid raw household transitions.")
    panel = np.stack(panels)
    del panels
    np.save(folder / "households.npy", np.array(households))
    np.save(folder / "times.npy", times.asi8)
    np.save(folder / "partition_ids.npy", labels)
    coverage = []
    pair_counts = {}
    demand_scale, demand_sum, demand_count = 0., 0., 0
    for partition_id, name in enumerate(PARTITIONS):
        counts = [len(valid_starts(values, labels, partition_id)) for values in panel]
        total_pairs = sum(counts)
        if not total_pairs:
            raise ValueError(f"No transitions in {name}.")
        pairs = np.lib.format.open_memmap(folder / f"{name}.npy", mode="w+", dtype=PAIR_TYPE, shape=(total_pairs,))
        offset = 0
        for household_index, (values, count) in enumerate(zip(panel, counts)):
            starts = valid_starts(values, labels, partition_id)
            rows = pairs[offset:offset + count]
            rows["current_kwh"] = values[starts]
            rows["next_kwh"] = values[starts + 1]
            rows["price_id"] = price_ids[starts + 1]  # Price applies to the target half-hour.
            rows["season_id"] = season_ids[starts + 1]
            rows["household_index"] = household_index
            rows["start"] = starts
            offset += count
        pairs.flush()
        if partition_id == 0:
            demand_scale = float(max(pairs["current_kwh"].max(), pairs["next_kwh"].max()))
            demand_sum = float(pairs["next_kwh"].sum(dtype=np.float64))
            demand_count = len(pairs)
        for season_id, season in enumerate(SEASONS):
            selected = pairs["season_id"] == season_id
            for price_id, price in enumerate(PRICES):
                pair_count = int(np.sum(selected & (pairs["price_id"] == price_id)))
                if pair_count == 0:
                    raise ValueError(f"Missing {season}/{price} exposure in {name}.")
                used_targets = np.unique(pairs["start"][selected & (pairs["price_id"] == price_id)] + 1)
                coverage.append({"partition": name, "season": season, "price_pence_per_kwh": price,
                                 "pairs": pair_count, "unique_target_timestamps": len(used_targets)})
        pair_counts[name] = total_pairs
        print(f"Saved {name}: {total_pairs:,} measured transitions.", flush=True)
        del pairs
    if not 0 < demand_scale < float("inf"):
        raise ValueError("Training demand maximum must be positive and finite.")
    pd.DataFrame(coverage).to_csv(folder / "season_price_coverage.csv", index=False)
    calendar_counts = pd.crosstab(np.array(SEASONS)[season_ids], np.array(PARTITIONS)[labels])
    calendar_counts.to_csv(folder / "season_split_counts.csv")
    manifest = {"raw_file": str(Path(raw_file).resolve()), "raw_sha256": file_hash(raw_file),
                "tariff_workbook": str(Path(tariff_file).resolve()), "tariff_sha256": file_hash(tariff_file),
                "script_sha256": file_hash(__file__), "households": len(households),
                "households_without_any_pairs": empty_households, "cleaning": totals,
                "pair_counts": pair_counts, "demand_scale_kwh": demand_scale,
                "training_target_mean_kwh": demand_sum / demand_count,
                "split": "80/10/10 timestamps in each season, rounded to nearest interval; shared monthly blocks",
                "boundary_rule": "both observations must belong to same partition",
                "inputs": ["current half-hour consumption kWh", "next half-hour recorded price pence/kWh"],
                "target": "next half-hour measured consumption kWh",
                "prices_pence_per_kwh": list(PRICES), "seasons": list(SEASONS),
                "scope": "all valid ToU households and readings inside official tariff coverage; no cap or imputation",
                "evaluation_scope": "within-year interpolation; households may occur in every partition"}
    manifest["output_hashes"] = {path.name: file_hash(path) for path in folder.iterdir() if path.suffix in (".npy", ".csv")}
    save_json(folder / "manifest.json", manifest)
    save_json(folder / "status.json", {"state": "complete"})
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=ROOT / "raw_data/lcl_tou_sorted.csv")
    parser.add_argument("--tariff-file", type=Path, default=ROOT / "raw_data/Tariffs.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "results/simple_response_final/data")
    args = parser.parse_args()
    prepare(args.raw_file, args.tariff_file, args.output)


if __name__ == "__main__":
    main()

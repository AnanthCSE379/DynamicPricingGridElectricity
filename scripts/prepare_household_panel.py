"""Rebuild all raw household histories and audit coverage for a five-hour GRU."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.split_simple_response import (
    ROOT, PARTITIONS, SEASONS, PRICES, clean_household, file_hash,
    household_groups, save_json, seasonal_split,
)


def prepare(raw_file, tariff_file, folder):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("Choose an empty folder; previous runs are preserved.")
    save_json(folder / "status.json", {"state": "reading_raw"})
    tariffs = pd.read_excel(tariff_file).sort_values("TariffDateTime")
    times = pd.DatetimeIndex(tariffs.TariffDateTime)
    if times.empty or times.hasnans or times.has_duplicates or times.tz is not None:
        raise ValueError("Require valid unique native tariff timestamps.")
    if not np.all(np.diff(times.asi8) == 30 * 60 * 1_000_000_000):
        raise ValueError("Tariff timestamps must be consecutive half-hours.")
    mapped_prices = tariffs.Tariff.map({"Low": 0, "Normal": 1, "High": 2})
    if mapped_prices.isna().any():
        raise ValueError("Missing/unknown tariff: no price filling is allowed.")
    price_ids = mapped_prices.to_numpy(dtype=np.uint8)
    partitions, season_ids = seasonal_split(times, price_ids)
    households, readings, household_coverage = [], [], []
    totals = {key: 0 for key in ("rows", "invalid_rows", "duplicate_rows", "conflicting_times", "outside_tariff_coverage")}
    no_readings = 0
    for household, group in household_groups(raw_file, 500_000):
        values, counts = clean_household(group, times)
        for key, count in counts.items():
            totals[key] += count
        if not np.isfinite(values).any():
            no_readings += 1
            continue
        households.append(household)
        readings.append(values)
        household_coverage.append({"household": household, **{
            f"{name}_coverage": float(np.isfinite(values[partitions == partition_id]).mean())
            for partition_id, name in enumerate(PARTITIONS)}})
        if len(households) % 100 == 0:
            print(f"Read {totals['rows']:,} raw rows; retained {len(households):,} households.", flush=True)
    panel = np.stack(readings)  # Rows are fixed household IDs; columns are native timestamps.
    known = np.isfinite(panel)
    observed_count = known.sum(axis=0)  # Never silently treat missing readings as zero.
    training_complete = known[:, partitions == 0].all(axis=1)
    np.save(folder / "demand.npy", panel)
    np.save(folder / "households.npy", np.array(households))
    np.save(folder / "times.npy", times.asi8)
    np.save(folder / "prices.npy", price_ids)
    np.save(folder / "partition_ids.npy", partitions)
    np.save(folder / "season_ids.npy", season_ids)
    pd.DataFrame(household_coverage).to_csv(folder / "household_coverage.csv", index=False)
    pd.DataFrame({"timestamp": times, "partition": np.array(PARTITIONS)[partitions],
                  "season": np.array(SEASONS)[season_ids],
                  "price_pence_per_kwh": np.array(PRICES)[price_ids],
                  "observed_households": observed_count,
                  "all_households_observed": observed_count == len(households)}
                 ).to_csv(folder / "calendar_and_coverage.csv", index=False)
    manifest = {"raw_file": str(Path(raw_file).resolve()), "raw_sha256": file_hash(raw_file),
                "tariff_workbook": str(Path(tariff_file).resolve()), "tariff_sha256": file_hash(tariff_file),
                "preparation_script_sha256": file_hash(__file__),
                "cleaning_and_split_script_sha256": file_hash(ROOT / "scripts/split_simple_response.py"),
                "households": len(households), "households_without_any_reading": no_readings,
                "cleaning": totals, "missing_readings": int((~known).sum()),
                "observed_households_min": int(observed_count.min()),
                "observed_households_median": float(np.median(observed_count)),
                "observed_households_max": int(observed_count.max()),
                "complete_all_household_halfhours": int((observed_count == len(households)).sum()),
                "households_complete_on_training_timestamps": int(training_complete.sum()),
                "missing_data_rule": "preserve NaN; do not impute or sum incomplete populations",
                "history_steps": 10, "interval_minutes": 30,
                "split": "shared season-stratified 80/10/10 calendar; entire 10-history-plus-target window must stay in one split"}
    manifest["output_hashes"] = {path.name: file_hash(path) for path in folder.iterdir() if path.suffix in (".npy", ".csv")}
    save_json(folder / "manifest.json", manifest)
    save_json(folder / "status.json", {"state": "complete"})
    print(manifest, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-file", type=Path, default=ROOT / "raw_data/lcl_tou_sorted.csv")
    parser.add_argument("--tariff-file", type=Path, default=ROOT / "raw_data/Tariffs.xlsx")
    parser.add_argument("--output", type=Path, default=ROOT / "results/five_hour_response/data")
    args = parser.parse_args()
    prepare(args.raw_file, args.tariff_file, args.output)

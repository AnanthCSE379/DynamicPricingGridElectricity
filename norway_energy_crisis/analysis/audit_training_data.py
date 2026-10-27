"""Audit additional household data and the response model's sampled edge cases."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json


def transitions(values, cap, run=1):
    """Count measured low-history runs followed by consumption above the cap."""
    low = values <= cap
    prefix = np.concatenate(
        (np.zeros((len(values), 1), dtype=np.int64), low.cumsum(axis=1)), axis=1
    )
    # Both the 24-hour history and complete 24-hour target stay inside training.
    origin = np.arange(24, values.shape[1] - 23)
    consecutive = prefix[:, origin] - prefix[:, origin - run] == run
    events = consecutive & (values[:, origin] > cap)
    if run == 1:
        direct = (values[:, origin - 1] <= cap) & (values[:, origin] > cap)
        require(np.array_equal(events, direct), "Transition alignment mismatch")
    return dict(
        events=int(events.sum()), households=int(np.count_nonzero(events.any(axis=1)))
    )


def group_summary(values, hours):
    """Describe marginal loads and natural transitions in training only."""
    require(values.shape[1] == hours, "Unexpected training shape.")
    previous = values[:, 23 : hours - 24]
    following = values[:, 24 : hours - 23]
    zero = previous == 0
    recovery = zero & (following > 0)
    by_household = (values == 0).sum(axis=1)
    zero_count = int(by_household.sum())
    return dict(
        households=len(values),
        training_readings=int(values.size),
        zeros=zero_count,
        zero_households=int(np.count_nonzero(by_household)),
        zero_top_five_household_share=(
            float(np.sort(by_household)[-5:].sum() / zero_count) if zero_count else None
        ),
        at_most_0_05=int((values <= 0.05).sum()),
        at_most_0_1=int((values <= 0.1).sum()),
        eligible_zero_preceding_origins=int(zero.sum()),
        zero_to_positive_events=int(recovery.sum()),
        zero_to_positive_households=int(np.count_nonzero(recovery.any(axis=1))),
        next_load_after_zero_mean_kwh=(
            float(following[zero].mean()) if zero.any() else None
        ),
        positive_next_load_after_zero_mean_kwh=(
            float(following[recovery].mean()) if recovery.any() else None
        ),
        natural_low_transitions={
            f"last_{run}h_at_most_{cap}_then_above": transitions(values, cap, run)
            for cap in (0.05, 0.1, 0.75, 1.25)
            for run in (1, 6)
        },
    )


def audit(output):
    output = Path(output)
    require(not output.exists(), f"Output already exists: {output}")
    cleaned = ROOT / "cleaned"
    manifest = json.loads((cleaned / "manifest.json").read_text())
    for name in ("demand.npy", "times.npy", "household_ids.npy", "cohort_mask.npy"):
        require(digest(cleaned / name) == manifest["output_hashes"][name], name)
    demand = np.load(cleaned / "demand.npy", mmap_mode="r")
    times = np.load(cleaned / "times.npy").astype("datetime64[ns]")
    mask = np.load(cleaned / "cohort_mask.npy")
    ids = np.load(cleaned / "household_ids.npy")
    area = np.load(cleaned / "area_index.npy")
    households = pd.read_csv(ROOT / "households.csv")
    answers = pd.read_csv(ROOT / "answers.csv")
    survey = pd.read_csv(ROOT / "survey.csv")
    linked = households.merge(answers[["ID", "Q32"]], on="ID", validate="one_to_one")
    metered = linked.set_index("ID").loc[ids]
    require(np.array_equal(metered.Q32.eq(1).to_numpy(), mask), "Contract mismatch")
    train_mask = times < np.datetime64("2022-01-01")
    train = np.asarray(demand[:, train_mask])
    hours = train.shape[1]
    groups = {
        name: group_summary(train[chosen], hours)
        for name, chosen in {
            "spot": mask,
            "additional": ~mask,
            "all": np.ones(len(mask), dtype=bool),
        }.items()
    }
    for code, frame in metered.groupby("Q32", dropna=False):
        selected = metered.Q32.eq(code).to_numpy()
        groups[f"contract_{int(code)}"] = group_summary(train[selected], hours)
    definitions = {
        str(int(row.Option_number)): str(row.Option_translation)
        for row in survey[survey.Question_ID.eq("Q32")].itertuples()
        if pd.notna(row.Option_number)
    }
    origins = np.load(ROOT / "prepared" / "train_origins.npy")
    require(
        np.array_equal(origins, np.arange(24, hours - 23)),
        "Sampling audit assumes consecutive eligible training origins",
    )
    spot = train[mask]
    run = ROOT / "runs/comparison_v1/transformer_24h_seed42_price"
    config = json.loads((run / "config.json").read_text())
    history = json.loads((run / "history.json").read_text())
    generator = torch.Generator().manual_seed(config["seed"])
    sampled = []
    epochs = []
    for row in history:
        epoch_keys = []
        for _ in range(config["batches"]):
            selected_hh = torch.randint(
                len(spot), (config["batch_size"],), generator=generator
            ).numpy()
            selected_origins = origins[
                torch.randint(
                    len(origins), (config["batch_size"],), generator=generator
                ).numpy()
            ]
            epoch_keys.append(selected_hh * len(origins) + selected_origins - 24)
        keys = np.concatenate(epoch_keys)
        h = keys // len(origins)
        t = keys % len(origins) + 24
        previous = spot[h, t - 1]
        epochs.append(
            dict(
                epoch=row["epoch"],
                horizon=row["horizon"],
                examples=len(keys),
                zero_preceding_examples=int((previous == 0).sum()),
                zero_to_positive_examples=int(
                    ((previous == 0) & (spot[h, t] > 0)).sum()
                ),
                at_most_0_1_preceding_examples=int((previous <= 0.1).sum()),
            )
        )
        sampled.append(keys)
    keys = np.concatenate(sampled)
    unique = np.unique(keys)
    pool = len(spot) * len(origins)
    sampling = dict(
        seed=config["seed"],
        epochs=len(history),
        batch_size=config["batch_size"],
        batches_per_epoch=config["batches"],
        possible_household_origins=pool,
        sampled_examples=len(keys),
        unique_sampled_origins=len(unique),
        unique_origin_coverage=float(len(unique) / pool),
        supervised_target_exposures=sum(
            row["examples"] * row["horizon"] for row in epochs
        ),
        zero_preceding_examples=sum(row["zero_preceding_examples"] for row in epochs),
        zero_to_positive_examples=sum(
            row["zero_to_positive_examples"] for row in epochs
        ),
        unique_zero_to_positive_origins=int(
            (
                (spot[unique // len(origins), unique % len(origins) + 23] == 0)
                & (spot[unique // len(origins), unique % len(origins) + 24] > 0)
            ).sum()
        ),
        epochs_detail=epochs,
        scope=(
            "Exact original sampler replay for seed42; counts initial histories only. "
            "Recursive targets provide additional overlapping label exposures."
        ),
    )
    best_epoch = json.loads((run / "validation.json").read_text())["best_epoch"]
    best_keys = np.concatenate(sampled[:best_epoch])
    best_unique = np.unique(best_keys)
    best_h = best_unique // len(origins)
    best_t = best_unique % len(origins) + 24
    sampling["best_checkpoint"] = dict(
        epoch=best_epoch,
        sampled_examples=len(best_keys),
        unique_sampled_origins=len(best_unique),
        unique_origin_coverage=float(len(best_unique) / pool),
        zero_to_positive_examples=sum(
            row["zero_to_positive_examples"] for row in epochs[:best_epoch]
        ),
        unique_zero_to_positive_origins=int(
            ((spot[best_h, best_t - 1] == 0) & (spot[best_h, best_t] > 0)).sum()
        ),
    )
    auxiliary = {}
    for name in ("prices.csv", "demand_areas.csv"):
        frame = pd.read_csv(ROOT / name)
        auxiliary[name] = dict(
            rows=len(frame),
            start=frame.Date.min(),
            end=frame.Date.max(),
            areas=sorted(frame.Price_area.unique().tolist()),
        )
    result = dict(
        state="complete",
        training_period=[str(times[train_mask][0]), str(times[train_mask][-1])],
        training_hours=hours,
        household_period=[str(times[0]), str(times[-1])],
        household_hours=len(times),
        contract_definitions=definitions,
        contract_counts={
            str(int(k)): int(v) for k, v in metered.Q32.value_counts().items()
        },
        extra_regions={
            manifest["areas"][int(k)]: int(v)
            for k, v in zip(*np.unique(area[~mask], return_counts=True))
        },
        survey_households=len(linked),
        survey_only_households=int(linked.Demand_data.eq("No").sum()),
        groups=groups,
        sampling=sampling,
        auxiliary=auxiliary,
        cleaned_manifest_sha256=digest(cleaned / "manifest.json"),
        source_sha256=digest(__file__),
        notes=[
            "No extra hourly spot-contract demand is present in these downloads.",
            "Additional households share the same two regional price time series.",
            "Paid hourly tariffs are unverified for the other contract types.",
            "Aggregate demand is not extra labeled household-level demand.",
            "Transitions describe measured consumption, not confirmed curtailment.",
        ],
    )
    output.mkdir(parents=True)
    save_json(output / "audit.json", result)
    print(
        json.dumps(
            {
                "output": str(output),
                "groups": {
                    k: v for k, v in groups.items() if not k.startswith("contract")
                },
                "sampling": {k: v for k, v in sampling.items() if k != "epochs_detail"},
                "contract_counts": result["contract_counts"],
                "auxiliary": auxiliary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/training_data_audit_v1"
    )
    audit(parser.parse_args().output)

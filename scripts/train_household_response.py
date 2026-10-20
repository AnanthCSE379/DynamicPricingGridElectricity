"""Train a fresh multivariate GRU: five-hour household matrix + next price."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from household_response.data import PRICES, HISTORY_STEPS, batch_inputs, fit_scales, make_features, window_targets
from household_response.model import HouseholdGRU
from scripts.split_simple_response import ROOT, SEASONS, file_hash, save_json


def error_summary(squared, absolute, counts):
    """Pool only observed targets; retain equal-season/tariff and natural errors."""
    def score(indices):
        count = int(counts[indices].sum())
        return {"count": count, "rmse_kwh": float(np.sqrt(squared[indices].sum() / count)),
                "mae_kwh": float(absolute[indices].sum() / count)}
    result = score(np.arange(12))
    result["balanced_season_price_mse_kwh2"] = float(np.mean(squared / counts))
    result["by_price"] = {str(float(price)): score(np.arange(price_id, 12, 3)) for price_id, price in enumerate(PRICES)}
    result["by_season"] = {name: score(np.arange(season_id * 3, season_id * 3 + 3)) for season_id, name in enumerate(SEASONS)}
    return result


def evaluate(model, features, prices, target_values, observed, scales, targets, group_ids, folder=None):
    squared = np.zeros(12); absolute = np.zeros(12); counts = np.zeros(12, dtype=np.int64)
    persistence_squared = np.zeros(12); persistence_absolute = np.zeros(12)
    saved = []
    model.eval()
    with torch.no_grad():
        for selected in targets.split(128):
            history, next_price = batch_inputs(features, prices, selected)
            predicted = model(history, next_price) * scales[None, :]
            actual = target_values[selected]
            known = observed[selected]
            error = predicted - actual
            # Persistence needs a measured current reading as well as measured target.
            current_known = observed[selected - 1]
            paired = known & current_known
            current = target_values[selected - 1]
            persistence_error = current - actual
            for group in range(12):
                rows = group_ids[selected] == group
                group_known = known[rows]
                group_error = error[rows]
                squared[group] += float(group_error.square()[group_known].double().sum())
                absolute[group] += float(group_error.abs()[group_known].double().sum())
                counts[group] += int(group_known.sum())
                valid_persistence = paired[rows]
                persistence_squared[group] += float(persistence_error[rows].square()[valid_persistence].double().sum())
                persistence_absolute[group] += float(persistence_error[rows].abs()[valid_persistence].double().sum())
            if folder is not None:
                saved.append(predicted.numpy())
    # Baseline and GRU must be scored on exactly the same observed pairs.
    paired_counts = np.zeros(12, dtype=np.int64)
    paired_squared = np.zeros(12); paired_absolute = np.zeros(12)
    with torch.no_grad():
        for selected in targets.split(128):
            predicted = model(*batch_inputs(features, prices, selected)) * scales[None, :]
            paired = observed[selected] & observed[selected - 1]
            error = predicted - target_values[selected]
            for group in range(12):
                rows = group_ids[selected] == group
                valid = paired[rows]
                paired_counts[group] += int(valid.sum())
                paired_squared[group] += float(error[rows].square()[valid].double().sum())
                paired_absolute[group] += float(error[rows].abs()[valid].double().sum())
    if (counts == 0).any() or (paired_counts == 0).any():
        raise ValueError("A season/tariff group has no valid targets after five-hour window purging.")
    if folder is not None:
        np.save(Path(folder) / "test_predictions.npy", np.concatenate(saved))
        np.save(Path(folder) / "test_target_indices.npy", targets.numpy())
    return {"gru_all_observed_targets": error_summary(squared, absolute, counts),
            "gru_persistence_comparable": error_summary(paired_squared, paired_absolute, paired_counts),
            "persistence": error_summary(persistence_squared, persistence_absolute, paired_counts)}


def train(data_folder, folder, epochs=40, width=64):
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    if any(folder.iterdir()):
        raise FileExistsError("Choose an empty model folder; all weights start fresh.")
    torch.set_num_threads(4)
    torch.manual_seed(20261017)
    data_folder = Path(data_folder).resolve()
    manifest = json.loads((data_folder / "manifest.json").read_text())
    for name, expected in manifest["output_hashes"].items():
        if file_hash(data_folder / name) != expected:
            raise ValueError(f"Raw-panel fingerprint mismatch: {name}")
    demand = np.load(data_folder / "demand.npy")
    households = np.load(data_folder / "households.npy")
    partitions = np.load(data_folder / "partition_ids.npy")
    seasons = np.load(data_folder / "season_ids.npy")
    price_ids = np.load(data_folder / "prices.npy")
    times = np.load(data_folder / "times.npy")
    targets = window_targets(partitions, demand)
    for name, selected in targets.items():
        if not len(selected):
            raise ValueError(f"No five-hour windows in {name}.")
        if set(seasons[selected] * 3 + price_ids[selected]) != set(range(12)):
            raise ValueError(f"Five-hour purge removed seasonal tariff coverage from {name}.")
    means, maxima = fit_scales(demand, partitions == 0)
    features = make_features(demand, means, maxima)
    scales = torch.from_numpy(maxima.copy())
    prices = torch.from_numpy((PRICES[price_ids] - PRICES[0]) / (PRICES[-1] - PRICES[0]))
    observed = torch.from_numpy(np.isfinite(demand).T.copy())
    target_values = torch.from_numpy(np.nan_to_num(demand, nan=0.).T.copy())
    group_ids = torch.from_numpy((seasons.astype(np.int64) * 3 + price_ids).copy())
    targets = {key: torch.from_numpy(value) for key, value in targets.items()}
    # Balance observed target readings, not fabricated entries or household copies.
    counts = torch.zeros(12, dtype=torch.float64)
    for group in range(12):
        selected = targets["train"][group_ids[targets["train"]] == group]
        counts[group] = observed[selected].sum()
    group_weights = (counts.sum() / (12 * counts)).float()
    training_mean = float(np.nanmean(demand[:, partitions == 0], dtype=np.float64))
    loss_scale = training_mean ** 2
    model = HouseholdGRU(len(households), width)
    # Initialize each household's positive prediction near its training mean.
    starting_level = np.maximum(means / maxima, 1e-5)
    torch.nn.init.normal_(model.next_demand.weight, std=.002)
    with torch.no_grad():
        model.next_demand.bias.copy_(torch.from_numpy(np.log(np.expm1(starting_level))))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    settings = {"households": len(households), "width": width, "history_steps": HISTORY_STEPS,
                "seed": 20261017, "epochs_limit": epochs, "batch_size": 128, "patience": 7,
                "learning_rate": .001, "weight_decay": .0001,
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "window_counts": {key: len(value) for key, value in targets.items()},
                "inputs": "10 x household demand matrix, observed flags, next shared tariff",
                "output": "one next-half-hour kWh prediction for each household in saved ID order",
                "missing_inputs": "training-only household mean, plus explicit observed indicator",
                "missing_targets": "excluded from loss and metrics; never imputed",
                "normalization": "household demand / household training maximum; fixed LCL price range",
                "initialization": "fresh random GRU; output biases use training-only means; no previous weights",
                "group_weights": group_weights.tolist(), "training_group_observations": counts.tolist(),
                "data_folder": str(data_folder), "data_manifest_sha256": file_hash(data_folder / "manifest.json"),
                "training_script_sha256": file_hash(__file__),
                "model_script_sha256": file_hash(ROOT / "household_response/model.py")}
    save_json(folder / "settings.json", settings)
    np.save(folder / "household_means.npy", means); np.save(folder / "household_maxima.npy", maxima)
    np.save(folder / "households.npy", households)
    used = {}
    coverage = []
    for name, selected in targets.items():
        indices = selected.numpy()
        used[name] = set((indices[:, None] + np.arange(-10, 1)).ravel())
        for season_id, season in enumerate(SEASONS):
            for price_id, price in enumerate(PRICES):
                chosen = indices[(seasons[indices] == season_id) & (price_ids[indices] == price_id)]
                coverage.append({"partition": name, "season": season, "price_pence_per_kwh": float(price),
                                 "windows": len(chosen), "unique_target_dates": len(np.unique(times[chosen].astype('datetime64[ns]').astype('datetime64[D]')))})
    for first, second in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if not used[first].isdisjoint(used[second]):
            raise ValueError("History/target overlap across partitions.")
    pd.DataFrame(coverage).to_csv(folder / "window_coverage.csv", index=False)
    log = []; best_score = float("inf"); best_epoch = 0
    print(json.dumps({"parameters": settings["parameters"], "window_counts": settings["window_counts"]}), flush=True)
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        save_json(folder / "status.json", {"state": "training", "epoch": epoch, "best_epoch": best_epoch})
        model.train(); order = targets["train"][torch.randperm(len(targets["train"]))]
        total_loss = 0.; total_known = 0
        for selected in order.split(128):
            predicted = model(*batch_inputs(features, prices, selected)) * scales[None, :]
            known = observed[selected]
            error = predicted - target_values[selected]
            weighted = error.square() * group_weights[group_ids[selected], None]
            loss = weighted[known].mean() / loss_scale
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            total_loss += float(loss.detach()) * int(known.sum()); total_known += int(known.sum())
        validation = evaluate(model, features, prices, target_values, observed, scales, targets["validation"], group_ids)
        score = validation["gru_all_observed_targets"]["balanced_season_price_mse_kwh2"]
        if score < best_score:
            best_score = score; best_epoch = epoch
            torch.save({"weights": model.state_dict(), "width": width, "households": len(households),
                        "household_means": torch.from_numpy(means), "household_maxima": scales,
                        "household_ids": households.tolist(), "history_steps": HISTORY_STEPS,
                        "best_epoch": best_epoch, "data_manifest_sha256": settings["data_manifest_sha256"]}, folder / "gru.pt")
        log.append({"epoch": epoch, "training_loss": total_loss / total_known,
                    "validation_rmse_kwh": validation["gru_all_observed_targets"]["rmse_kwh"],
                    "validation_balanced_mse_kwh2": score, "seconds": time.monotonic() - started})
        pd.DataFrame(log).to_csv(folder / "training.csv", index=False)
        print(f"Epoch {epoch:02d}: validation RMSE {log[-1]['validation_rmse_kwh']:.6f} kWh; "
              f"balanced MSE {score:.6f}; {log[-1]['seconds']:.1f}s; best epoch {best_epoch}", flush=True)
        if epoch - best_epoch >= 7:
            break
    bundle = torch.load(folder / "gru.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(bundle["weights"])
    save_json(folder / "status.json", {"state": "evaluating_test", "best_epoch": best_epoch})
    metrics = {"best_epoch": best_epoch,
               "validation": evaluate(model, features, prices, target_values, observed, scales, targets["validation"], group_ids),
               "test": evaluate(model, features, prices, target_values, observed, scales, targets["test"], group_ids, folder)}
    save_json(folder / "metrics.json", metrics)
    example_target = int(targets["test"][0])
    history = demand[:, example_target - HISTORY_STEPS:example_target].T
    save_json(folder / "example_input.json", {"household_ids": households.tolist(),
              "demand_history_kwh": [[None if not np.isfinite(value) else float(value) for value in row] for row in history],
              "history_timestamps": pd.DatetimeIndex(times[example_target-10:example_target]).astype(str).tolist(),
              "next_timestamp": str(pd.Timestamp(times[example_target])), "next_price_pence_per_kwh": float(PRICES[price_ids[example_target]])})
    with torch.no_grad():
        prediction = model(*batch_inputs(features, prices, torch.tensor([example_target])))[0] * scales
    pd.DataFrame({"household": households, "current_demand_kwh": demand[:, example_target - 1],
                  "predicted_next_demand_kwh": prediction.numpy(), "actual_next_demand_kwh": demand[:, example_target]}
                 ).to_csv(folder / "example_predictions.csv", index=False)
    save_json(folder / "status.json", {"state": "complete", "best_epoch": best_epoch})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "results/five_hour_response/data")
    parser.add_argument("--output", type=Path, default=ROOT / "results/five_hour_response/model")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    if min(args.epochs, args.width) < 1:
        parser.error("epochs and width must be positive")
    train(args.data, args.output, args.epochs, args.width)

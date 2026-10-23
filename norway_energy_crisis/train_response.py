"""Train shared RNN/GRU/LSTM/Transformer response models with recursive targets."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.preprocess import PreparedData, Preprocessor, calendar

ARCHITECTURES = ("RNN", "GRU", "LSTM", "Transformer")
SEEDS = (42, 7, 2026)


class ResponseModel(nn.Module):
    """Predict normalized next-hour household demand with shared model weights.

    A two-layer sequence encoder feeds a common nonlinear prediction head.
    The head corrects persistence in inverse-softplus space so forecasts remain
    nonnegative without imposing a direction of response to price.
    """

    def __init__(self, architecture, lookback, static_size, use_price=True, width=64):
        super().__init__()
        require(
            architecture in ARCHITECTURES and lookback in (5, 24),
            "Unknown architecture/history.",
        )
        self.config = dict(
            architecture=architecture,
            lookback=lookback,
            static_size=static_size,
            use_price=use_price,
            width=width,
        )
        self.use_price = use_price
        if architecture == "Transformer":
            self.projection = nn.Linear(9, width)
            layer = nn.TransformerEncoderLayer(
                width, 4, 128, 0.1, activation="gelu", batch_first=True
            )
            self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            positions = torch.arange(24, dtype=torch.float32)[:, None]
            frequencies = torch.exp(
                torch.arange(0, width, 2) * (-math.log(10000.0) / width)
            )
            encoding = torch.zeros(24, width)
            encoding[:, 0::2], encoding[:, 1::2] = torch.sin(
                positions * frequencies
            ), torch.cos(positions * frequencies)
            self.register_buffer("position", encoding, persistent=False)
        else:
            self.encoder = getattr(nn, architecture)(
                9, width, num_layers=2, dropout=0.1, batch_first=True
            )
        self.head = nn.Sequential(
            nn.Linear(width + static_size + 7, width), nn.SiLU(), nn.Linear(width, 1)
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, static, next_price, target_calendar):
        """Return one scaled demand per example; raw kWh requires its saved scale."""
        encoded = self.encode_history(history)
        return self.predict_encoded(
            encoded, static, next_price, target_calendar, history[:, -1, 0]
        )

    def encode_history(self, history):
        """Encode history once, masking historical prices for the price-free control."""
        if not self.use_price:
            history = history.clone()
            history[:, :, 1] = 0
        if self.config["architecture"] == "Transformer":
            return self.encoder(
                self.projection(history) + self.position[None, : history.shape[1]]
            )[:, -1]
        return self.encoder(history)[0][:, -1]

    def predict_encoded(
        self, encoded, static, next_price, target_calendar, previous_demand
    ):
        """Apply the shared head to histories, optionally across candidate prices.

        Leading dimensions may represent households or actions-by-households.
        Keeping this calculation here makes normal and candidate-price inference
        use the same price masking, prediction head and persistence correction.
        """
        if not self.use_price:
            next_price = torch.zeros_like(next_price)
        correction = self.head(
            torch.cat((encoded, static, next_price[..., None], target_calendar), dim=-1)
        )[..., 0]
        previous_demand = previous_demand.clamp_min(1e-6)
        # Inverse softplus anchors a zero head correction at the preceding load.
        persistence_logit = previous_demand + torch.log(-torch.expm1(-previous_demand))
        return torch.nn.functional.softplus(persistence_logit + correction)


def rollout(model, batch, horizon=24):
    """Forecast recursively in kWh without reading future demand or weather.

    Append each prediction to history, supply the recorded candidate prices and
    advance known calendar features. Hold temperature at its forecast-origin value.
    """
    history = batch["history"]
    temperature = history[:, -1, 2]
    predictions = []
    for step in range(horizon):
        predicted = model(
            history,
            batch["static"],
            batch["prices"][:, step],
            batch["calendar"][:, step],
        )
        predictions.append(predicted * batch["scales"])
        next_row = torch.cat(
            (
                predicted[:, None],
                batch["prices"][:, step, None],
                temperature[:, None],
                batch["calendar"][:, step],
            ),
            dim=1,
        )
        history = torch.cat((history[:, 1:], next_row[:, None]), dim=1)
    return torch.stack(predictions, dim=1)


def atomic_torch_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def load_model(path):
    """Restore a response model and its preprocessing bundle on CPU in eval mode."""
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    require(
        bundle.get("format") == "norway_response_v1",
        "Not a Norwegian response checkpoint.",
    )
    model = ResponseModel(**bundle["model_config"])
    model.load_state_dict(bundle["weights"])
    return model.eval(), bundle


class Predictor:
    """Run checkpoint inference from kWh, NOK/kWh and Celsius history arrays.

    Rows represent households in the requested ID order; columns represent the
    checkpoint's history length. Returned next-hour predictions are in kWh.
    """

    def __init__(self, checkpoint):
        self.model, self.bundle = load_model(checkpoint)
        self.preprocessor = Preprocessor(self.bundle["preprocessing"])
        self.lookup = {
            value: index for index, value in enumerate(self.preprocessor.ids)
        }

    def predict(
        self, demand, prices, temperature, times, household_ids, candidate_price
    ):
        indices = np.array([self.lookup[value] for value in household_ids])
        require(
            np.asarray(demand).shape == (len(indices), self.model.config["lookback"]),
            "History length does not match checkpoint.",
        )
        history = self.preprocessor.history_features(
            demand, prices, temperature, times, indices
        )
        timestamps = np.asarray(times, dtype="datetime64[ns]")
        if timestamps.ndim == 1:
            timestamps = np.broadcast_to(timestamps, np.asarray(demand).shape)
        require(
            timestamps.shape == np.asarray(demand).shape
            and np.all(np.diff(timestamps.astype("int64"), axis=1) == 3600 * 10**9),
            "History times must be consecutive local dataset hours.",
        )
        next_time = timestamps[:, -1] + np.timedelta64(1, "h")
        price = np.broadcast_to(
            self.preprocessor.price(candidate_price), (len(indices),)
        ).copy()
        with torch.inference_mode():
            result = self.model(
                torch.from_numpy(history),
                torch.from_numpy(self.preprocessor.static[indices]),
                torch.from_numpy(price),
                torch.from_numpy(calendar(next_time)),
            )
        return result.numpy() * self.preprocessor.scales[indices]


def summarize(predicted, actual, data, origins):
    """Score identical household/hour forecasts in kWh and regional aggregate kWh."""
    require(
        predicted.shape == actual.shape
        and np.isfinite(predicted).all()
        and (predicted >= 0).all(),
        "Nonfinite/negative forecasts or shape mismatch.",
    )
    error = predicted.astype(np.float64) - actual

    def errors(values):
        return dict(
            rmse_kwh=float(np.sqrt(np.mean(values**2))),
            mae_kwh=float(np.mean(np.abs(values))),
        )

    metrics = dict(
        one_hour=errors(error[:, :, 0]),
        rollout_24h=errors(error),
        rmse_by_horizon_kwh=np.sqrt(np.mean(error**2, axis=(0, 1))).tolist(),
        mae_by_horizon_kwh=np.mean(np.abs(error), axis=(0, 1)).tolist(),
        days=len(origins),
        households=data.households,
        one_hour_scope=(
            "First forecast lead at the same fixed daily origins used for 24-hour"
            " evaluation."
        ),
    )
    daily = dict(
        squared=np.mean(error**2, axis=(1, 2)).tolist(),
        absolute=np.mean(np.abs(error), axis=(1, 2)).tolist(),
    )
    region_metrics = {}
    for area_index, area in enumerate(data.state["areas"]):
        chosen = data.area_index == area_index
        aggregate = error[:, chosen, :].sum(axis=1)
        region_metrics[area] = dict(
            rmse_aggregate_kwh=float(np.sqrt(np.mean(aggregate**2))),
            mae_aggregate_kwh=float(np.mean(np.abs(aggregate))),
        )
    metrics["regional_aggregate"] = region_metrics
    # Fixed training quartiles are descriptive bands, never fabricated tariff classes.
    raw_prices = data.prices[
        data.area_index[None, :, None],
        origins[:, None, None] + np.arange(24)[None, None, :],
    ]
    bands = np.searchsorted(data.state["price_band_edges"], raw_prices, side="right")
    metrics["price_bands"] = {
        str(band): dict(
            count=int(np.sum(bands == band)), **errors(error[bands == band])
        )
        for band in range(4)
        if np.any(bands == band)
    }
    return metrics, daily


def forecast(model, data, split, batch_size=128):
    """Enumerate every eligible daily origin and forecast all households recursively."""
    origins = data.origins[split]
    household_count = data.households
    flat_households = np.tile(np.arange(household_count), len(origins))
    flat_origins = np.repeat(origins, household_count)
    predicted, actual = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(flat_origins), batch_size):
            batch = data.batch(
                flat_households[start : start + batch_size],
                flat_origins[start : start + batch_size],
                model.config["lookback"],
            )
            predicted.append(rollout(model, batch).numpy())
            actual.append(batch["target"].numpy())
    return np.concatenate(predicted).reshape(
        len(origins), household_count, 24
    ), np.concatenate(actual).reshape(len(origins), household_count, 24)


def baseline_forecasts(data, split):
    """Return measured targets and persistence/previous-day baselines in kWh."""
    origins = data.origins[split]
    actual = np.stack([data.demand[:, origin : origin + 24] for origin in origins])
    persistence = np.stack(
        [np.repeat(data.demand[:, origin - 1, None], 24, axis=1) for origin in origins]
    )
    previous_day = np.stack(
        [data.demand[:, origin - 24 : origin] for origin in origins]
    )
    return actual, dict(persistence=persistence, previous_day=previous_day)


def checkpoint(model, data, configuration, epoch, score):
    """Bundle weights, model configuration, preprocessing and data provenance."""
    return dict(
        format="norway_response_v1",
        model_config=model.config,
        weights=model.state_dict(),
        preprocessing=data.state,
        run_config=configuration,
        epoch=epoch,
        validation_24h_rmse=score,
        prepared_manifest_sha256=digest(data.folder / "manifest.json"),
    )


def train_run(
    data,
    folder,
    architecture,
    lookback,
    seed,
    use_price,
    epochs=20,
    batches=250,
    batch_size=128,
):
    """Train one response configuration with a 1-, 6-, then 24-hour curriculum.

    Select the best full-horizon validation RMSE after epoch five. Save resumable
    state only after completed epochs, and reject changed data, code or settings.
    """
    folder = Path(folder)
    configuration = dict(
        architecture=architecture,
        lookback=lookback,
        seed=seed,
        use_price=use_price,
        epochs=epochs,
        batches=batches,
        batch_size=batch_size,
        prepared_manifest_sha256=digest(data.folder / "manifest.json"),
        training_source_sha256=digest(__file__),
        preprocessing_source_sha256=digest(Path(__file__).with_name("preprocess.py")),
    )
    folder.mkdir(parents=True, exist_ok=True)
    config_path = folder / "config.json"
    if config_path.exists():
        require(
            json.loads(config_path.read_text()) == configuration,
            f"Run configuration/source changed: {folder}; use a new output directory.",
        )
    else:
        save_json(config_path, configuration)
    if (folder / "validation.json").is_file():
        (folder / "last.pt").unlink(missing_ok=True)
        return json.loads((folder / "validation.json").read_text())
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    model = ResponseModel(architecture, lookback, data.static.shape[1], use_price)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    best_score, best_epoch, history, start_epoch = float("inf"), 0, [], 1
    resume = folder / "last.pt"
    if resume.is_file():
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["weights"])
        optimizer.load_state_dict(saved["optimizer"])
        best_score, best_epoch, history, start_epoch = (
            saved["best_score"],
            saved["best_epoch"],
            saved["history"],
            saved["epoch"] + 1,
        )
        generator.set_state(saved["sample_rng"])
        torch.set_rng_state(saved["torch_rng"])

    def save_resume(epoch):
        atomic_torch_save(
            dict(
                weights=model.state_dict(),
                optimizer=optimizer.state_dict(),
                best_score=best_score,
                best_epoch=best_epoch,
                history=history,
                epoch=epoch,
                sample_rng=generator.get_state(),
                torch_rng=torch.get_rng_state(),
            ),
            resume,
        )

    if not resume.exists():
        save_resume(0)
    # A fixed training-only divisor stabilizes gradients while preserving the
    # relative weighting of squared errors measured in actual household kWh.
    loss_scale = (
        float(
            np.mean(np.asarray(data.demand[:, data.partitions == 0], dtype=np.float64))
        )
        ** 2
    )
    require(loss_scale > 0, "Training demand must have positive mean.")
    # Save only completed epochs so an interrupted epoch can be replayed.
    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        horizon = 1 if epoch <= 2 else 6 if epoch <= 5 else 24
        model.train()
        losses = []
        for _ in range(batches):
            households = torch.randint(
                data.households, (batch_size,), generator=generator
            ).numpy()
            origins = data.origins["train"][
                torch.randint(
                    len(data.origins["train"]), (batch_size,), generator=generator
                ).numpy()
            ]
            batch = data.batch(households, origins, lookback, horizon)
            predicted = rollout(model, batch, horizon)
            loss = torch.mean((predicted - batch["target"]) ** 2) / loss_scale
            require(bool(torch.isfinite(loss)), f"Nonfinite training loss: {folder}")
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        predicted, actual = forecast(model, data, "validation", batch_size)
        metrics, _ = summarize(predicted, actual, data, data.origins["validation"])
        score = metrics["rollout_24h"]["rmse_kwh"]
        if epoch >= 6 and score < best_score:
            best_score, best_epoch = score, epoch
            atomic_torch_save(
                checkpoint(model, data, configuration, epoch, score),
                folder / "best.pt",
            )
        row = dict(
            epoch=epoch,
            horizon=horizon,
            loss=float(np.mean(losses)),
            validation_24h_rmse=score,
            best_epoch=best_epoch,
            seconds=time.monotonic() - started,
        )
        history.append(row)
        save_json(folder / "history.json", history)
        save_resume(epoch)
        print(
            f"{folder.name} epoch {epoch:02}/{epochs}, horizon={horizon},"
            f' validation24h={score:.6f}, best={best_epoch}, {row["seconds"]:.1f}s',
            flush=True,
        )
        if epoch >= 10 and epoch - best_epoch >= 5:
            break
    require(
        (folder / "best.pt").is_file(),
        "No full-horizon checkpoint; epochs must be at least six.",
    )
    model, bundle = load_model(folder / "best.pt")
    predicted, actual = forecast(model, data, "validation", batch_size)
    metrics, daily = summarize(predicted, actual, data, data.origins["validation"])
    result = dict(
        config=configuration, best_epoch=bundle["epoch"], metrics=metrics, daily=daily
    )
    save_json(folder / "validation.json", result)
    resume.unlink(missing_ok=True)
    return result


def paired_uncertainty(candidate, reference, seed=42):
    """Bootstrap paired calendar days, retaining all households and regions together."""
    differences = np.asarray(candidate) - np.asarray(reference)
    rng = np.random.default_rng(seed)
    samples = differences[
        rng.integers(len(differences), size=(5000, len(differences)))
    ].mean(axis=1)
    return dict(
        mean_daily_mse_difference=float(differences.mean()),
        ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
        days=len(differences),
    )


def _compare(data_folder, output, epochs=20, batches=250, batch_size=128):
    """Screen architectures, confirm finalists, then freeze selection before testing."""
    require(
        epochs >= 6 and batches > 0 and batch_size > 0,
        "Require epochs>=6 and positive batch counts/sizes.",
    )
    torch.set_num_threads(4)
    data = PreparedData(data_folder)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    signature = dict(
        prepared_manifest_sha256=digest(data.folder / "manifest.json"),
        epochs=epochs,
        batches=batches,
        batch_size=batch_size,
        seeds=list(SEEDS),
        history_lengths=[5, 24],
        architectures=list(ARCHITECTURES),
        training_source_sha256=digest(__file__),
    )
    if (output / "settings.json").exists():
        require(
            json.loads((output / "settings.json").read_text()) == signature,
            "Comparison inputs/settings changed; choose a new output.",
        )
    else:
        save_json(output / "settings.json", signature)
    if (output / "status.json").is_file() and json.loads(
        (output / "status.json").read_text()
    ).get("state") == "complete":
        print(f"Comparison already complete: {output}", flush=True)
        return
    entries = {}

    def run(architecture, lookback, seed, use_price):
        price_mode = "price" if use_price else "no_price"
        name = f"{architecture.lower()}_{lookback}h_seed{seed}_{price_mode}"
        save_json(output / "status.json", dict(state="training", run=name))
        result = train_run(
            data,
            output / name,
            architecture,
            lookback,
            seed,
            use_price,
            epochs,
            batches,
            batch_size,
        )
        entries[name] = result
        save_json(output / "validation_results.json", entries)
        return name, result

    screened = []
    for architecture in ARCHITECTURES:
        for lookback in [5, 24]:
            name, result = run(architecture, lookback, 42, True)
            screened.append(
                (result["metrics"]["rollout_24h"]["rmse_kwh"], architecture, lookback)
            )
    finalists = sorted(screened)[:2]
    confirmed = []
    for _, architecture, lookback in finalists:
        for seed in SEEDS[1:]:
            run(architecture, lookback, seed, True)
        names = [
            f"{architecture.lower()}_{lookback}h_seed{seed}_price" for seed in SEEDS
        ]
        scores = [entries[name]["metrics"]["rollout_24h"]["rmse_kwh"] for name in names]
        confirmed.append((float(np.mean(scores)), architecture, lookback, names))
    mean_score, architecture, lookback, winner_names = min(confirmed)
    control_names = []
    for seed in SEEDS:
        control_names.append(run(architecture, lookback, seed, False)[0])
    selection = dict(
        architecture=architecture,
        lookback=lookback,
        seeds=list(SEEDS),
        winner_runs=winner_names,
        control_runs=control_names,
        mean_validation_24h_rmse=mean_score,
        deployment_checkpoint=str((output / winner_names[0] / "best.pt").resolve()),
        rule=(
            "Lowest three-seed mean validation 24h RMSE among the two screening"
            " finalists; seed42 is deployment candidate, no best-seed cherry-picking."
        ),
        frozen_before_test=True,
    )
    if (output / "selection.json").exists():
        require(
            json.loads((output / "selection.json").read_text()) == selection,
            "Frozen model selection cannot change within a run.",
        )
    else:
        save_json(output / "selection.json", selection)
    # Only now access the held-out targets for evaluation.
    save_json(
        output / "status.json", dict(state="evaluating_test", selection=selection)
    )
    test_results = {}
    actual, baselines = baseline_forecasts(data, "test")
    for name, predicted in baselines.items():
        metrics, daily = summarize(predicted, actual, data, data.origins["test"])
        test_results[name] = dict(metrics=metrics, daily=daily)
    # Validation baselines accompany the selection evidence, but were not tuned.
    actual_val, baselines_val = baseline_forecasts(data, "validation")
    validation_baselines = {
        name: summarize(prediction, actual_val, data, data.origins["validation"])[0]
        for name, prediction in baselines_val.items()
    }
    for name in winner_names + control_names:
        path = output / name / "test.json"
        if path.exists():
            test_results[name] = json.loads(path.read_text())
            continue
        model, _ = load_model(output / name / "best.pt")
        predicted, truth = forecast(model, data, "test", batch_size)
        metrics, daily = summarize(predicted, truth, data, data.origins["test"])
        test_results[name] = dict(metrics=metrics, daily=daily)
        save_json(path, test_results[name])
    # Average seeds within each day before estimating uncertainty across days.
    winner_daily = np.mean(
        [test_results[run_name]["daily"]["squared"] for run_name in winner_names],
        axis=0,
    )
    control_daily = np.mean(
        [test_results[run_name]["daily"]["squared"] for run_name in control_names],
        axis=0,
    )
    uncertainty = {
        name: paired_uncertainty(winner_daily, test_results[name]["daily"]["squared"])
        for name in baselines
    }
    uncertainty["price_free"] = paired_uncertainty(winner_daily, control_daily)
    winner_rmse = float(
        np.mean(
            [
                test_results[run_name]["metrics"]["rollout_24h"]["rmse_kwh"]
                for run_name in winner_names
            ]
        )
    )
    control_rmse = float(
        np.mean(
            [
                test_results[run_name]["metrics"]["rollout_24h"]["rmse_kwh"]
                for run_name in control_names
            ]
        )
    )
    recommended = (
        all(value["ci95"][1] < 0 for value in uncertainty.values())
        and winner_rmse < control_rmse
        and all(
            winner_rmse < test_results[run_name]["metrics"]["rollout_24h"]["rmse_kwh"]
            for run_name in baselines
        )
    )
    result = dict(
        selection=selection,
        test=test_results,
        validation_baselines=validation_baselines,
        paired_day_uncertainty=uncertainty,
        recommend_simulator=recommended,
        test_price_rmse_mean=winner_rmse,
        test_price_rmse_std=float(
            np.std(
                [
                    test_results[run_name]["metrics"]["rollout_24h"]["rmse_kwh"]
                    for run_name in winner_names
                ]
            )
        ),
        test_price_free_rmse_mean=control_rmse,
        conclusion=(
            "Candidate passes forecast-comparison gate; causal response remains"
            " unvalidated."
            if recommended
            else "Do not promote to DQN: forecast/price-utility gate was not met."
        ),
        limitations=data.state["assumptions"]
        + [
            (
                "Historical temperature is a reanalysis proxy; future temperature is"
                " persisted."
            ),
            (
                "Same-household future-month evaluation, not unseen-household"
                " generalization."
            ),
            (
                "Observed-price forecasting and bootstrap intervals do not establish"
                " causal response to arbitrary DQN prices."
            ),
        ],
    )
    save_json(output / "comparison.json", result)
    report = [
        "# Norwegian demand-response comparison",
        "",
        (
            f"Selected {architecture}, {lookback}h history using validation only. All"
            " four architectures and both histories were screened; two finalists were"
            " confirmed across three seeds."
        ),
        "",
        "| Test method | 24h RMSE (kWh) | 24h MAE (kWh) |",
        "|---|---:|---:|",
    ]
    for name, run_result in test_results.items():
        metric = run_result["metrics"]["rollout_24h"]
        report.append(
            f'| {name} | {metric["rmse_kwh"]:.6f} | {metric["mae_kwh"]:.6f} |'
        )
    report += [
        "",
        "## Validation screening",
        "",
        "| Configuration | 24h RMSE (kWh) |",
        "|---|---:|",
    ]
    for _, model_type, history_length in sorted(screened):
        name = f"{model_type.lower()}_{history_length}h_seed42_price"
        report.append(
            f"| {model_type}, {history_length}h |"
            f' {entries[name]["metrics"]["rollout_24h"]["rmse_kwh"]:.6f} |'
        )
    report += (
        [
            "",
            "## Recommendation",
            "",
            result["conclusion"],
            "",
            (
                "Paired 95% intervals use calendar-day resampling, retaining all"
                " households and regions in each day. See comparison.json for horizon,"
                " regional and training-price-band errors, seed variability and"
                " confidence intervals."
            ),
            "",
            "## Assumptions",
            "",
        ]
        + ["- " + limitation for limitation in result["limitations"]]
        + [""]
    )
    (output / "REPORT.md").write_text("\n".join(report))
    save_json(
        output / "status.json", dict(state="complete", recommend_simulator=recommended)
    )
    print(f'Comparison complete: {output / "REPORT.md"}', flush=True)


def compare(data_folder, output, epochs=20, batches=250, batch_size=128):
    """Run or resume a response comparison while holding an exclusive output lock."""
    import fcntl

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another comparison is already active in {output}"
            ) from error
        try:
            return _compare(data_folder, output, epochs, batches, batch_size)
        except Exception as error:
            save_json(
                output / "status.json",
                dict(
                    state="failed",
                    error=str(error),
                    resume="Rerun the identical command after resolving the error.",
                ),
            )
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/comparison_v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batches", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--evaluate",
        type=Path,
        help="Evaluate an existing checkpoint; does not select/retrain models",
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.check_only:
        data = PreparedData(args.data)
        for architecture in ARCHITECTURES:
            model = ResponseModel(architecture, 24, data.static.shape[1]).eval()
            batch = data.batch(np.array([0]), np.array([data.origins["train"][0]]), 24)
            with torch.inference_mode():
                require(
                    bool(torch.isfinite(rollout(model, batch)).all()),
                    "Invalid rollout.",
                )
        print("Prepared hashes and all four architecture rollouts passed.")
        return
    if args.evaluate:
        data = PreparedData(args.data)
        model, bundle = load_model(args.evaluate)
        require(
            bundle["prepared_manifest_sha256"] == digest(data.folder / "manifest.json"),
            "Checkpoint/prepared data mismatch.",
        )
        predicted, actual = forecast(model, data, args.split, args.batch_size)
        print(
            json.dumps(
                summarize(predicted, actual, data, data.origins[args.split])[0],
                indent=2,
            )
        )
        return
    compare(args.data, args.output, args.epochs, args.batches, args.batch_size)


if __name__ == "__main__":
    main()

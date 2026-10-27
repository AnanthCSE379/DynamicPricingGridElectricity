"""Audit, train/resume and evaluate the bounded consumer-simulator refinement."""

import argparse
from copy import deepcopy
import importlib.metadata
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.train_response import load_model as load_incumbent
from norway_energy_crisis.response_v2.data import (
    ConsumerData,
    TransitionSampler,
    origins_for,
    support_report,
)
from norway_energy_crisis.response_v2.model import (
    ConsumerModel,
    atomic_save,
    bundle,
    load_model,
    rollout,
)
from norway_energy_crisis.response_v2.evaluate import (
    actuals,
    baseline_predictions,
    forecast,
    natural_low_errors,
    paired_blocks,
    price_curves,
    sequence_suite,
    verify_trained_raw_api,
    summarize,
    write_report,
)

CONFIGURATIONS = {
    "A": {"head": "persistence", "pretrain_all": False},
    "B": {"head": "direct", "pretrain_all": False},
    "C": {"head": "direct", "pretrain_all": True},
}
SEEDS = (42, 7, 2026)


def signature(data, budget_hours, slot_seconds, incumbent):
    return {
        "data": data.fingerprint,
        "sources": {
            p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))
        },
        "shared_sources": {
            name: digest(ROOT / name)
            for name in ("clean_data.py", "preprocess.py", "train_response.py")
        },
        "preprocessing_state": hashlib.sha256(
            json.dumps(data.state, sort_keys=True).encode()
        ).hexdigest(),
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "pandas", "matplotlib")
        },
        "python": sys.version,
        "incumbents": {
            str(seed): digest(incumbent / f"transformer_24h_seed{seed}_price/best.pt")
            for seed in (42, 7)
        },
        "budget_hours": budget_hours,
        "slot_seconds": slot_seconds,
        "configurations": CONFIGURATIONS,
        "seeds": list(SEEDS),
        "pretrain_fraction": 0.3,
        "sampling": "80% uniform, 5% zero-rise, 15% low-rise; p/q weights",
        "loss": ".5 first-hour MSE + .5 recursive MSE, kWh, p/q weights",
        "selection": "48h January RMSE; 2% 24h and 5% regional regression limits",
        "scope": "All configs share extended features; C tests extra-data pretraining.",
    }


class IncumbentAdapter(torch.nn.Module):
    def __init__(self, checkpoint, device):
        super().__init__()
        model, _ = load_incumbent(checkpoint)
        self.model = model.to(device).eval().requires_grad_(False)

    def forward(self, history, static, price, target_calendar, price_enabled=True):
        return self.model(history[:, :, :9], static, price, target_calendar)


def incumbents(folder, device):
    return [
        IncumbentAdapter(
            Path(folder) / f"transformer_24h_seed{seed}_price/best.pt",
            device,
        ).eval()
        for seed in (42, 7)
    ]


def validation_subset(data):
    days = data.origins["validation"]
    return days[np.linspace(0, len(days) - 1, min(6, len(days)), dtype=int)]


def score(models, data, origins, device, horizon=48):
    predictions = forecast(models, data, origins, horizon, device)
    return summarize(predictions, actuals(data, origins, horizon), data, origins)


def eligible(result, incumbent):
    return result["rmse_24h"] <= 1.02 * incumbent["rmse_24h"] and all(
        result["regions"][name]["aggregate_rmse_24h"]
        <= 1.05 * incumbent["regions"][name]["aggregate_rmse_24h"]
        for name in incumbent["regions"]
    )


def preflight(data, output, device, incumbent_folder, slot_seconds):
    """Real-data inference, 48h gradients, parity, checkpoints and throughput."""
    require(
        device == "mps" and torch.backends.mps.is_available(),
        "Explicit MPS is required for the real experiment; no CPU fallback.",
    )
    if (output / "preflight.json").exists():
        return json.loads((output / "preflight.json").read_text())
    started = time.monotonic()
    save_json(output / "support.json", support_report(data))
    torch.manual_seed(42)
    n = min(128, len(data.primary))
    hh = data.primary[:n]
    origin = data.origins["train"][0]
    inputs_cpu = data.inputs(hh, np.full(n, origin), 48)
    target = data.targets(hh, np.full(n, origin), 48).to(device)
    rows = {}
    initial_mean = float(
        np.mean(
            data.demand[data.primary[:, None], np.flatnonzero(data.partitions == 0)]
            / data.features.scales[data.primary, None]
        )
    )
    for name, config in CONFIGURATIONS.items():
        model = ConsumerModel(
            data.features.static.shape[1], config["head"], initial_mean=initial_mean
        ).eval()
        # Exercise encoder gradients as well as the initially zero output layer.
        torch.nn.init.normal_(model.head[-1].weight, std=0.01)
        cpu = rollout(model, inputs_cpu, 6)
        cpu.square().mean().backward()
        cpu_grads = [p.grad.detach().clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        accelerator = deepcopy(model).to(device).eval()
        inputs = inputs_cpu.to(device)
        mps = rollout(accelerator, inputs, 6)
        mps.square().mean().backward()
        np.testing.assert_allclose(
            mps.detach().cpu(), cpu.detach(), rtol=3e-3, atol=3e-4
        )
        for expected, parameter in zip(cpu_grads, accelerator.parameters()):
            np.testing.assert_allclose(
                parameter.grad.cpu(), expected, rtol=3e-2, atol=3e-3
            )
        optimizer = torch.optim.AdamW(accelerator.parameters(), lr=1e-3)
        accelerator.train()
        durations = []
        for step in range(3):
            step_started = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            predicted = rollout(accelerator, inputs, price_enabled=step > 0)
            loss = ((predicted - target) ** 2).mean()
            require(bool(torch.isfinite(loss)), "Nonfinite MPS loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                accelerator.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            torch.mps.synchronize()
            durations.append(time.monotonic() - step_started)
        accelerator.eval()
        full = forecast(accelerator, data, data.origins["validation"][:1], 48, device)
        require(np.isfinite(full).all(), "Full-cohort inference failed.")
        with torch.inference_mode():
            before = rollout(accelerator, inputs, 6).cpu()
        path = output / f"preflight_{name}.pt"
        atomic_save(bundle(accelerator, data), path)
        restored, _ = load_model(path, device)
        with torch.inference_mode():
            torch.testing.assert_close(rollout(restored, inputs, 6).cpu(), before)
        atomic_save(
            {
                "optimizer": optimizer.state_dict(),
                "mps_rng": torch.mps.get_rng_state(),
            },
            path,
        )
        active = torch.load(path, map_location="cpu", weights_only=False)
        restored.requires_grad_(True)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
        restored_optimizer.load_state_dict(active["optimizer"])
        torch.mps.set_rng_state(active["mps_rng"])
        restored_optimizer.zero_grad(set_to_none=True)
        resumed_loss = ((rollout(restored, inputs) - target) ** 2).mean()
        resumed_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            restored.parameters(), 5, error_if_nonfinite=True
        )
        restored_optimizer.step()
        require(bool(torch.isfinite(resumed_loss)), "MPS optimizer resume failed.")
        path.unlink()
        worst = max(durations)
        # Final 45% of fine tuning (70% of slot), less final validation reserve.
        require(
            worst * 100 < slot_seconds * 0.7 * 0.45 * 0.8,
            "Slot cannot support 100 full-horizon updates; revise before training.",
        )
        rows[name] = {
            "parameter_count": sum(p.numel() for p in accelerator.parameters()),
            "max_cpu_mps_kwh_difference": float(
                (mps.detach().cpu() - cpu.detach()).abs().max()
            ),
            "48h_update_seconds": durations,
            "last_gradient_norm": float(norm),
            "full_cohort_shape": list(full.shape),
            "checkpoint_roundtrip": True,
            "optimizer_and_mps_rng_recovery": True,
        }
        del accelerator, restored, optimizer, restored_optimizer
        torch.mps.empty_cache()
    frozen = incumbents(incumbent_folder, device)
    subset = validation_subset(data)
    incumbent_sub = score(frozen, data, subset, device)
    incumbent_all = score(frozen, data, data.origins["validation"], device)
    del frozen
    torch.mps.empty_cache()
    validation_seconds = time.monotonic() - started
    result = {
        "device": str(device),
        "models": rows,
        "initial_mean": initial_mean,
        "seconds": validation_seconds,
        "incumbent_subset": incumbent_sub,
        "incumbent_validation": incumbent_all,
        "validation_subset": subset.tolist(),
        "completed": True,
    }
    save_json(output / "preflight.json", result)
    return result


def train_run(
    data,
    folder,
    configuration,
    seed,
    use_price,
    device,
    slot_seconds,
    initial_mean,
    incumbent_subset,
    incumbent_all,
    deadline=None,
    batch_size=128,
    smoke_updates=None,
    heartbeat=None,
):
    """Resume after completed updates with matched pretraining/fine-tuning budgets."""
    run_started = time.monotonic()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    config = CONFIGURATIONS[configuration]
    settings = dict(
        configuration=configuration,
        seed=seed,
        use_price=use_price,
        slot_seconds=slot_seconds,
        batch_size=batch_size,
        model=config,
        data=data.fingerprint,
    )
    if (folder / "config.json").exists():
        require(
            json.loads((folder / "config.json").read_text()) == settings,
            "Fit settings changed.",
        )
    else:
        save_json(folder / "config.json", settings)
    if (folder / "result.json").exists():
        return json.loads((folder / "result.json").read_text())
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = ConsumerModel(
        data.features.static.shape[1], config["head"], use_price, initial_mean
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pretrain_hh = data.all_households if config["pretrain_all"] else data.primary
    samplers = [
        TransitionSampler(data, pretrain_hh, seed),
        TransitionSampler(data, data.primary, seed + 1),
    ]
    consumed, updates, full_updates = 0.0, 0, 0
    history = []
    best, best_fallback = float("inf"), float("inf")
    best_eligible = False
    next_validation, next_save = 600.0, 300.0
    resume = folder / "active.pt"
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=False)
        require(state["settings"] == settings, "Resume mismatch.")
        model.load_state_dict(state["weights"])
        optimizer.load_state_dict(state["optimizer"])
        consumed, updates, full_updates = (
            state["consumed"],
            state["updates"],
            state["full_updates"],
        )
        best, best_fallback, best_eligible = (
            state["best"],
            state["best_fallback"],
            state["best_eligible"],
        )
        history = state["history"]
        next_validation, next_save = state["next_validation"], state["next_save"]
        for sampler, rng in zip(samplers, state["sampler_rng"]):
            sampler.rng.bit_generator.state = rng
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(state["mps_rng"])
    train_mean = float(
        np.mean(
            data.demand[data.primary[:, None], np.flatnonzero(data.partitions == 0)]
        )
    )
    loss_scale = max(train_mean**2, 1e-6)
    subset = validation_subset(data)
    started = run_started
    previous_consumed = consumed
    last_update_seconds = 0.1
    validation_reserve = min(slot_seconds * 0.15, 180.0)

    def elapsed():
        return previous_consumed + time.monotonic() - started

    def save_state():
        atomic_save(
            {
                "settings": settings,
                "weights": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "consumed": elapsed(),
                "updates": updates,
                "full_updates": full_updates,
                "best": best,
                "best_fallback": best_fallback,
                "best_eligible": best_eligible,
                "history": history,
                "next_validation": next_validation,
                "next_save": next_save,
                "sampler_rng": [s.rng.bit_generator.state for s in samplers],
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
            },
            resume,
        )
        if heartbeat is not None:
            heartbeat()

    def validate():
        nonlocal best, best_fallback, best_eligible, validation_reserve
        at = time.monotonic()
        model.eval()
        result = score(model, data, subset, device)
        valid = eligible(result, incumbent_subset)
        value = result["rmse_48h"]
        # Keep a valid candidate in preference to any lower-RMSE failing candidate.
        improved = (valid and (not best_eligible or value < best)) or (
            not best_eligible and not valid and value < best_fallback
        )
        if improved and full_updates > 0:
            best, best_fallback, best_eligible = value, value, valid
            atomic_save(
                bundle(
                    model,
                    data,
                    settings=settings,
                    updates=updates,
                    full_updates=full_updates,
                    validation=result,
                    regression_gate=valid,
                ),
                folder / "best.pt",
            )
        row = dict(
            updates=updates,
            full_updates=full_updates,
            seconds=elapsed(),
            validation=result,
            regression_gate=valid,
            selected=improved and full_updates > 0,
        )
        history.append(row)
        save_json(folder / "history.json", history)
        validation_reserve = max(validation_reserve, 1.3 * (time.monotonic() - at))
        print(
            f"{folder.name}: updates={updates}, seconds={elapsed():.0f}, "
            f"48h RMSE={value:.5f}, gate={valid}",
            flush=True,
        )

    if not resume.exists():
        save_state()
        model.eval()
        initial = score(model, data, subset, device)
        save_json(folder / "initial.json", initial)
    try:
        while elapsed() + validation_reserve + last_update_seconds < slot_seconds:
            if (
                deadline is not None
                and time.monotonic() + validation_reserve >= deadline
            ):
                break
            progress = min(elapsed() / max(slot_seconds - validation_reserve, 1), 1)
            pretrain = progress < 0.3
            fine_progress = max((progress - 0.3) / 0.7, 0)
            if pretrain:
                horizon = 1 if progress < 0.1 else 6
            else:
                horizon = (
                    1
                    if fine_progress < 0.1
                    else (
                        6
                        if fine_progress < 0.25
                        else 24 if fine_progress < 0.55 else 48
                    )
                )
            if smoke_updates is not None:
                pretrain, horizon = updates < 1, 48
            sampler = samplers[0 if pretrain else 1]
            h, t, weights = sampler.sample(batch_size)
            inputs = data.inputs(h, t, horizon).to(device)
            targets = data.targets(h, t, horizon).to(device)
            weights = weights.to(device)
            lr = 1e-3 + (1e-4 - 1e-3) * progress
            for group in optimizer.param_groups:
                group["lr"] = lr
            update_started = time.monotonic()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            predictions = rollout(model, inputs, price_enabled=not pretrain)
            squared = (predictions - targets) ** 2
            per_example = 0.5 * squared[:, 0] + 0.5 * squared.mean(1)
            loss = (weights * per_example).mean() / loss_scale
            require(bool(torch.isfinite(loss)), "Nonfinite training loss.")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 5, error_if_nonfinite=True
            )
            optimizer.step()
            updates += 1
            full_updates += int(not pretrain and horizon == 48)
            last_update_seconds = time.monotonic() - update_started
            if updates % 25 == 0:
                status = dict(
                    state="training",
                    updates=updates,
                    full_updates=full_updates,
                    seconds=elapsed(),
                    phase="pretrain" if pretrain else "fine_tune",
                    horizon=horizon,
                    loss=float(loss.detach()),
                    learning_rate=lr,
                    gradient_norm=float(norm),
                    weights_mean=float(weights.mean()),
                    initial_zero_count=int((inputs.history[:, -1, 0] == 0).sum()),
                    pools=[s.counts for s in samplers],
                )
                save_json(folder / "progress.json", status)
            if elapsed() >= next_validation and not pretrain:
                validate()
                next_validation = (int(elapsed() // 600) + 1) * 600
                save_state()
            if elapsed() >= next_save:
                next_save += 300
                save_state()
            if smoke_updates is not None and updates >= smoke_updates:
                break
        require(
            full_updates >= (1 if smoke_updates is not None else 100),
            "Insufficient completed 48h fine-tuning updates; fit is incomplete.",
        )
        validate()
        save_state()
        model, saved = load_model(folder / "best.pt", device)
        full_metrics = score(model, data, data.origins["validation"], device)
        result = {
            "settings": settings,
            "validation": full_metrics,
            "eligible": eligible(full_metrics, incumbent_all),
            "updates": updates,
            "full_updates": full_updates,
            "selected_updates": saved["updates"],
            "seconds": elapsed(),
            "overrun_seconds": max(elapsed() - slot_seconds, 0),
            "completed": True,
        }
        save_json(folder / "result.json", result)
        resume.unlink(missing_ok=True)
        save_json(folder / "progress.json", {"state": "complete", **result})
        return result
    except BaseException:
        save_state()
        raise


def run_name(configuration, seed, use_price=True):
    return f"{configuration}_seed{seed}_{'price' if use_price else 'no_price'}"


def train_experiment(
    data, output, device, slot_seconds, preflight_result, deadline, heartbeat=None
):
    screen = {}
    for configuration in CONFIGURATIONS:
        name = run_name(configuration, 42)
        save_json(output / "status.json", {"stage": "training", "run": name})
        screen[configuration] = train_run(
            data,
            output / name,
            configuration,
            42,
            True,
            device,
            slot_seconds,
            preflight_result["initial_mean"],
            preflight_result["incumbent_subset"],
            preflight_result["incumbent_validation"],
            deadline,
            heartbeat=heartbeat,
        )
    # Freeze one bounded screen. If no regression gate passes, retain a research
    # candidate for diagnostics but do not recommend replacing the incumbent.
    ranked = sorted(
        CONFIGURATIONS,
        key=lambda c: (
            not screen[c]["eligible"],
            screen[c]["validation"]["rmse_48h"],
            c,
        ),
    )
    configuration = ranked[0]
    screen_manifest = {
        "configuration": configuration,
        "screen": screen,
        "all_failed_regression_gate": not any(x["eligible"] for x in screen.values()),
    }
    path = output / "screen.json"
    if path.exists():
        require(json.loads(path.read_text()) == screen_manifest, "Screen changed.")
    else:
        save_json(path, screen_manifest)
    for use_price, seeds in ((True, (7, 2026)), (False, SEEDS)):
        for seed in seeds:
            name = run_name(configuration, seed, use_price)
            save_json(output / "status.json", {"stage": "training", "run": name})
            train_run(
                data,
                output / name,
                configuration,
                seed,
                use_price,
                device,
                slot_seconds,
                preflight_result["initial_mean"],
                preflight_result["incumbent_subset"],
                preflight_result["incumbent_validation"],
                deadline,
                heartbeat=heartbeat,
            )
    selected = {
        "configuration": configuration,
        "seeds": list(SEEDS),
        "screen": screen_manifest,
        "frozen_before_followup": True,
        "price_checkpoints": [
            str(output / run_name(configuration, s) / "best.pt") for s in SEEDS
        ],
        "control_checkpoints": [
            str(output / run_name(configuration, s, False) / "best.pt") for s in SEEDS
        ],
        "interpretation": "Previously viewed dates and seed2026; retrospective.",
        "checkpoint_sha256": {
            str(output / run_name(configuration, s, price) / "best.pt"): digest(
                output / run_name(configuration, s, price) / "best.pt"
            )
            for price in (True, False)
            for s in SEEDS
        },
    }
    if (output / "selection.json").exists():
        require(
            json.loads((output / "selection.json").read_text()) == selected,
            "Frozen selection changed.",
        )
    else:
        save_json(output / "selection.json", selected)


def evaluate_experiment(data, output, device, incumbent_folder, check_budget=None):
    selection = json.loads((output / "selection.json").read_text())
    require(selection["frozen_before_followup"], "Selection is not frozen.")
    for path, expected in selection["checkpoint_sha256"].items():
        require(digest(path) == expected, "Selected response weights changed.")
    output.joinpath("predictions").mkdir(exist_ok=True)
    configurations = [
        (selection["configuration"], seed, price)
        for price in (True, False)
        for seed in SEEDS
    ]
    configurations += [
        (c, 42, True) for c in CONFIGURATIONS if c != selection["configuration"]
    ]
    result = {
        "validation": {},
        "test": {},
        "natural_low": {},
        "curves": {},
        "trained_verification": {},
    }
    for split in ("validation", "test"):
        days = data.origins[split]
        actual = actuals(data, days, 72)
        for name, prediction in baseline_predictions(data, days, 72).items():
            result[split][name] = summarize(prediction, actual, data, days)
        engines = {"incumbent": incumbents(incumbent_folder, device)}
        for c, seed, price in configurations:
            name = run_name(c, seed, price)
            engines[name] = [load_model(output / name / "best.pt", device)[0]]
        chosen = selection["configuration"]
        engines["revised_ensemble"] = [engines[run_name(chosen, s)][0] for s in (42, 7)]
        engines["price_free_ensemble"] = [
            engines[run_name(chosen, s, False)][0] for s in (42, 7)
        ]
        for name, models in engines.items():
            if check_budget is not None:
                check_budget()
            save_json(
                output / "status.json",
                {"stage": "evaluation", "split": split, "engine": name},
            )
            path = output / "predictions" / f"{split}_{name}.npz"
            if path.exists():
                prediction = np.load(path)["prediction"]
            else:
                prediction = forecast(models, data, days, 72, device)
                temporary = path.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, prediction=prediction, origins=days)
                temporary.replace(path)
            result[split][name] = summarize(prediction, actual, data, days)
            result["natural_low"][f"{split}_{name}"] = natural_low_errors(
                models,
                data,
                split,
                device,
            )
            if split == "test" and name in (
                "revised_ensemble",
                "price_free_ensemble",
                run_name(chosen, 42),
                run_name(chosen, 7),
                run_name(chosen, 2026),
            ):
                curve = price_curves(
                    models, data, days[:: max(len(days) // 6, 1)][:6], device
                )
                np.save(output / f"curves_{name}.npy", curve.pop("household_curves"))
                result["curves"][name] = curve
        if split == "test":
            for name in (
                "revised_ensemble",
                "price_free_ensemble",
                run_name(chosen, 2026),
            ):
                sequence_days = [
                    origins_for(data.partitions, i, 120)[
                        len(origins_for(data.partitions, i, 120)) // 2
                    ]
                    for i in (1, 2)
                ]
                sequence_suite(
                    engines[name],
                    data,
                    sequence_days,
                    device,
                    output / "sequences" / name,
                    check_budget=check_budget,
                )
                if name == "revised_ensemble":
                    paths = selection["price_checkpoints"][:2]
                elif name == "price_free_ensemble":
                    paths = selection["control_checkpoints"][:2]
                else:
                    paths = [selection["price_checkpoints"][2]]
                result["trained_verification"][name] = verify_trained_raw_api(
                    paths,
                    data,
                    output / "sequences" / name,
                )
        del engines
        if device == "mps":
            torch.mps.empty_cache()
        save_json(output / "evaluation_partial.json", result)
    revised = result["test"]["revised_ensemble"]
    incumbent = result["test"]["incumbent"]
    control = result["test"]["price_free_ensemble"]
    comparisons = {}
    for name, reference in (
        ("incumbent", incumbent),
        ("price_free", control),
        ("previous_day", result["test"]["previous_day"]),
    ):
        comparisons[name] = {
            str(block): paired_blocks(
                revised["daily_mse_48h"], reference["daily_mse_48h"], block
            )
            for block in (3, 7, 14)
        }
    price_free_span = max(
        result["curves"]["price_free_ensemble"]["household_relative_span_quantiles"]
    )
    require(price_free_span <= 1e-6, "Price-free invariance failed.")
    result["comparisons"] = comparisons
    result["seed_variability"] = {}
    for price in (True, False):
        group = {}
        for horizon in (24, 48, 72):
            values = [
                result["test"][run_name(selection["configuration"], s, price)][
                    f"rmse_{horizon}h"
                ]
                for s in SEEDS
            ]
            group[str(horizon)] = {
                "mean": float(np.mean(values)),
                "std_across_seeds": float(np.std(values, ddof=1)),
                "values": values,
            }
        result["seed_variability"][str(price)] = group
    chosen = selection["configuration"]
    individual_curves = [
        np.load(output / f"curves_{run_name(chosen,s)}.npy") for s in SEEDS
    ]
    changes = np.stack([v[:, :, -1] - v[:, :, 0] for v in individual_curves])
    result["price_seed_sensitivity"] = {
        "endpoint_direction_disagreement_fraction": float(
            (np.sign(changes) != np.sign(changes[:1])).any(0).mean()
        ),
        "median_endpoint_change_kwh_by_seed": [float(np.median(v)) for v in changes],
        "scope": "Descriptive disagreement, not calibrated uncertainty.",
    }
    result["acceptance"] = {
        "24h_regression_gate": eligible(revised, incumbent),
        "48h_improves_incumbent_ci": comparisons["incumbent"]["7"]["ci95"][1] < 0,
        "48h_improves_price_free_ci": comparisons["price_free"]["7"]["ci95"][1] < 0,
        "price_free_invariance": True,
        "price_effects_measured_not_forced": True,
        "consumption_flat_fraction": revised["variety"]["flat_household_day_fraction"],
        "price_flat_fraction": result["curves"]["revised_ensemble"][
            "flat_response_fraction"
        ],
        "causal_validation": False,
    }
    save_json(output / "evaluation.json", result)
    write_report(output)
    save_json(
        output / "status.json",
        {
            "stage": "complete",
            "training_complete": True,
            "evaluation_complete": True,
            "acceptance": result["acceptance"],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=ROOT / "cleaned")
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/response_refinement_v2"
    )
    parser.add_argument("--incumbent", type=Path, default=ROOT / "runs/comparison_v1")
    parser.add_argument("--device", default="mps", choices=["mps"])
    parser.add_argument("--budget-hours", type=float, default=10)
    parser.add_argument("--slot-seconds", type=float, default=3000)
    parser.add_argument(
        "--stage", choices=["all", "audit", "train", "evaluate"], default="all"
    )
    args = parser.parse_args()
    require(args.budget_hours > 0 and args.slot_seconds >= 600, "Invalid budgets.")
    require(
        args.slot_seconds * 8 <= args.budget_hours * 3600 * 0.8,
        "Training fits must leave at least 20% of budget for evaluation/setup.",
    )
    require(torch.backends.mps.is_available(), "MPS unavailable; no CPU fallback.")
    require(
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
        "Disable PYTORCH_ENABLE_MPS_FALLBACK for this explicit-MPS experiment.",
    )
    torch.set_num_threads(4)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "run.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ValueError(
            f"Run lock exists: {lock}; verify the owning process before recovery."
        )
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    started = time.monotonic()
    previous = (
        json.loads((output / "budget.json").read_text())["consumed"]
        if (output / "budget.json").exists()
        else 0
    )

    def heartbeat():
        save_json(
            output / "budget.json", {"consumed": previous + time.monotonic() - started}
        )

    deadline = started + args.budget_hours * 3600 - previous

    def check_budget():
        heartbeat()
        require(time.monotonic() < deadline, "Global compute budget exhausted.")

    try:
        data = ConsumerData(args.cleaned, args.data, output / "prepared")
        manifest = signature(data, args.budget_hours, args.slot_seconds, args.incumbent)
        if (output / "manifest.json").exists():
            require(
                json.loads((output / "manifest.json").read_text()) == manifest,
                "Data/code/objective/dependency fingerprint changed; use fresh output.",
            )
        else:
            save_json(output / "manifest.json", manifest)
            snapshot = output / "source"
            snapshot.mkdir()
            for source in Path(__file__).parent.glob("*.py"):
                (snapshot / source.name).write_text(source.read_text())
            shared = snapshot / "shared"
            shared.mkdir()
            for name in manifest["shared_sources"]:
                (shared / name).write_text((ROOT / name).read_text())
        if args.stage == "all" and (output / "status.json").exists():
            prior = json.loads((output / "status.json").read_text())
            if prior.get("stage") == "complete":
                print(f"Experiment already complete: {output}", flush=True)
                return
        if args.stage in ("all", "audit", "train"):
            save_json(
                output / "status.json", {"stage": "preflight", "pid": os.getpid()}
            )
            checks = preflight(
                data, output, args.device, args.incumbent, args.slot_seconds
            )
            if args.stage == "audit":
                save_json(
                    output / "status.json",
                    {"stage": "preflight_complete", "pid": os.getpid()},
                )
                return
        if args.stage in ("all", "train"):
            train_experiment(
                data,
                output,
                args.device,
                args.slot_seconds,
                checks,
                deadline,
                heartbeat,
            )
        if args.stage in ("all", "evaluate"):
            require(time.monotonic() < deadline, "Budget exhausted before evaluation.")
            evaluate_experiment(data, output, args.device, args.incumbent, check_budget)
    except BaseException as error:
        save_json(
            output / "status.json",
            {"stage": "failed", "error": repr(error), "pid": os.getpid()},
        )
        raise
    finally:
        heartbeat()
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()

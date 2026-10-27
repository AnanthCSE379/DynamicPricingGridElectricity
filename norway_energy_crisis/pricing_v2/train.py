"""One entry point for the fixed six-fit pricing checkpoint and its reports."""

import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time

import numpy as np
import torch
from stable_baselines3.common.logger import configure

from norway_energy_crisis.clean_data import ROOT, digest, require
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.capacity.scenarios import calibrate
from norway_energy_crisis.dqn.environment import fingerprint, price_grid
from norway_energy_crisis.dqn.train import (
    SlotClock,
    make_agent,
    atomic_model,
    save_resume,
    load_resume,
)
from norway_energy_crisis.inspection.recorder import EpisodeRecorder
from .engine import ResponseEngine
from .environment import EpisodeProvider, StressPriceEnv
from .evaluation import (
    baselines,
    evaluate,
    read_json,
)
from .objective import VERSION
from .io import save_json
from .audit import simulator_audit

SEEDS = (42, 7, 2026)
RUN_ORDER = tuple((family, seed) for seed in SEEDS for family in ("old", "new"))


def source_hashes():
    """Fingerprint simulation, reward, learner and recorder code, excluding the UI."""
    files = list(Path(__file__).parent.glob("*.py"))
    files += [
        ROOT / name
        for name in (
            "preprocess.py",
            "clean_data.py",
            "train_response.py",
            "response_v2/model.py",
            "response_v2/data.py",
            "dqn/environment.py",
            "dqn/train.py",
            "capacity/scenarios.py",
            "inspection/recorder.py",
            "inspection/store.py",
        )
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(files)}


@contextmanager
def experiment_lock(output):
    """Allow one writer; a dead process's lock may be recovered on rerun."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "experiment.lock"
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            lock.unlink()
        else:
            raise RuntimeError(f"This experiment already has an active process: {pid}")
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def preflight(data, provider, engines, scenarios, cache, output, signature):
    """Require real full-cohort inference/update/recovery before timed training."""
    path = Path(output) / "preflight.json"
    if path.exists() and read_json(path).get("signature") == signature:
        return read_json(path)
    results = {
        "signature": signature,
        "device": "mps",
        "households": provider.households,
        "engines": {},
    }
    for family, engine in engines.items():
        env = StressPriceEnv(provider, engine, scenarios, cache)
        observation, _ = env.reset(
            seed=42, options={"episode_index": 0, "scenario_index": 0}
        )
        before = [
            [p.detach().cpu().clone() for p in m.parameters()] for m in engine.models
        ]
        predicted = engine.predict(env.history, 0.7, env.episode.future_calendar[0])
        cpu = ResponseEngine(data, family, "cpu")
        expected = cpu.predict(env.history, 0.7, env.episode.future_calendar[0])
        np.testing.assert_allclose(predicted, expected, rtol=3e-3, atol=3e-4)
        with tempfile.TemporaryDirectory() as temporary:
            recorded = EpisodeRecorder(env, temporary, {"preflight": family})
            clock = SlotClock(600)
            model = make_agent(
                recorded,
                "mps",
                42,
                clock,
                dict(buffer_size=64, learning_starts=0, batch_size=8),
            )
            model.set_logger(configure(None, []))
            recorded.bind(model, clock)
            started = time.monotonic()
            model.learn(48, log_interval=None)
            require(
                recorded.hour == 0 and model._n_updates > 0,
                "Incomplete preflight learner block.",
            )
            require(
                next(model.policy.parameters()).device.type == "mps",
                "Policy is not on MPS.",
            )
            state = {"signature": {"preflight": family}}
            resume = Path(temporary) / "resume.pkl"
            save_resume(resume, model, recorded, clock, state, "mps")
            restored, restored_clock, _ = load_resume(
                resume, recorded, 600, state["signature"], "mps"
            )
            restored_observation = recorded.observation()
            np.testing.assert_array_equal(
                model.predict(restored_observation, deterministic=True)[0],
                restored.predict(restored_observation, deterministic=True)[0],
            )
            require(restored.num_timesteps == 48, "Recovery lost decisions.")
            require(
                all(
                    torch.equal(p.detach().cpu(), value)
                    for m, stored in zip(engine.models, before)
                    for p, value in zip(m.parameters(), stored)
                ),
                "Response weights changed.",
            )
            results["engines"][family] = dict(
                max_cpu_mps_difference=float(abs(predicted - expected).max()),
                block_seconds=time.monotonic() - started,
                updates=model._n_updates,
                recovery=True,
                frozen=True,
            )
            recorded.close()
            model.logger.close()
            del restored, model
        del cpu
        torch.mps.empty_cache()
    control = ResponseEngine(data, "control", "mps")
    control_env = StressPriceEnv(provider, control, scenarios, cache)
    paths = []
    for action in (0, 20):
        control_env.reset(options={"episode_index": 0, "scenario_index": 0})
        for _ in range(48):
            control_env.step(action)
        paths.append(np.asarray(control_env.demands))
    np.testing.assert_array_equal(*paths)
    results["price_free_invariance"] = True
    save_json(path, results)
    return results


def fit(
    provider,
    validation,
    engine,
    scenarios,
    cache,
    output,
    seed,
    seconds,
    protocol,
    device="mps",
    validation_seconds=600.0,
    checkpoint_seconds=300.0,
    overrides=None,
):
    """One wall-clock slot; validation, recording and checkpointing count inside it.

    - Checkpoint only after a complete 48-step episode and learner updates.
    - Choose minimum validation cost; exact ties retain the earlier checkpoint.
    - Preserve discarded interactions on recovery without silently extending time.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        protocol=protocol,
        family=engine.family,
        seed=seed,
        seconds=seconds,
        device=device,
        validation_seconds=validation_seconds,
        checkpoint_seconds=checkpoint_seconds,
        overrides=overrides or {},
    )
    config = output / "config.json"
    if config.exists():
        require(
            read_json(config) == identity, "Fit inputs changed; choose a new output."
        )
    else:
        save_json(config, identity)
    marker = output.parent / "active_run.json"
    resume = output.parent / "active_resume.pkl"
    if (output / "result.json").exists():
        if marker.exists() and read_json(marker)["run"] == output.name:
            resume.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        return read_json(output / "result.json")
    if marker.exists():
        require(
            read_json(marker)["run"] == output.name, "Recovery belongs to another fit."
        )
    else:
        require(
            not resume.exists(), "Orphaned recovery state; inspect before continuing."
        )
        save_json(marker, {"run": output.name})
    env = EpisodeRecorder(
        StressPriceEnv(provider, engine, scenarios, cache),
        output,
        {"fit": identity, "source_run": str(output.resolve())},
    )
    val_env = StressPriceEnv(validation, engine, scenarios, cache)
    if resume.exists():
        model, clock, state = load_resume(resume, env, seconds, identity, device)
        for path in (output / "progress.json", output / "best.json"):
            if path.exists():
                saved = read_json(path)
                clock.consumed = max(clock.consumed, float(saved.get("seconds", 0)))
        if (output / "history.json").exists():
            state["history"] = read_json(output / "history.json")
        if (output / "best.json").exists():
            state["best_score"] = read_json(output / "best.json")["validation"][
                "mean_cost"
            ]
    else:
        clock = SlotClock(seconds)
        model = make_agent(env, device, seed, clock, overrides)
        env.reset(seed=seed)
        state = dict(
            signature=identity,
            best_score=None,
            history=[],
            next_validation=0.0,
            next_checkpoint=checkpoint_seconds,
            validation_reserve=0.0,
            block_reserve=0.0,
        )
    env.bind(model, clock)
    model.set_logger(configure(str(output), ["csv"]))

    def validate():
        started = time.monotonic()
        save_json(
            output.parent / "status.json",
            dict(
                stage="validation",
                run=output.name,
                decisions=model.num_timesteps,
                seconds=clock.elapsed,
                budget_seconds=seconds,
            ),
        )
        result = evaluate(val_env, model)
        row = dict(
            seconds=clock.elapsed,
            decisions=model.num_timesteps,
            updates=model._n_updates,
            mean_cost=result["mean_cost"],
            terms=result["terms"],
            stats=result["stats"],
            learning_rate=clock.learning_rate(),
            epsilon=clock.epsilon(),
        )
        state["history"].append(row)
        if state["best_score"] is None or result["mean_cost"] < state["best_score"]:
            state["best_score"] = result["mean_cost"]
            atomic_model(model, output / "best.zip")
            save_json(
                output / "best.json",
                dict(
                    signature=identity,
                    validation=result,
                    **row,
                    household_ids=provider.preprocessor.ids,
                    preprocessing=provider.preprocessor.state,
                    feature_shapes=env.feature_shapes,
                    initial_untrained=model.num_timesteps == 0,
                ),
            )
        save_json(output / "history.json", state["history"])
        model.logger.record("validation/cost", result["mean_cost"])
        model.logger.record("budget/seconds", clock.elapsed)
        model.logger.dump(step=model.num_timesteps)
        state["validation_reserve"] = max(
            state["validation_reserve"], 1.1 * (time.monotonic() - started)
        )
        state["next_validation"] = (
            int(clock.elapsed / validation_seconds) + 1
        ) * validation_seconds
        print(
            f"{output.name}: validation={result['mean_cost']:.6f}, "
            f"best={state['best_score']:.6f}, decisions={model.num_timesteps}, "
            f"budget={clock.elapsed:.0f}/{seconds:.0f}s",
            flush=True,
        )

    try:
        if not state["history"]:
            validate()
            save_resume(resume, model, env, clock, state, device)
        while (
            clock.elapsed + state["validation_reserve"] + state["block_reserve"]
            < seconds
        ):
            started = time.monotonic()
            model.learn(48, reset_num_timesteps=False, log_interval=None)
            require(env.hour == 0, "Expected automatic reset at the episode boundary.")
            state["block_reserve"] = max(
                state["block_reserve"], 1.1 * (time.monotonic() - started)
            )
            if clock.elapsed >= state["next_validation"]:
                validate()
            if clock.elapsed >= state["next_checkpoint"]:
                state["next_checkpoint"] = (
                    int(clock.elapsed / checkpoint_seconds) + 1
                ) * checkpoint_seconds
                save_resume(resume, model, env, clock, state, device)
            progress = dict(
                run=output.name,
                seconds=clock.elapsed,
                budget_seconds=seconds,
                decisions=model.num_timesteps,
                updates=model._n_updates,
                best_validation_cost=state["best_score"],
                learning_rate=clock.learning_rate(),
                epsilon=clock.epsilon(),
            )
            save_json(output / "progress.json", progress)
            save_json(output.parent / "status.json", dict(stage="training", **progress))
        if state["history"][-1]["decisions"] != model.num_timesteps:
            validate()
        best = read_json(output / "best.json")
        result = dict(
            family=engine.family,
            seed=seed,
            validation=best["validation"],
            decisions=model.num_timesteps,
            updates=model._n_updates,
            best_decisions=best["decisions"],
            initial_untrained=best["initial_untrained"],
            elapsed_seconds=clock.elapsed,
            overrun_seconds=max(clock.elapsed - seconds, 0),
            best_checkpoint=str((output / "best.zip").resolve()),
        )
        save_json(output / "result.json", result)
        resume.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        return result
    finally:
        model.logger.close()
        env.close()


def run(
    output=ROOT / "runs/pricing_checkpoint_v1",
    budget_hours=10.0,
    device="mps",
    stage="all",
):
    require(
        device == "mps" and torch.backends.mps.is_available(),
        "MPS is required; no CPU fallback.",
    )
    require(
        os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") != "1",
        "Disable MPS fallback explicitly.",
    )
    require(budget_hours > 0, "Positive training budget required.")
    torch.set_num_threads(4)
    output = Path(output)
    with experiment_lock(output):
        data = PreparedData(ROOT / "prepared")
        train = EpisodeProvider(data, "train")
        validation = EpisodeProvider(data, "validation")
        require(
            data.households == 717 and len(validation.episodes) == 29,
            "Unexpected cohort/validation coverage.",
        )
        scenarios = calibrate(data)
        engines = {name: ResponseEngine(data, name, device) for name in ("old", "new")}
        settings = dict(
            version=VERSION,
            sources=source_hashes(),
            data=digest(data.folder / "manifest.json"),
            engines={k: v.signature for k, v in engines.items()},
            budget_hours=budget_hours,
            seeds=SEEDS,
            horizon=48,
            grid=price_grid(2.0).tolist(),
            scenarios=[
                dict(
                    name=s.name,
                    quantile=s.quantile,
                    capacity_kw=list(s.capacity_kw),
                    calibration=s.calibration_provenance,
                    allocation=False,
                )
                for s in scenarios
            ],
            dependencies={
                n: importlib.metadata.version(n)
                for n in ("numpy", "pandas", "torch", "gymnasium", "stable-baselines3")
            },
            python=platform.python_version(),
            device=device,
            observations=(
                "common saved scales; daily market/reference reveal; no future "
                "measured demand/weather"
            ),
            paper=dict(
                path="Papers/1812.09567v1.pdf",
                sha256=digest(ROOT.parent / "Papers/1812.09567v1.pdf"),
                adopted=(
                    "Stateful prices/loads; no assumed backlog or minimum-consumption"
                    " rule."
                ),
            ),
            retrospective=True,
            representative_seed=42,
        )
        settings = json.loads(json.dumps(settings))
        path = output / "settings.json"
        if path.exists():
            require(
                read_json(path) == settings,
                "Experiment changed; choose a new output directory.",
            )
        else:
            save_json(path, settings)
        if (output / "status.json").exists() and read_json(output / "status.json").get(
            "stage"
        ) == "complete":
            print(f"Already complete: {output}", flush=True)
            return
        protocol = fingerprint(settings)
        cache = output / "reference_cache"
        if stage == "evaluate":
            selection_path = output / "selection.json"
            require(
                selection_path.exists()
                and all(
                    (output / f"{f}_seed{s}/result.json").exists() for f, s in RUN_ORDER
                ),
                "Complete all fits and freeze selection before evaluation.",
            )
            from .report import final_evaluation

            validation_baselines = {
                name: read_json(output / "validation" / name / "baselines.json")
                for name in engines
            }
            durations = read_json(output / "durations.json")
            started = time.monotonic()
            final_evaluation(
                data,
                engines,
                scenarios,
                cache,
                output,
                validation_baselines,
                selection_path,
                device,
            )
            durations["final_evaluation_seconds"] = (
                durations.get("final_evaluation_seconds", 0)
                + time.monotonic()
                - started
            )
            save_json(output / "durations.json", durations)
            save_json(
                output / "status.json", dict(stage="complete", durations=durations)
            )
            return
        save_json(output / "status.json", dict(stage="preflight", pid=os.getpid()))
        preflight(data, train, engines, scenarios, cache, output, protocol)
        if stage == "preflight":
            save_json(output / "status.json", dict(stage="preflight complete"))
            return
        durations = (
            read_json(output / "durations.json")
            if (output / "durations.json").exists()
            else {}
        )
        started = time.monotonic()
        save_json(output / "status.json", dict(stage="reward/simulator audit"))
        simulator_audit(
            data,
            {"train": train, "validation": validation},
            engines,
            scenarios,
            cache,
            output,
        )
        validation_baselines = {}
        for name, engine in engines.items():
            save_json(
                output / "status.json", dict(stage="validation baselines", engine=name)
            )
            env = StressPriceEnv(validation, engine, scenarios, cache)
            validation_baselines[name] = baselines(
                env, output / "validation" / name, output / "constant_cache"
            )
        for name, engine in engines.items():
            save_json(
                output / "status.json",
                dict(stage="training reference warmup", engine=name),
            )
            env = StressPriceEnv(train, engine, scenarios, cache)
            for index in range(len(train.episodes)):
                for day in (0, 1):
                    env.reference_cache.get(index, day)
                if index % 25 == 0 or index == len(train.episodes) - 1:
                    save_json(
                        output / "status.json",
                        dict(
                            stage="training reference warmup",
                            engine=name,
                            origins=index + 1,
                            total=len(train.episodes),
                        ),
                    )
                    print(
                        f"{name}: training references {index+1}/{len(train.episodes)}",
                        flush=True,
                    )
        durations["setup_seconds"] = (
            durations.get("setup_seconds", 0) + time.monotonic() - started
        )
        save_json(output / "durations.json", durations)
        if stage == "setup":
            return
        fits = []
        for family, seed in RUN_ORDER:
            require(
                source_hashes() == settings["sources"],
                "Source changed during the experiment.",
            )
            fits.append(
                fit(
                    train,
                    validation,
                    engines[family],
                    scenarios,
                    cache,
                    output / f"{family}_seed{seed}",
                    seed,
                    budget_hours * 3600 / 6,
                    protocol,
                    device,
                )
            )
            save_json(output / "fit_summaries.json", fits)
        durations["training_seconds"] = sum(f["elapsed_seconds"] for f in fits)
        save_json(output / "durations.json", durations)
        selection = dict(
            frozen_before_test=True,
            protocol=protocol,
            representative_seed=42,
            fits={
                f"{r['family']}_seed{r['seed']}": dict(
                    checkpoint=r["best_checkpoint"],
                    checkpoint_sha256=digest(r["best_checkpoint"]),
                    validation_cost=r["validation"]["mean_cost"],
                    best_decisions=r["best_decisions"],
                )
                for r in fits
            },
            interpretation="Previously examined dates/seeds; retrospective follow-up.",
        )
        selection_path = output / "selection.json"
        if selection_path.exists():
            require(read_json(selection_path) == selection, "Frozen selection changed.")
        else:
            save_json(selection_path, selection)
        if stage == "train":
            save_json(
                output / "status.json",
                dict(stage="training complete; selection frozen"),
            )
            return
        from .report import final_evaluation

        started = time.monotonic()
        final_evaluation(
            data,
            engines,
            scenarios,
            cache,
            output,
            validation_baselines,
            selection_path,
            device,
        )
        durations["final_evaluation_seconds"] = time.monotonic() - started
        save_json(output / "durations.json", durations)
        save_json(output / "status.json", dict(stage="complete", durations=durations))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/pricing_checkpoint_v1"
    )
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument("--device", choices=["mps"], default="mps")
    parser.add_argument(
        "--stage",
        choices=["all", "preflight", "setup", "train", "evaluate"],
        default="all",
    )
    args = parser.parse_args()
    try:
        run(args.output, args.budget_hours, args.device, args.stage)
    except BaseException as error:
        lock = args.output / "experiment.lock"
        other_writer = lock.exists() and lock.read_text().strip() != str(os.getpid())
        status = args.output / "status.json"
        already_complete = (
            status.exists() and read_json(status).get("stage") == "complete"
        )
        if args.output.exists() and not other_writer and not already_complete:
            save_json(
                args.output / "status.json",
                dict(
                    stage=(
                        "interrupted"
                        if isinstance(error, KeyboardInterrupt)
                        else "failed"
                    ),
                    error=f"{type(error).__name__}: {error}",
                    resume="Run the identical command after addressing the error.",
                ),
            )
        raise


if __name__ == "__main__":
    main()

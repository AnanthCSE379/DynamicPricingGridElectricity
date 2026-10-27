"""Audited setup, bounded training, recovery and retrospective evaluation.

Run ``python -m norway_energy_crisis.capacity.train --stage all`` on an MPS
machine. Setup and evaluation are measured separately from the nine fit slots.
"""

import argparse
from contextlib import contextmanager
import gc
import io
import json
import os
from pathlib import Path
import pickle
import time

import numpy as np
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.logger import configure

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.inspection.recorder import EpisodeRecorder
from norway_energy_crisis.dqn.environment import EpisodeProvider, fingerprint
from norway_energy_crisis.dqn.evaluation import price_support, plotting
from norway_energy_crisis.dqn.train import (
    atomic_model,
    checkpoint_paths,
    make_engine,
    parity,
    restore_rng,
    rng_snapshot,
)
from .environment import CapacityPriceEnv
from .evaluation import (
    baselines,
    cached_evaluate,
    evaluate,
    make_report,
    measured_replay,
    prefer,
    read_json,
)
from .policies import SlotClock, algorithm, attach_schedules, make_agent
from .scenarios import CONFIGURATIONS, RUN_ORDER, SEEDS, calibrate, evidence


def save_resume(path, model, env, clock, state, device):
    """Save trusted local state only after a complete update and episode boundary."""
    require(env.hour == 0, "Resume snapshots require an episode boundary.")
    if not isinstance(model, DQN):
        require(
            model.rollout_buffer.full or model.num_timesteps == 0,
            "PPO checkpoints require a completed rollout/update.",
        )
    stream = io.BytesIO()
    model.save(stream)
    payload = dict(
        state=state,
        model=stream.getvalue(),
        environment=env.snapshot(),
        replay=getattr(model, "replay_buffer", None),
        rollout=getattr(model, "rollout_buffer", None),
        last_observation=model._last_obs,
        last_episode_starts=model._last_episode_starts,
        rng=rng_snapshot(device),
    )
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Replay serialization counts against the slot, including on resumption.
        pickle.dump(dict(elapsed=clock.elapsed), handle)
    temporary.replace(path)


def load_resume(path, env, seconds, signature, configuration, device):
    """Validate provenance before restoring a trusted local pickle checkpoint."""
    with Path(path).open("rb") as handle:
        saved, timing = pickle.load(handle), pickle.load(handle)
    require(
        saved["state"]["signature"] == signature,
        "Incompatible resume fingerprint; use a new output directory.",
    )
    clock = SlotClock(seconds, configuration, timing["elapsed"])
    model = algorithm(configuration).load(
        io.BytesIO(saved["model"]), env=env, device=device, force_reset=False
    )
    if isinstance(model, DQN):
        model.replay_buffer = saved["replay"]
        model.replay_buffer.device = torch.device(device)
    else:
        model.rollout_buffer = saved["rollout"]
        model.rollout_buffer.device = torch.device(device)
    model.env.reset()  # initialize fresh Monitor and VecEnv wrappers first
    env.restore(saved["environment"])
    model._last_obs = saved["last_observation"]
    model._last_episode_starts = saved["last_episode_starts"]
    attach_schedules(model, clock)
    restore_rng(saved["rng"], device)
    return model, clock, saved["state"]


def reconcile_progress(output, model, clock, state, identity):
    """Preserve a better validation saved just before an interrupted checkpoint.

    Updates after the last atomic resume may be lost, but their durable progress
    must still consume budget. Keep their validation evidence and count known
    discarded transitions rather than granting the resumed fit extra compute.
    """
    best_path, history_path, progress_path = [
        Path(output) / name for name in ("best.json", "history.json", "progress.json")
    ]
    if best_path.exists():
        best = read_json(best_path)
        require(best["signature"] == identity, "Best policy provenance changed.")
        state["best"] = min(state["best"], best["validation"]["means"]["loss"])
        clock.consumed = max(clock.consumed, best["checkpoint_seconds"])
    rows = read_json(history_path) if history_path.exists() else state["history"]
    if len(rows) > len(state["history"]):
        state["history"] = rows
    progress = read_json(progress_path) if progress_path.exists() else {}
    records = [*state["history"], progress]
    clock.consumed = max(
        [clock.consumed, *(row.get("seconds", 0.0) for row in records)]
    )
    previous_discarded = state.get("discarded_decisions", 0)
    known_total = max(
        [
            model.num_timesteps + previous_discarded,
            *(
                row.get("decisions_collected", row.get("decisions", 0))
                for row in records
            ),
        ]
    )
    state["discarded_decisions"] = int(known_total - model.num_timesteps)


def fit(
    train_provider,
    validation_provider,
    engine,
    scenarios,
    cache,
    output,
    configuration,
    seed,
    seconds,
    device,
    signature,
    support=None,
    validation_seconds=600.0,
    checkpoint_seconds=300.0,
    overrides=None,
):
    """One equal-compute slot; final validation and complete blocks get a reserve."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(
        protocol=signature,
        configuration=configuration,
        seed=seed,
        seconds=seconds,
        device=device,
        overrides=overrides or {},
        validation_seconds=validation_seconds,
        checkpoint_seconds=checkpoint_seconds,
    )
    config = output / "config.json"
    if config.exists():
        require(
            read_json(config) == identity,
            "Fit configuration changed; use a new output.",
        )
    else:
        save_json(config, identity)
    resume = output.parent / "active_resume.pkl"
    marker = output.parent / "active_run.json"
    if (output / "result.json").exists():
        # Recover the small interruption window between committing the result
        # and deleting its now-obsolete resumable state, without touching a
        # later run's active checkpoint when earlier completed fits are skipped.
        if marker.exists() and read_json(marker)["run"] == output.name:
            resume.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
        return read_json(output / "result.json")
    if marker.exists():
        require(
            read_json(marker)["run"] == output.name,
            "The active checkpoint belongs to another fit.",
        )
    else:
        require(
            not resume.exists(), "Orphaned resume state; inspect before proceeding."
        )
        save_json(marker, dict(run=output.name))
    env = EpisodeRecorder(
        CapacityPriceEnv(train_provider, engine, scenarios, cache, support=support),
        output,
        {"fit": identity, "source_run": str(output.resolve())},
    )
    validation_env = CapacityPriceEnv(
        validation_provider, engine, scenarios, cache, support=support
    )
    if resume.exists():
        model, clock, state = load_resume(
            resume, env, seconds, identity, configuration, device
        )
        reconcile_progress(output, model, clock, state, identity)
    else:
        clock = SlotClock(seconds, configuration)
        model = make_agent(env, configuration, seed, device, clock, overrides)
        env.reset(seed=seed)
        state = dict(
            signature=identity,
            history=[],
            best=None,
            validation_reserve=0.0,
            block_reserve=0.0,
            next_validation=0.0,
            next_checkpoint=checkpoint_seconds,
            discarded_decisions=0,
        )
    model.set_logger(configure(str(output), ["csv"]))
    env.bind(model, clock)
    block = 24 if isinstance(model, DQN) else model.n_steps
    require(block % 24 == 0, "Every learner block must contain complete days.")

    def validate():
        started = time.monotonic()
        result = evaluate(validation_env, model)
        row = dict(
            seconds=clock.elapsed,
            decisions=model.num_timesteps,
            decisions_collected=model.num_timesteps + state["discarded_decisions"],
            updates=model._n_updates,
            learning_rate=clock.learning_rate(),
            epsilon=clock.epsilon() if isinstance(model, DQN) else None,
            **result["means"],
        )
        state["history"].append(row)
        # Within a fit, retain the lowest loss, without tolerance-based promotion
        # of a checkpoint with a slightly larger primary endpoint.
        if state["best"] is None or result["means"]["loss"] < state["best"]:
            state["best"] = result["means"]["loss"]
            atomic_model(model, output / "best.zip")
            save_json(
                output / "best.json",
                dict(
                    signature=identity,
                    validation=result,
                    checkpoint_seconds=clock.elapsed,
                    decisions=model.num_timesteps,
                    parameters=sum(
                        parameter.numel() for parameter in model.policy.parameters()
                    ),
                    household_ids=train_provider.preprocessor.ids,
                    preprocessing=train_provider.preprocessor.state,
                    feature_order=env.feature_order,
                    observation_shapes={
                        key: list(space.shape)
                        for key, space in env.observation_space.spaces.items()
                    },
                ),
            )
        save_json(output / "history.json", state["history"])
        model.logger.record("validation/service_loss", result["means"]["loss"])
        model.logger.record("budget/seconds", clock.elapsed)
        model.logger.dump(step=model.num_timesteps)
        print(
            f"{output.name}: validation L={result['means']['loss']:.7f}; "
            f"best={state['best']:.7f}; {model.num_timesteps} decisions; "
            f"{clock.elapsed:.1f}/{seconds:.1f}s",
            flush=True,
        )
        state["validation_reserve"] = max(
            state["validation_reserve"], 1.15 * (time.monotonic() - started)
        )
        state["next_validation"] = (
            int(clock.elapsed / validation_seconds) + 1
        ) * validation_seconds

    try:
        if not state["history"]:
            validate()
            save_resume(resume, model, env, clock, state, device)
        next_progress = 0.0
        while (
            clock.elapsed + state["validation_reserve"] + state["block_reserve"]
            < seconds
        ):
            started = time.monotonic()
            model.learn(
                total_timesteps=block, reset_num_timesteps=False, log_interval=None
            )
            require(env.hour == 0, "Learner block did not end at a day boundary.")
            state["block_reserve"] = max(
                state["block_reserve"], 1.2 * (time.monotonic() - started)
            )
            if clock.elapsed >= state["next_validation"]:
                validate()
            if clock.elapsed >= state["next_checkpoint"]:
                state["next_checkpoint"] = (
                    int(clock.elapsed / checkpoint_seconds) + 1
                ) * checkpoint_seconds
                save_resume(resume, model, env, clock, state, device)
            if clock.elapsed >= next_progress:
                progress = dict(
                    configuration=configuration,
                    seed=seed,
                    seconds=clock.elapsed,
                    budget_seconds=seconds,
                    decisions=model.num_timesteps,
                    decisions_collected=model.num_timesteps
                    + state["discarded_decisions"],
                    updates=model._n_updates,
                    best_validation_loss=state["best"],
                    learning_rate=clock.learning_rate(),
                    epsilon=clock.epsilon() if isinstance(model, DQN) else None,
                )
                save_json(output / "progress.json", progress)
                save_json(
                    output.parent / "status.json",
                    dict(stage="training", run=output.name, **progress),
                )
                print(
                    f"{output.name}: {model.num_timesteps} decisions,"
                    f" {model._n_updates} updates, {clock.elapsed:.0f}/{seconds:.0f}s",
                    flush=True,
                )
                next_progress = clock.elapsed + 60.0
        if state["history"][-1]["decisions"] != model.num_timesteps:
            validate()
        best = read_json(output / "best.json")
        result = dict(
            configuration=configuration,
            seed=seed,
            validation=best["validation"],
            decisions=model.num_timesteps,
            decisions_collected=model.num_timesteps + state["discarded_decisions"],
            known_discarded_decisions=state["discarded_decisions"],
            updates=model._n_updates,
            best_decisions=best["decisions"],
            elapsed_seconds=clock.elapsed,
            budget_seconds=seconds,
            overrun_seconds=max(clock.elapsed - seconds, 0.0),
            parameters=best["parameters"],
            best_checkpoint=str((output / "best.zip").resolve()),
        )
        save_json(output / "result.json", result)
        resume.unlink(missing_ok=True)
        marker.unlink(missing_ok=True)
        return result
    finally:
        model.logger.close()
        env.close()
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()


def preflight(data, provider, engine, scenarios, cache, output, response_root, device):
    """Full-cohort inference, parity, gradients and policy serialization checks."""
    from tempfile import TemporaryDirectory

    result = dict(response=parity(data, engine, provider, response_root), policies={})
    env = CapacityPriceEnv(provider, engine, scenarios, cache)
    observation, _ = env.reset(seed=42)
    for configuration in CONFIGURATIONS:
        clock = SlotClock(4000, configuration)
        overrides = (
            dict(learning_starts=0, buffer_size=192) if configuration == "dqn" else {}
        )
        model = make_agent(env, configuration, 42, device, clock, overrides)
        model.set_logger(configure(None, []))
        before = [
            parameter.detach().cpu().clone() for parameter in model.policy.parameters()
        ]
        # DQN samples a full batch (with replacement); PPO uses its actual
        # 192-step rollout, batch size and five passes before saving.
        model.learn(
            total_timesteps=24 if configuration == "dqn" else 192, log_interval=None
        )
        require(model._n_updates > 0, "Preflight did not execute a gradient update.")
        require(
            all(
                torch.isfinite(parameter).all().item()
                for parameter in model.policy.parameters()
            ),
            "Nonfinite policy weights after an update.",
        )
        require(
            any(
                not torch.equal(left, right.detach().cpu())
                for left, right in zip(before, model.policy.parameters())
            ),
            "No parameter changed.",
        )
        require(
            all(
                not member.training
                and not any(p.requires_grad for p in member.parameters())
                for member in engine.models
            ),
            "Response weights must stay frozen.",
        )
        cpu = algorithm(configuration)(
            "MultiInputPolicy",
            env,
            device="cpu",
            policy_kwargs=model.policy_kwargs,
            **({"n_steps": 192} if configuration != "dqn" else {"buffer_size": 1}),
        )
        cpu.policy.load_state_dict(model.policy.state_dict())

        def output_values(policy):
            values, _ = policy.obs_to_tensor(observation)
            with torch.no_grad():
                if configuration == "dqn":
                    return policy.q_net(values).cpu().numpy()
                return (
                    policy.get_distribution(values, action_masks=np.ones((1, 21), bool))
                    .distribution.probs.cpu()
                    .numpy()
                )

        np.testing.assert_allclose(
            output_values(model.policy), output_values(cpu.policy), rtol=1e-3, atol=2e-4
        )
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "best.zip"
            atomic_model(model, path)
            loaded = algorithm(configuration).load(path, device=device)
            np.testing.assert_allclose(
                output_values(model.policy),
                output_values(loaded.policy),
                rtol=1e-5,
                atol=1e-6,
            )
            recovery = Path(temporary) / "resume.pkl"
            save_resume(
                recovery, model, env, clock, dict(signature="preflight"), device
            )
            restored_env = CapacityPriceEnv(provider, engine, scenarios, cache)
            recovered, _, _ = load_resume(
                recovery, restored_env, 4000, "preflight", configuration, device
            )
            np.testing.assert_allclose(
                output_values(model.policy),
                output_values(recovered.policy),
                rtol=1e-5,
                atol=1e-6,
            )
        result["policies"][configuration] = dict(
            households=data.households,
            decisions=model.num_timesteps,
            updates=model._n_updates,
            parameters=sum(
                parameter.numel() for parameter in model.policy.parameters()
            ),
            finite_gradients=all(
                parameter.grad is None or torch.isfinite(parameter.grad).all().item()
                for parameter in model.policy.parameters()
            ),
            cpu_device_parity=True,
            checkpoint_roundtrip=True,
            completed_update_recovery=True,
        )
        require(
            result["policies"][configuration]["finite_gradients"],
            "Nonfinite gradients.",
        )
        print(
            f"Preflight {configuration}: full cohort, gradients, parity and policy"
            " reload passed",
            flush=True,
        )
        model.logger.close()
        del model, cpu, loaded, recovered, before
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    save_json(Path(output) / "preflight.json", result)
    return result


def freeze_selection(output, results, fixed_action, validation_baselines, signature):
    """Predeclare representative seed 42; family selection uses all three seeds."""
    family_metrics = {}
    selected, best = None, None
    for configuration in CONFIGURATIONS:
        rows = [
            result["validation"]["means"]
            for result in results
            if result["configuration"] == configuration
        ]
        require(len(rows) == 3, "Family selection requires all three completed fits.")
        means = {
            key: (
                float(np.mean([row[key] for row in rows if row[key] is not None]))
                if any(row[key] is not None for row in rows)
                else None
            )
            for key in rows[0]
        }
        family_metrics[configuration] = means
        if prefer(means, best):
            selected, best = configuration, means
    baseline = "fixed"
    for name in ("market", "rule"):
        if prefer(
            validation_baselines[name]["means"], validation_baselines[baseline]["means"]
        ):
            baseline = name
    selection = dict(
        frozen_before_test=True,
        protocol=signature,
        configuration=selected,
        representative_seed=42,
        fixed_action=int(fixed_action),
        baseline=baseline,
        validation_family_means=family_metrics,
        policies={
            f"{name}_seed{seed}": digest(
                Path(output) / f"{name}_seed{seed}" / "best.zip"
            )
            for name, seed in RUN_ORDER
        },
        interpretation=(
            "Retrospective follow-up; test months and response seed 2026 were"
            " previously examined."
        ),
    )
    path = Path(output) / "selection.json"
    if path.exists():
        require(
            read_json(path) == selection,
            "Selection is frozen; policies or inputs changed.",
        )
    else:
        save_json(path, selection)
    return selection


def policy_results(
    output,
    provider,
    engine,
    scenarios,
    cache,
    support,
    signature,
    configurations=CONFIGURATIONS,
    alpha=0.5,
):
    results = {}
    env = CapacityPriceEnv(provider, engine, scenarios, cache, support=support)
    for configuration in configurations:
        results[configuration] = []
        for seed in SEEDS:
            path = Path(output) / f"{configuration}_seed{seed}" / "best.zip"
            model = algorithm(configuration).load(path, device=engine.device)
            identity = dict(
                protocol=signature,
                model=digest(path),
                engine=engine.signature,
                alpha=alpha,
                origins=[ep.origin for ep in provider.episodes],
            )
            result = cached_evaluate(
                env,
                model,
                Path(output)
                / "evaluation"
                / engine.label
                / f"alpha{alpha}"
                / f"{configuration}_{seed}.json",
                identity,
                alpha,
            )
            results[configuration].append(result)
            print(
                f"Evaluation {engine.label} alpha={alpha} {configuration}/{seed}: "
                f"L={result['means']['loss']:.7f}",
                flush=True,
            )
            del model
    return results


def sensitivity_baselines(
    provider, engine, scenarios, cache, output, support, signature, fixed_action, alpha
):
    env = CapacityPriceEnv(provider, engine, scenarios, cache, support=support)
    result = {}
    for name, policy in (
        ("fixed", lambda current, observation: fixed_action),
        ("market", lambda current, observation: current.market_action()),
        ("rule", lambda current, observation: current.rule_action()),
    ):
        result[name] = cached_evaluate(
            env,
            policy,
            Path(output) / f"{name}.json",
            dict(
                protocol=signature,
                engine=engine.signature,
                fixed_action=fixed_action,
                alpha=alpha,
            ),
            alpha,
        )
    return result


def final_evaluation(
    data,
    response_root,
    output,
    scenarios,
    cache,
    support,
    signature,
    selection,
    device,
    durations,
    manifest,
    limit_days=None,
):
    """No test snapshots or reserved response engine exist before selection freezes."""
    seal = Path(output) / "selection.json"
    require(read_json(seal) == selection, "Missing or changed frozen selection.")
    for name, expected in selection["policies"].items():
        require(
            digest(Path(output) / name / "best.zip") == expected,
            "A frozen selected/comparison policy changed.",
        )
    provider = EpisodeProvider(data, "test", seal)
    if limit_days is not None:
        provider.episodes = provider.episodes[:limit_days]
    primary, sensitivities = {}, {}
    started = time.monotonic()
    for name in ("ensemble", "reserved", "control"):
        engine = make_engine(response_root, name, data, device, seal)
        base, _ = baselines(
            provider,
            engine,
            scenarios,
            cache,
            Path(output) / "evaluation" / name / "alpha0.5" / "baselines",
            support,
            signature,
            selection["fixed_action"],
        )
        policies = policy_results(
            output, provider, engine, scenarios, cache, support, signature
        )
        primary[name] = dict(baselines=base, policies=policies)
        if name == "control":
            # Prices can affect charges, never the physical request/allocation
            # trajectories of the matched price-free control under a fixed allocator.
            reference = base["fixed"]
            for result in [
                *base.values(),
                *(run for runs in policies.values() for run in runs),
            ]:
                # Allocation ablations intentionally redistribute deliveries.
                if result["days"][0]["method"] != "fair":
                    continue
                for left, right in zip(result["days"], reference["days"]):
                    np.testing.assert_allclose(
                        [row["requested"] for row in left["hourly"]],
                        [row["requested"] for row in right["hourly"]],
                        atol=1e-5,
                    )
                    np.testing.assert_allclose(
                        left["delivered_kwh"], right["delivered_kwh"], atol=1e-5
                    )
        if name == "ensemble":
            for alpha in (0.25, 0.75):
                sensitivities[f"ensemble_alpha{alpha}"] = dict(
                    baselines=sensitivity_baselines(
                        provider,
                        engine,
                        scenarios,
                        cache,
                        Path(output)
                        / "evaluation"
                        / name
                        / f"alpha{alpha}"
                        / "baselines",
                        support,
                        signature,
                        selection["fixed_action"],
                        alpha,
                    ),
                    policies=policy_results(
                        output,
                        provider,
                        engine,
                        scenarios,
                        cache,
                        support,
                        signature,
                        (selection["configuration"],),
                        alpha,
                    ),
                )
        del engine
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    for name in ("s42", "s7"):
        engine = make_engine(response_root, name, data, device, seal)
        sensitivities[name] = dict(
            baselines=sensitivity_baselines(
                provider,
                engine,
                scenarios,
                cache,
                Path(output) / "evaluation" / name / "alpha0.5" / "baselines",
                support,
                signature,
                selection["fixed_action"],
                0.5,
            ),
            policies=policy_results(
                output,
                provider,
                engine,
                scenarios,
                cache,
                support,
                signature,
                (selection["configuration"],),
            ),
        )
        del engine
    measured = {}
    for alpha in (0.25, 0.5, 0.75):
        path = Path(output) / "evaluation" / f"historical_alpha{alpha}.json"
        if path.exists():
            measured[str(alpha)] = read_json(path)
        else:
            measured[str(alpha)] = measured_replay(data, provider, scenarios, alpha)
            save_json(path, measured[str(alpha)])
    durations["evaluation_seconds"] = time.monotonic() - started
    report = make_report(
        output, selection, primary, measured, sensitivities, durations, manifest
    )
    learning_plots(output)
    save_json(Path(output) / "durations.json", durations)
    return report


def learning_plots(output):
    plt = plotting(output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for configuration, seed in RUN_ORDER:
        history = read_json(
            Path(output) / f"{configuration}_seed{seed}" / "history.json"
        )
        for ax, key in zip(axes, ("seconds", "decisions_collected")):
            ax.plot(
                [row[key] for row in history],
                [row["loss"] for row in history],
                label=f"{configuration}/{seed}",
            )
            ax.set_xlabel(key)
            ax.set_ylabel("Validation reference-service loss")
    axes[-1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(Path(output) / "learning_curves.png", dpi=160)
    plt.close(fig)


@contextmanager
def experiment_lock(output):
    """Prevent concurrent fits or evaluators from mutating the same experiment."""
    import fcntl

    path = Path(output) / ".lock"
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(
                "Another process owns this capacity experiment."
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield


def run(
    data_folder,
    response_root,
    output,
    device="mps",
    budget_hours=10.0,
    stage="all",
    check_only=False,
    limit_days=None,
    overrides=None,
):
    """Execute/resume fixed stages; fixture limits are never exposed in the CLI."""
    require(device in ("cpu", "mps"), "Explicit CPU or MPS device required.")
    if limit_days is None:
        require(device == "mps", "The official experiment requires explicit MPS.")
        require(
            abs(budget_hours - 10.0) < 1e-9,
            "The frozen protocol has exactly ten training hours.",
        )
    if device == "mps":
        require(torch.backends.mps.is_available(), "MPS unavailable; no CPU fallback.")
    torch.set_num_threads(4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with experiment_lock(output):
        started = time.monotonic()
        data = PreparedData(data_folder)
        scenarios = calibrate(data)
        manifest = evidence(data, scenarios)
        manifest.update(
            response_weights={
                name: [
                    {"path": str(path.resolve()), "sha256": digest(path)}
                    for path in checkpoint_paths(response_root, name)
                ]
                for name in ("ensemble", "reserved", "control")
            },
            protocol=dict(
                budget_hours=budget_hours,
                slot_seconds=budget_hours * 3600 / 9,
                configurations=list(CONFIGURATIONS),
                seeds=list(SEEDS),
                price_grid="0.05, 0.10, 0.20, ..., 2.00",
                history_hours=24,
                episode_hours=24,
                gamma=1,
                reward="terminal -100 L; otherwise zero",
                validation_seconds=600,
                checkpoint_seconds=300,
                fixture_limit_days=limit_days,
            ),
        )
        # Normalize tuples to their JSON representation before equality checks.
        manifest = json.loads(json.dumps(manifest))
        signature = fingerprint(manifest)
        evidence_path = output / "evidence.json"
        if evidence_path.exists():
            require(
                read_json(evidence_path) == manifest,
                "Experiment provenance changed; choose a new output.",
            )
        else:
            save_json(evidence_path, manifest)
        status_path = output / "status.json"
        if (
            stage == "all"
            and not check_only
            and status_path.exists()
            and read_json(status_path).get("stage") == "complete"
            and (output / "comparison.json").exists()
        ):
            print(
                f"Capacity experiment already complete: {output / 'REPORT.md'}",
                flush=True,
            )
            return read_json(output / "comparison.json")
        (output / "failure.json").unlink(missing_ok=True)
        cache = output / "cache" / signature
        train_provider = EpisodeProvider(data, "train")
        validation_provider = EpisodeProvider(data, "validation")
        if limit_days is None:
            require(
                data.households == 717 and len(validation_provider.episodes) == 30,
                "Expected the agreed 717-household cohort and 30 validation days.",
            )
        else:
            for provider in (train_provider, validation_provider):
                provider.episodes = provider.episodes[:limit_days]
        support = price_support(data)
        save_json(output / "price_support.json", support)
        durations = (
            read_json(output / "durations.json")
            if (output / "durations.json").exists()
            else {}
        )
        engine = make_engine(response_root, "ensemble", data, device)
        save_json(
            output / "status.json",
            dict(stage="setup", pid=os.getpid(), signature=signature),
        )
        if check_only or stage in ("audit", "all", "train"):
            preflight(
                data,
                train_provider,
                engine,
                scenarios,
                cache,
                output,
                response_root,
                device,
            )
        if check_only:
            return read_json(output / "preflight.json")
        if stage in ("audit", "all", "train"):
            # Warm policy-independent references before starting timed slots.
            for provider in (train_provider, validation_provider):
                env = CapacityPriceEnv(provider, engine, scenarios, cache)
                for index in range(len(provider.episodes)):
                    env.reference_cache.get(index)
                    if index % 30 == 0:
                        print(
                            f"Reference setup {provider.split}:"
                            f" {index+1}/{len(provider.episodes)} days",
                            flush=True,
                        )
            durations["setup_seconds"] = (
                durations.get("setup_seconds", 0.0) + time.monotonic() - started
            )
            save_json(output / "durations.json", durations)
        (output / "README.md").write_text(
            "# Capacity and allocation experiment\n\nTwo constructed regional"
            " hourly-capacity pools; no energy transfers. Curtailment is lost service,"
            " with no rebound or outage response.\n\nevidence.json records data,"
            " response weights, source, dependencies and assumptions. status.json and"
            " run.log show live progress. Each fit retains best.zip, validation history"
            " and result.json. active_resume.pkl is a trusted local recovery snapshot."
            " selection.json is frozen before retrospective follow-up evaluation."
            " REPORT.md, comparison.json, test_metrics.json and PNG plots are written"
            " on completion.\n\nResume with the identical command and software. MPS"
            " continuation is not guaranteed bitwise identical. Do not load pickle"
            " checkpoints from untrusted sources.\n"
        )
        if stage == "audit":
            return manifest
        baseline_path = output / "validation" / "baselines.json"
        if stage in ("baselines", "all", "train") or not baseline_path.exists():
            started = time.monotonic()
            save_json(
                output / "status.json",
                dict(stage="validation_baselines", pid=os.getpid()),
            )
            validation_baselines, fixed_action = baselines(
                validation_provider,
                engine,
                scenarios,
                cache,
                output / "validation",
                support,
                signature,
            )
            durations["baseline_seconds"] = (
                durations.get("baseline_seconds", 0.0) + time.monotonic() - started
            )
            save_json(output / "durations.json", durations)
        else:
            saved = read_json(baseline_path)
            validation_baselines, fixed_action = saved["results"], saved["fixed_action"]
        if stage == "baselines":
            return validation_baselines
        if stage in ("all", "train"):
            results = []
            for configuration, seed in RUN_ORDER:
                results.append(
                    fit(
                        train_provider,
                        validation_provider,
                        engine,
                        scenarios,
                        cache,
                        output / f"{configuration}_seed{seed}",
                        configuration,
                        seed,
                        budget_hours * 3600 / 9,
                        device,
                        signature,
                        support,
                        overrides=(overrides or {}).get(configuration),
                    )
                )
            save_json(output / "fit_summaries.json", results)
        else:
            results = [
                read_json(output / f"{configuration}_seed{seed}" / "result.json")
                for configuration, seed in RUN_ORDER
            ]
        durations["training_consumed_seconds"] = sum(
            result["elapsed_seconds"] for result in results
        )
        selection = freeze_selection(
            output, results, fixed_action, validation_baselines, signature
        )
        del engine
        gc.collect()
        save_json(output / "durations.json", durations)
        if stage == "train":
            save_json(
                output / "status.json",
                dict(stage="training_complete", selection=selection),
            )
            return selection
        save_json(output / "status.json", dict(stage="evaluation", pid=os.getpid()))
        result = final_evaluation(
            data,
            response_root,
            output,
            scenarios,
            cache,
            support,
            signature,
            selection,
            device,
            durations,
            manifest,
            limit_days,
        )
        save_json(
            output / "status.json",
            dict(
                stage="complete", report=str(output / "REPORT.md"), durations=durations
            ),
        )
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument(
        "--response-root", type=Path, default=ROOT / "runs/comparison_v1"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs/capacity_v1")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--budget-hours", type=float, default=10.0)
    parser.add_argument(
        "--stage",
        choices=("audit", "baselines", "train", "evaluate", "all"),
        default="all",
    )
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        run(
            args.data,
            args.response_root,
            args.output,
            args.device,
            args.budget_hours,
            args.stage,
            args.check_only,
        )
    except Exception as error:
        if args.output.exists():
            save_json(
                args.output / "failure.json",
                dict(error=type(error).__name__, message=str(error)),
            )
            save_json(
                args.output / "status.json",
                dict(
                    stage="failed",
                    error=type(error).__name__,
                    message=str(error),
                    pid=os.getpid(),
                ),
            )
        raise


if __name__ == "__main__":
    main()

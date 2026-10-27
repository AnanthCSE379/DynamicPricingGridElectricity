"""Post-hoc reward and learning audit; never retrain or alter the frozen study.

Run with --device mps outside a sandbox that hides the Apple GPU. All new
counterfactuals and diagnostics go into a separate audit directory.
"""

import argparse
import csv
from pathlib import Path
import time

import numpy as np
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.dqn.environment import (
    EpisodeProvider,
    append_history,
    fingerprint,
)
from norway_energy_crisis.dqn.train import make_engine
from norway_energy_crisis.capacity.allocation import service
from norway_energy_crisis.capacity.environment import CapacityPriceEnv
from norway_energy_crisis.capacity.evaluation import read_json
from norway_energy_crisis.capacity.policies import algorithm, choose_action
from norway_energy_crisis.capacity.scenarios import (
    calibrate,
    CONFIGURATIONS,
    SEEDS,
    sources,
)


def summary(values):
    values = np.asarray(values, dtype=float)
    return dict(
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        median=float(np.median(values)),
    )


def score_profile(reference, requested, delivered):
    """Separate daily netting from uncancelled hourly reference deficits.

    The hourly diagnostic is not measured welfare or an automatically adopted
    replacement objective. It exposes a difference between two service proxies.
    """
    reference, requested, delivered = [
        np.asarray(v, dtype=np.float64) for v in (reference, requested, delivered)
    ]
    energy = reference.sum(0)
    active = energy > 1e-6
    original = service(energy, delivered.sum(0))["loss"]
    hourly = np.maximum(reference - delivered, 0).sum(0)
    requested_deficit = np.maximum(reference - requested, 0).sum(0)
    forced_reference_deficit = (
        np.minimum(requested, reference) - np.minimum(delivered, reference)
    ).sum(0)
    hourly_loss = (
        float(np.mean(hourly[active] / energy[active])) if active.any() else 0.0
    )
    return dict(
        loss=original,
        hourly_reference_deficit=hourly_loss,
        temporal_cancellation=hourly_loss - original,
        request_reference_deficit=(
            float(np.mean(requested_deficit[active] / energy[active]))
            if active.any()
            else 0.0
        ),
        forced_reference_deficit=(
            float(np.mean(forced_reference_deficit[active] / energy[active]))
            if active.any()
            else 0.0
        ),
        over_reference_delivery_kwh=float(np.maximum(delivered - reference, 0).sum()),
        hourly_reference_shortage_kwh=float(hourly.sum()),
        forced_unserved_kwh=float((requested - delivered).sum()),
        requested_change_kwh=float(requested.sum() - reference.sum()),
        delivered_kwh=float(delivered.sum()),
        reference_kwh=float(reference.sum()),
    )


def adversarial_checks():
    reference = np.ones((2, 2))
    # A constant 2 kWh regional budget is respected in each hour. Each assumed
    # essential 0.5 kWh request is served, but timing deficits cancel in the day.
    exchanged = np.array([[0.5, 1.5], [1.5, 0.5]])
    timing = score_profile(reference, exchanged, exchanged)
    assert timing["loss"] == 0 and timing["hourly_reference_deficit"] == 0.25
    # Voluntary modeled request suppression is penalized even without overload.
    suppressed = score_profile(reference, 0.5 * reference, 0.5 * reference)
    assert suppressed["loss"] == 0.5 and suppressed["forced_unserved_kwh"] == 0
    # More simulated delivery can eliminate the score, without any price or
    # charge penalty. This is a proxy opportunity, not proof an agent used it.
    inflated = score_profile(reference, 2 * reference, 2 * reference)
    assert inflated["loss"] == 0
    assert service([1, 1], [2, 0])["loss"] == 0.5
    assert service([0, 0], [0, 0])["loss"] == 0
    return dict(
        timing_cancellation=timing,
        suppression_without_curtailment=suppressed,
        demand_inflation=inflated,
        cross_household_compensation_blocked=True,
    )


def artifacts(run):
    result = dict(training={}, baselines={}, policy_outcomes={})
    for configuration in CONFIGURATIONS:
        for seed in SEEDS:
            name = f"{configuration}_seed{seed}"
            folder = run / name
            fit = read_json(folder / "result.json")
            history = read_json(folder / "history.json")
            logs = list(csv.DictReader((folder / "progress.csv").open()))
            result["training"][name] = dict(
                initial_loss=history[0]["loss"],
                best_loss=fit["validation"]["means"]["loss"],
                final_loss=history[-1]["loss"],
                best_decisions=fit["best_decisions"],
                total_decisions=fit["decisions"],
                updates=fit["updates"],
                best_is_untrained=fit["best_decisions"] == 0,
                final_switches=history[-1]["internal_switches"],
                final_energy_weighted_price=history[-1]["charge_nok"]
                / history[-1]["delivered_kwh"],
                last_logged_update={
                    key: value
                    for key, value in logs[-1].items()
                    if key.startswith("train/")
                },
                history=history,
            )
    for dynamics in ("ensemble", "reserved", "control"):
        folder = run / "evaluation" / dynamics / "alpha0.5"
        base = read_json(folder / "baselines/baselines.json")["results"]
        result["baselines"][dynamics] = {
            name: dict(means=value["means"], by_scenario=value["by_scenario"])
            for name, value in base.items()
        }
        result["policy_outcomes"][dynamics] = {}
        for configuration in CONFIGURATIONS:
            result["policy_outcomes"][dynamics][configuration] = [
                read_json(folder / f"{configuration}_{seed}.json")["result"]["means"]
                for seed in SEEDS
            ]
    result["specification_tests"] = adversarial_checks()
    return result


def capacity_coverage(data, providers, engine, scenarios, cache):
    result = {}
    for split, provider in providers.items():
        env = CapacityPriceEnv(provider, engine, scenarios, cache)
        measured, modelled = [], []
        for index, episode in enumerate(provider.episodes):
            reference = env.reference_cache.get(index)
            modelled.append(
                np.stack(
                    [reference[:, data.area_index == area].sum(1) for area in range(2)],
                    axis=1,
                )
            )
            measured.append(
                np.stack(
                    [
                        data.demand[
                            data.area_index == area,
                            episode.origin : episode.origin + 24,
                        ].sum(0)
                        for area in range(2)
                    ],
                    axis=1,
                )
            )
        rows = {}
        for scenario in scenarios:
            rows[scenario.name] = {}
            for label, values in (
                ("measured", measured),
                ("reference_forecast", modelled),
            ):
                values = np.asarray(values, dtype=float)
                rows[scenario.name][label] = dict(
                    overloaded_region_hour_fraction=float(
                        (values > scenario.budget_kwh).mean()
                    ),
                    essential_infeasible_region_hour_fraction=float(
                        (0.5 * values > scenario.budget_kwh).mean()
                    ),
                    overloaded_day_fraction=float(
                        (values > scenario.budget_kwh).any(axis=(1, 2)).mean()
                    ),
                )
        mask = data.partitions == ("train", "validation", "test").index(split)
        result[split] = dict(
            days=len(provider.episodes),
            scarcity=rows,
            observed_price={
                str(area): summary(data.prices[area, mask]) for area in range(2)
            },
        )
    return result


def record_episode(
    env,
    policy,
    episode_index,
    scenario_index,
    stochastic_seed=None,
    q_audit=False,
    long_branches=False,
):
    observation, _ = env.reset(
        options=dict(episode_index=episode_index, scenario_index=scenario_index)
    )
    requested, delivered, qchecks, probabilities = [], [], [], []
    rewards = 0.0
    if stochastic_seed is not None:
        torch.manual_seed(stochastic_seed)
        if env.engine.device.type == "mps":
            torch.mps.manual_seed(stochastic_seed)
    for hour in range(24):
        if hasattr(policy, "policy") and not hasattr(policy, "q_net"):
            tensors, _ = policy.policy.obs_to_tensor(observation)
            with torch.no_grad():
                probs = (
                    policy.policy.get_distribution(
                        tensors, action_masks=env.action_masks()[None]
                    )
                    .distribution.probs.cpu()
                    .numpy()[0]
                )
            probabilities.append(
                dict(
                    entropy=float(-(probs * np.log(np.maximum(probs, 1e-30))).sum()),
                    maximum_probability=float(probs.max()),
                )
            )
        if q_audit and (hour == 23 or (long_branches and hour in (0, 12))):
            qchecks.append(branch_values(env, policy, observation, full=hour != 23))
        if stochastic_seed is not None:
            action = int(
                policy.predict(
                    observation, deterministic=False, action_masks=env.action_masks()
                )[0]
            )
        else:
            action = choose_action(policy, env, observation)
        old_requested, old_delivered = (
            env.requested_energy.copy(),
            env.delivered_energy.copy(),
        )
        observation, reward, done, truncated, info = env.step(action)
        rewards += reward
        assert not truncated and (reward == 0 if hour < 23 else done)
        requested.append(env.requested_energy - old_requested)
        delivered.append(env.delivered_energy - old_delivered)
    independent = score_profile(env.reference, requested, delivered)
    assert abs(rewards + 100 * independent["loss"]) < 1e-7
    assert abs(info["loss"] - independent["loss"]) < 1e-9
    assert info["postallocation_violation_region_hours"] == 0
    assert (
        abs(
            independent["hourly_reference_deficit"]
            - independent["request_reference_deficit"]
            - independent["forced_reference_deficit"]
        )
        < 1e-8
    )
    return dict(
        origin=env.episode.timestamp,
        scenario=env.scenario.name,
        dynamics=env.engine.label,
        **independent,
        prices=env.prices,
        q_checks=qchecks,
        probabilities=probabilities,
        floor_binding_fraction=info["floor_binding_fraction"],
        essential_unserved_kwh=info["essential_unserved_kwh"],
    )


def branch_values(env, model, observation, full=False):
    """Actual action values follow the saved policy after the tested first action."""
    state = env.snapshot()
    tensors, _ = model.policy.obs_to_tensor(observation)
    with torch.no_grad():
        q = model.q_net(tensors).cpu().numpy()[0]
    returns = []
    for action in range(21):
        env.restore(state)
        obs, reward, done, _, _ = env.step(action)
        while full and not done:
            obs, r, done, _, _ = env.step(choose_action(model, env, obs))
            reward += r
        returns.append(reward)
    env.restore(state)
    actual = np.asarray(returns)
    chosen, best = int(q.argmax()), int(actual.argmax())
    return dict(
        hour=env.hour,
        q_values=q.astype(float).tolist(),
        actual_returns=actual.tolist(),
        chosen_price=float(env.grid[chosen]),
        best_branch_price=float(env.grid[best]),
        regret=float(actual[best] - actual[chosen]),
        chosen_value_error=float(q[chosen] - actual[chosen]),
        positive_q_fraction=float((q > 0).mean()),
        q_span=float(np.ptp(q)),
        true_return_span=float(np.ptp(actual)),
    )


def price_curves(provider, engines, output):
    grid = np.r_[0.05, np.arange(1, 21) / 10]
    rows, curves = [], []
    # Six evenly spaced validation days, four times of day; every household.
    for index in np.linspace(0, len(provider.episodes) - 1, 6, dtype=int):
        episode = provider.episodes[index]
        history = provider.features(episode)
        for hour in range(24):
            if hour in (0, 6, 12, 18):
                predictions = {
                    name: engine.predict_candidates(
                        history, grid, episode.future_calendar[hour]
                    )
                    for name, engine in engines.items()
                }
                diffs = {}
                for name, values in predictions.items():
                    change = values[-1] - values[0]
                    diffs[name] = change
                    regional = np.stack(
                        [
                            values[:, provider.area_index == area].sum(1)
                            for area in range(2)
                        ],
                        axis=1,
                    )
                    rows.append(
                        dict(
                            origin=episode.timestamp,
                            hour=hour,
                            dynamics=name,
                            positive_household_fraction=float((change > 1e-6).mean()),
                            negative_household_fraction=float((change < -1e-6).mean()),
                            nonmonotonic_household_fraction=float(
                                (
                                    (np.diff(values, axis=0) > 1e-6).any(0)
                                    & (np.diff(values, axis=0) < -1e-6).any(0)
                                ).mean()
                            ),
                            regional_low_to_high_change_fraction=(
                                regional[-1] / np.maximum(regional[0], 1e-6) - 1
                            ).tolist(),
                        )
                    )
                    curves.append(
                        dict(
                            origin=episode.timestamp,
                            hour=hour,
                            dynamics=name,
                            regional_kwh=regional.astype(float).tolist(),
                        )
                    )
                rows.append(
                    dict(
                        origin=episode.timestamp,
                        hour=hour,
                        dynamics="sign_disagreement",
                        ensemble_reserved_household_fraction=float(
                            (
                                np.sign(diffs["ensemble"]) != np.sign(diffs["reserved"])
                            ).mean()
                        ),
                    )
                )
            prediction = engines["ensemble"].predict(
                history,
                episode.future_prices[provider.area_index, hour],
                episode.future_calendar[hour],
            )
            history = append_history(
                history,
                prediction,
                episode.future_prices[provider.area_index, hour],
                episode.future_calendar[hour],
                provider.preprocessor,
            )
        print(f"Price-response probes: {episode.timestamp}", flush=True)
    save_json(
        output / "price_curves.json",
        dict(grid=grid.tolist(), states=rows, curves=curves),
    )
    return dict(states=rows, curves=curves)


def training_constants(provider, engine, scenarios, original_cache, output, count=24):
    rng = np.random.default_rng(31415)
    indices = np.sort(
        rng.choice(
            len(provider.episodes), min(count, len(provider.episodes)), replace=False
        )
    )
    env = CapacityPriceEnv(provider, engine, scenarios, original_cache)
    records = []
    grid = env.grid
    for index in indices:
        episode = provider.episodes[index]
        reference = env.reference_cache.get(int(index))
        path = output / "cache" / f"train_{episode.origin}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                raw = saved["raw"]
        else:
            raw = np.zeros((21, 24, provider.households), np.float32)
            for start in range(0, 21, 4):
                prices = grid[start : start + 4, None]
                history = np.repeat(
                    provider.features(episode)[None], len(prices), axis=0
                )
                for hour in range(24):
                    predicted = engine.predict(
                        history, prices, episode.future_calendar[hour]
                    )
                    raw[start : start + len(prices), hour] = predicted
                    history = append_history(
                        history,
                        np.maximum(predicted, 0.5 * reference[hour]),
                        prices,
                        episode.future_calendar[hour],
                        provider.preprocessor,
                    )
            np.savez_compressed(path, raw=raw)
        for scenario in scenarios:
            for action, price in enumerate(grid):
                env.reset(
                    options=dict(
                        episode_index=int(index),
                        scenario_index=scenarios.index(scenario),
                    )
                )
                env.attach_trace(np.full(24, price), raw[action])
                record = record_attached(env, action)
                records.append(
                    dict(
                        origin=episode.timestamp,
                        scenario=scenario.name,
                        price=float(price),
                        **record,
                    )
                )
        print(
            f"Training constant-grid diagnostic: {len(records)//63}/{len(indices)}"
            " days",
            flush=True,
        )
    save_json(
        output / "training_constants.json",
        dict(
            sampling=(
                "24 uniform training days without replacement; seed 31415; diagnostic,"
                " not model selection"
            ),
            indices=indices.tolist(),
            days=records,
        ),
    )
    return records


def record_attached(env, action):
    requested, delivered = [], []
    for _ in range(24):
        old_r, old_a = env.requested_energy.copy(), env.delivered_energy.copy()
        _, reward, done, _, info = env.step(action)
        requested.append(env.requested_energy - old_r)
        delivered.append(env.delivered_energy - old_a)
    result = score_profile(env.reference, requested, delivered)
    assert done and abs(reward + 100 * result["loss"]) < 1e-7
    return result


def policy_probes(data, providers, engines, scenarios, cache, run, output, device):
    records = []
    checkpoints = {}
    candidates = [
        ("dqn", 42),
        ("dqn", 7),
        ("dqn", 2026),
        ("ppo_mlp", 42),
        ("ppo_mlp", 2026),
        ("ppo_transformer", 42),
        ("ppo_transformer", 2026),
    ]
    # Deliberately span January and the retrospective follow-up. These are
    # diagnostic cases, not a new holdout or another selection procedure.
    cases = [("validation", 0), ("validation", 29), ("test", 0), ("test", 57)]
    for dynamics in ("ensemble", "reserved"):
        engine = engines[dynamics]
        for configuration, seed in candidates:
            model = algorithm(configuration).load(
                run / f"{configuration}_seed{seed}/best.zip", device=device
            )
            checkpoints[f"{configuration}_{seed}"] = dict(
                saved_timesteps=model.num_timesteps, saved_updates=model._n_updates
            )
            for split, index in cases:
                env = CapacityPriceEnv(providers[split], engine, scenarios, cache)
                for scenario_index in range(3):
                    record = record_episode(
                        env,
                        model,
                        index,
                        scenario_index,
                        q_audit=configuration == "dqn",
                        long_branches=dynamics == "ensemble"
                        and split == "validation"
                        and index == 29
                        and scenario_index == 2
                        and seed == 42,
                    )
                    records.append(
                        dict(policy=f"{configuration}_{seed}", split=split, **record)
                    )
            if dynamics == "ensemble" and configuration != "dqn":
                env = CapacityPriceEnv(
                    providers["validation"], engine, scenarios, cache
                )
                for index in (0, 29):
                    for sampling_seed in (109, 110, 111):
                        records.append(
                            dict(
                                policy=f"{configuration}_{seed}",
                                split="validation",
                                stochastic=True,
                                sampling_seed=sampling_seed,
                                **record_episode(
                                    env, model, index, 2, stochastic_seed=sampling_seed
                                ),
                            )
                        )
            save_json(
                output / "policy_probes.json",
                dict(days=records, checkpoints=checkpoints),
            )
            print(
                "Policy decomposition and terminal-value checks:"
                f" {dynamics}/{configuration}/{seed}",
                flush=True,
            )
            del model
        for name, policy in (
            ("fixed", lambda current, obs: 20),
            ("market", lambda current, obs: current.market_action()),
        ):
            for split, index in cases:
                env = CapacityPriceEnv(providers[split], engine, scenarios, cache)
                for scenario_index in range(3):
                    records.append(
                        dict(
                            policy=name,
                            split=split,
                            **record_episode(env, policy, index, scenario_index),
                        )
                    )
    save_json(
        output / "policy_probes.json", dict(days=records, checkpoints=checkpoints)
    )
    return records


def run_audit(run, output, device="mps", stage="all"):
    output.mkdir(parents=True, exist_ok=True)
    original = read_json(run / "evidence.json")
    require(
        original["sources"] == sources(),
        "Frozen study source changed; audit the recorded version first.",
    )
    selection = read_json(run / "selection.json")
    for name, expected in selection["policies"].items():
        require(digest(run / name / "best.zip") == expected, "Frozen policy changed.")
    identity = fingerprint(
        dict(
            evidence=digest(run / "evidence.json"),
            selection=digest(run / "selection.json"),
            audit_source=digest(__file__),
            device=device,
        )
    )
    metadata = output / "provenance.json"
    if metadata.exists():
        require(
            read_json(metadata)["signature"] == identity,
            "Audit source changed; choose a new output.",
        )
    else:
        save_json(
            metadata,
            dict(
                signature=identity,
                run=str(run.resolve()),
                device=device,
                retrospective=True,
                retraining=False,
                audit_source_sha256=digest(__file__),
            ),
        )
    audit = artifacts(run)
    save_json(output / "artifact_checks.json", audit)
    if stage == "offline":
        return
    require(
        device == "mps" and torch.backends.mps.is_available(),
        "Explicit MPS required; no fallback.",
    )
    torch.set_num_threads(4)
    data = PreparedData(ROOT / "prepared")
    providers = {
        split: EpisodeProvider(data, split, run / "selection.json")
        for split in ("train", "validation", "test")
    }
    scenarios = calibrate(data)
    cache = run / "cache" / selection["protocol"]
    response_root = ROOT / "runs/comparison_v1"
    engines = {
        name: make_engine(response_root, name, data, device, run / "selection.json")
        for name in ("ensemble", "s42", "s7", "reserved")
    }
    save_json(
        output / "capacity_coverage.json",
        capacity_coverage(data, providers, engines["ensemble"], scenarios, cache),
    )
    price_curves(providers["validation"], engines, output)
    training_constants(
        providers["train"], engines["ensemble"], scenarios, cache, output
    )
    policy_probes(data, providers, engines, scenarios, cache, run, output, device)
    require(original["sources"] == sources(), "Audit altered original study source.")
    save_json(
        output / "status.json", dict(stage="diagnostics_complete", signature=identity)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/capacity_v1")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/capacity_v1_audit")
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--stage", choices=("offline", "all"), default="all")
    args = parser.parse_args()
    started = time.monotonic()
    run_audit(args.run, args.output, args.device, args.stage)
    print(
        f"Audit stage finished in {time.monotonic()-started:.1f}s: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Deterministic episode enumeration and common scoring for every comparator."""

from pathlib import Path
import json

import numpy as np

from norway_energy_crisis.clean_data import require
from norway_energy_crisis.dqn.environment import append_history, atomic_npz
from .objective import score_trajectory
from .io import save_json


def read_json(path):
    return json.loads(Path(path).read_text())


def rollout_episode(env, policy, index, scenario):
    """Run exactly 48 legal decisions; check actual rewards against full scoring."""
    observation, _ = env.reset(
        options={"episode_index": index, "scenario_index": scenario}
    )
    reward_sum = 0.0
    for _ in range(48):
        action = (
            int(policy.predict(observation, deterministic=True)[0])
            if hasattr(policy, "predict")
            else int(policy(env))
        )
        observation, reward, done, truncated, info = env.step(action)
        reward_sum += reward
    require(
        done and not truncated and abs(reward_sum + 100 * info["cost"]) < 1e-7,
        "Reward/scoring mismatch.",
    )
    return {
        name: value
        for name, value in info.items()
        if name not in ("telemetry", "telemetry_context")
    }


def summarize(days):
    require(days, "No evaluation cases.")
    return dict(
        mean_cost=float(np.mean([d["cost"] for d in days])),
        cases=len(days),
        terms={
            k: float(np.mean([d["terms"][k] for d in days]))
            for k in ("grid", "energy", "charge")
        },
        stats={
            k: float(np.mean([d["stats"][k] for d in days])) for k in days[0]["stats"]
        },
        days=days,
    )


def evaluate(env, policy, output=None):
    """Evaluate every origin×scenario once; reuse completed fingerprinted exports."""
    if output is not None and Path(output).exists():
        return read_json(output)
    days = []
    for index in range(len(env.provider.episodes)):
        for scenario in range(len(env.scenarios)):
            days.append(rollout_episode(env, policy, index, scenario))
    result = summarize(days)
    if output is not None:
        save_json(output, result)
    return result


def constant_paths(env, index, folder):
    """Enumerate all 21 constants; trajectories do not depend on capacity limits."""
    episode = env.provider.episodes[index]
    path = (
        Path(folder)
        / env.engine.signature
        / env.provider.split
        / f"{episode.origin}.npz"
    )
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return saved["demand"]
    histories = np.broadcast_to(
        episode.history, (len(env.grid), *episode.history.shape)
    ).copy()
    values = []
    for hour in range(48):
        predicted = env.engine.predict(
            histories, env.grid[:, None], episode.future_calendar[hour]
        )
        values.append(predicted)
        histories = append_history(
            histories,
            predicted,
            env.grid[:, None],
            episode.future_calendar[hour],
            env.provider.preprocessor,
        )
    demand = np.stack(values, 1)
    require(np.isfinite(demand).all() and (demand >= 0).all(), "Invalid constant path.")
    atomic_npz(path, demand=demand)
    return demand


def constant_scores(env, folder, progress=None):
    """Use identical scoring for grid enumeration and learned schedules."""
    all_days = []
    for index, episode in enumerate(env.provider.episodes):
        paths = constant_paths(env, index, folder)
        reference = np.concatenate(
            [env.reference_cache.get(index, d)[0] for d in (0, 1)]
        )
        market = np.concatenate(
            [env.provider.release_prices(index, d) for d in (0, 1)], 1
        )
        scenarios = []
        for scenario in env.scenarios:
            rows = []
            for price, demand in zip(env.grid, paths):
                rows.append(
                    dict(
                        origin=episode.timestamp,
                        scenario=scenario.name,
                        prices=np.full(48, price).tolist(),
                        **score_trajectory(
                            demand,
                            np.full(48, price),
                            reference,
                            market,
                            episode.demand_history[:, -1],
                            episode.previous_prices,
                            env.area_index,
                            scenario.capacity_kw,
                            env.training_median,
                        ),
                    )
                )
            scenarios.append(rows)
        all_days.append(scenarios)
        if progress:
            progress(index + 1, len(env.provider.episodes))
    return all_days


def baselines(env, folder, cache, fixed_actions=None):
    """Select fixed prices on validation only; oracle prices are case-specific."""
    folder = Path(folder)
    completed = folder / "baselines.json"
    if completed.exists():
        saved = read_json(completed)
        if fixed_actions is not None:
            require(
                saved["fixed_actions"] == list(fixed_actions),
                "Baseline selection differs.",
            )
        return saved
    table_file = folder / "constant_scores.json"
    if table_file.exists():
        table = read_json(table_file)
    else:

        def progress(done, total):
            save_json(
                folder / "progress.json",
                dict(stage="constant enumeration", origins=done, total=total),
            )
            if done % 5 == 0 or done == total:
                print(
                    f"{env.engine.family}/{env.provider.split}: "
                    f"constants {done}/{total}",
                    flush=True,
                )

        table = constant_scores(env, cache, progress)
        save_json(table_file, table)
    if fixed_actions is None:
        require(
            env.provider.split == "validation",
            "Choose constant prices on validation first.",
        )
        fixed_actions = np.argmin(
            np.asarray(
                [[[r["cost"] for r in scenario] for scenario in day] for day in table]
            ).mean(0),
            axis=1,
        ).tolist()
    results = {name: [] for name in ("minimum", "maximum", "fixed", "constant_oracle")}
    for day in table:
        for number, rows in enumerate(day):
            chosen = {
                "minimum": 0,
                "maximum": len(env.grid) - 1,
                "fixed": fixed_actions[number],
                "constant_oracle": int(np.argmin([r["cost"] for r in rows])),
            }
            for name, index in chosen.items():
                results[name].append(rows[index])
    results = {name: summarize(rows) for name, rows in results.items()}
    for name, policy in (
        ("market", lambda e: e.market_action()),
        ("capacity_rule", lambda e: e.rule_action()),
        ("greedy", lambda e: e.greedy_action()),
    ):
        print(f"{env.engine.family}/{env.provider.split}: baseline {name}", flush=True)
        results[name] = evaluate(env, policy, folder / f"{name}.json")
    saved = dict(fixed_actions=list(fixed_actions), results=results)
    save_json(completed, saved)
    return saved


def paired_intervals(left, right, seed=42, resamples=5000, metric="cost"):
    """Pair origins/scenarios first; average policy seeds before block sampling."""

    def series(results):
        ordered = sorted(
            {row["origin"] for result in results for row in result["days"]}
        )
        return (
            np.array(
                [
                    [
                        np.mean(
                            [
                                (d["cost"] if metric == "cost" else d["stats"][metric])
                                for d in result["days"]
                                if d["origin"] == origin
                            ]
                        )
                        for origin in ordered
                    ]
                    for result in results
                ]
            ).mean(0),
            ordered,
        )

    a, origins = series(left)
    b, other = series(right)
    require(origins == other, "Unpaired origins.")
    delta = a - b
    answer = {
        "mean_difference": float(delta.mean()),
        "direction": "left minus right; negative favors left",
    }
    for block in (7, 3, 14):
        rng = np.random.default_rng(seed + block)
        n = len(delta)
        length = min(block, n)
        # Moving, non-circular blocks avoid joining the end of March to February.
        starts = rng.integers(
            0, n - length + 1, size=(resamples, int(np.ceil(n / length)))
        )
        indices = (starts[:, :, None] + np.arange(length)).reshape(resamples, -1)[:, :n]
        sampled = delta[indices].mean(1)
        answer[str(block)] = dict(
            lower=float(np.quantile(sampled, 0.025)),
            upper=float(np.quantile(sampled, 0.975)),
            resamples=resamples,
        )
    return answer

"""Retrospective comparisons, policy-value checks and the bounded closing report."""

from pathlib import Path
import csv

import numpy as np
import torch
from stable_baselines3 import DQN

from norway_energy_crisis.clean_data import digest, require
from norway_energy_crisis.dqn.environment import append_history
from norway_energy_crisis.preprocess import calendar
from norway_energy_crisis.response_v2.data import origins_for
from .engine import ResponseEngine
from .environment import EpisodeProvider, StressPriceEnv
from .evaluation import (
    read_json,
    evaluate,
    baselines,
    paired_intervals,
    rollout_episode,
)
from .io import save_json


def value_audit(env, policy, path):
    """Compare network values with exact model returns under policy continuation.

    Branch enumeration is not an optimal-control oracle: later actions follow the
    same frozen policy. Sample only predefined validation origins/hours.
    """
    if Path(path).exists():
        return read_json(path)
    rows = []
    origins = np.unique(np.linspace(0, len(env.provider.episodes) - 1, 3, dtype=int))
    for number, index in enumerate(origins):
        for hour in (0, 12, 24, 36):
            observation, _ = env.reset(
                options={
                    "episode_index": int(index),
                    "scenario_index": number % len(env.scenarios),
                }
            )
            for _ in range(hour):
                observation, *_ = env.step(
                    int(policy.predict(observation, deterministic=True)[0])
                )
            state = env.snapshot()
            tensor, _ = policy.policy.obs_to_tensor(observation)
            with torch.no_grad():
                q = policy.q_net(tensor).cpu().numpy()[0]
            returns = []
            for action in range(len(env.grid)):
                env.restore(state)
                obs, reward, done, _, _ = env.step(action)
                total = reward
                while not done:
                    obs, reward, done, _, _ = env.step(
                        int(policy.predict(obs, deterministic=True)[0])
                    )
                    total += reward
                returns.append(total)
            chosen = int(np.argmax(q))
            best = int(np.argmax(returns))
            rows.append(
                dict(
                    origin=env.episode.timestamp,
                    hour=hour,
                    scenario=env.scenario.name,
                    chosen_price=float(env.grid[chosen]),
                    best_branch_price=float(env.grid[best]),
                    q_values=q.tolist(),
                    continuation_returns=returns,
                    regret=float(returns[best] - returns[chosen]),
                    chosen_value_error=float(q[chosen] - returns[chosen]),
                )
            )
    result = dict(
        states=len(rows),
        mean_regret=float(np.mean([r["regret"] for r in rows])),
        note=(
            "Validation model returns under the same policy's continuation; "
            "not optimal Q."
        ),
        rows=rows,
    )
    save_json(path, result)
    return result


def continuation_audit(data, env, policies, path):
    """Append 24 recorded-price hours after the 48-decision episode, diagnostically.

    - No additional learned decisions or rewards, deferred energy, or recovery rule.
    - All 72 hours remain in validation; temperature persists from the first origin.
    - This exceeds the upgraded consumer's 48-hour training rollout.
    """
    if Path(path).exists():
        return read_json(path)
    valid = set(origins_for(data.partitions, 1, 72, True).tolist())
    eligible = [i for i, e in enumerate(env.provider.episodes) if e.origin in valid]
    chosen = np.asarray(eligible)[
        np.linspace(0, len(eligible) - 1, min(3, len(eligible)), dtype=int)
    ]
    rows = []
    for index in chosen:
        for scenario in range(len(env.scenarios)):
            for name, policy in policies.items():
                original = rollout_episode(env, policy, int(index), scenario)
                controlled = env.history.copy()
                reference = env.reference_cache.get(int(index), 1)[1].copy()
                capacity = np.asarray(env.scenario.capacity_kw)
                controlled_values, reference_values = [], []
                for offset in range(48, 72):
                    stamp = env.episode.origin + offset
                    prices = np.asarray(data.prices[env.area_index, stamp])
                    time_features = calendar(np.array([data.times[stamp]]))[0]
                    prediction = env.engine.predict(controlled, prices, time_features)
                    benchmark = env.engine.predict(reference, prices, time_features)
                    controlled_values.append(prediction)
                    reference_values.append(benchmark)
                    controlled = append_history(
                        controlled,
                        prediction,
                        prices,
                        time_features,
                        env.provider.preprocessor,
                    )
                    reference = append_history(
                        reference,
                        benchmark,
                        prices,
                        time_features,
                        env.provider.preprocessor,
                    )
                q = np.asarray(controlled_values)
                q0 = np.asarray(reference_values)
                regional = np.stack([q[:, m].sum(1) for m in env.area_masks], 1)
                rows.append(
                    dict(
                        policy=name,
                        origin=env.episode.timestamp,
                        scenario=env.scenario.name,
                        episode_cost=original["cost"],
                        continuation_overload_region_hours=int(
                            (regional > capacity).sum()
                        ),
                        continuation_overload_kwh=float(
                            np.maximum(regional - capacity, 0).sum()
                        ),
                        day3_energy_change_fraction=float(
                            q.sum() / max(q0.sum(), 1e-6) - 1
                        ),
                    )
                )
    result = dict(
        note=(
            "24-hour fixed recorded-regional-price continuation; diagnostic "
            "only, beyond training horizon. Not observed rebound or queued "
            "demand."
        ),
        rows=rows,
    )
    save_json(path, result)
    return result


def final_evaluation(
    data,
    engines,
    scenarios,
    cache,
    output,
    validation_baselines,
    selection_path,
    device,
):
    """Freeze selection, compare all seeds on common dynamics and close the study."""
    output = Path(output)
    selection = read_json(selection_path)
    require(selection["frozen_before_test"], "Selection is not frozen.")
    provider = EpisodeProvider(data, "test", selection_path)
    require(len(provider.episodes) == 57, "Unexpected follow-up coverage.")
    engines = {
        **engines,
        "reserved": ResponseEngine(
            data, "reserved", device, selection_file=selection_path
        ),
        "control": ResponseEngine(data, "control", device),
    }
    comparisons = {}
    all_results = {}
    for dynamics, engine in engines.items():
        folder = output / "evaluation" / dynamics
        save_json(
            output / "status.json", dict(stage="follow-up evaluation", engine=dynamics)
        )
        print(f"Follow-up evaluation: {dynamics}", flush=True)
        env = StressPriceEnv(provider, engine, scenarios, cache)
        fixed = validation_baselines["old" if dynamics == "old" else "new"][
            "fixed_actions"
        ]
        baseline = baselines(
            env, folder / "baselines", output / "constant_cache", fixed
        )
        policies = {}
        for name, saved in selection["fits"].items():
            require(
                digest(saved["checkpoint"]) == saved["checkpoint_sha256"],
                "Policy changed after selection.",
            )
            result_path = folder / f"{name}.json"
            if result_path.exists():
                policies[name] = read_json(result_path)
            else:
                model = DQN.load(saved["checkpoint"], device=device)
                print(f"{dynamics}: {name}", flush=True)
                policies[name] = evaluate(env, model, result_path)
                del model
        old = [policies[f"old_seed{s}"] for s in (42, 7, 2026)]
        new = [policies[f"new_seed{s}"] for s in (42, 7, 2026)]
        comparisons[dynamics] = dict(
            new_minus_old=paired_intervals(new, old),
            new_minus_fixed=paired_intervals(new, [baseline["results"]["fixed"]]),
            new_minus_constant_oracle=paired_intervals(
                new, [baseline["results"]["constant_oracle"]]
            ),
        )
        comparisons[dynamics]["physical_differences"] = {
            metric: paired_intervals(new, old, metric=metric)
            for metric in (
                "overload_kwh",
                "overload_fraction",
                "household_mean_abs_energy_change",
                "charge_change_nok",
            )
        }
        all_results[dynamics] = dict(policies=policies, baselines=baseline["results"])
        if dynamics == "control":
            reference_cases = baseline["results"]["market"]["days"]
            for outcome in policies.values():
                for row, reference_row in zip(outcome["days"], reference_cases):
                    require(
                        (row["origin"], row["scenario"])
                        == (reference_row["origin"], reference_row["scenario"]),
                        "Control cases are not paired.",
                    )
                    np.testing.assert_array_equal(
                        row["regional_kw"], reference_row["regional_kw"]
                    )
                    require(
                        row["terms"]["grid"] == reference_row["terms"]["grid"]
                        and row["terms"]["energy"] == reference_row["terms"]["energy"],
                        "Prices changed price-free physical costs.",
                    )
            # Costs may differ through charges; physical paths must not change.
            origins = [0, len(provider.episodes) // 2, len(provider.episodes) - 1]
            for index in origins:
                paths = []
                for action in (0, len(env.grid) - 1):
                    rollout_episode(env, lambda e, a=action: a, index, 0)
                    paths.append(np.asarray(env.demands))
                np.testing.assert_array_equal(*paths)
            save_json(
                folder / "physical_invariance.json",
                dict(
                    verified=True,
                    origins=origins,
                    note="Only charge accounting can change under supplied prices.",
                ),
            )
    save_json(output / "comparison.json", comparisons)
    validation = EpisodeProvider(data, "validation")
    env = StressPriceEnv(validation, engines["new"], scenarios, cache)
    models = {
        name: DQN.load(selection["fits"][name]["checkpoint"], device=device)
        for name in ("old_seed42", "new_seed42")
    }
    value_audit(env, models["new_seed42"], output / "diagnostics/q_values.json")
    continuation_audit(data, env, models, output / "diagnostics/continuation.json")
    write_report(output, selection, comparisons, all_results)


def write_report(output, selection, comparisons, results):
    """Produce a short, evidence-qualified checkpoint report plus diagnostic plots."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Export scalar comparisons separately from the complete hourly JSON files.
    table = []
    for dynamics, outcomes in results.items():
        for name, result in {**outcomes["baselines"], **outcomes["policies"]}.items():
            table.append(
                dict(
                    dynamics=dynamics,
                    policy=name,
                    cost=result["mean_cost"],
                    **{f"cost_{k}": v for k, v in result["terms"].items()},
                    **result["stats"],
                )
            )
    with (Path(output) / "comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)

    rows = [
        (
            "| Dynamics | Policy | Mean cost | Overload region-hours | Energy"
            " change | Charge change NOK |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for dynamics, outcomes in results.items():
        for name, result in {**outcomes["baselines"], **outcomes["policies"]}.items():
            stats = result["stats"]
            rows.append(
                f"| {dynamics} | {name} | {result['mean_cost']:.6f} | "
                f"{stats['overload_region_hours']:.2f} | "
                f"{stats['energy_change_fraction']:+.2%} | "
                f"{stats['charge_change_nok']:+.2f} |"
            )
    primary = comparisons["new"]["new_minus_old"]["7"]
    sensitivity = comparisons["reserved"]["new_minus_old"]["7"]
    robust = primary["upper"] < 0 and sensitivity["upper"] < 0
    conclusion = (
        (
            "Upgraded-simulator training reduced common-simulator cost "
            "relative to old-simulator training, with a negative paired "
            "interval under both primary and reserved dynamics."
        )
        if robust
        else (
            "The comparison did not establish a reliable upgraded-training "
            "advantage under both primary and reserved dynamics. This "
            "negative or mixed result closes the predefined comparison."
        )
    )
    fits = read_json(output / "fit_summaries.json")
    lines = [
        "# Pricing checkpoint before exams",
        "",
        conclusion,
        "",
        (
            "- Six DQN fits; three seeds per frozen consumer family. Seed 42 "
            "is representative."
        ),
        (
            "- 717 households, 48 decisions, two separate constructed "
            "regional capacity pools."
        ),
        "- No allocation, forced floor, recovery, backlog or consumer retraining.",
        (
            "- Five-percent household energy/charge tolerances are "
            "assumptions, not measured welfare."
        ),
        (
            "- Fixed normalization prevents state weights from diluting "
            "household penalties."
        ),
        (
            "- Previously examined months/seeds: retrospective, not fresh "
            "holdout or causal validation."
        ),
        (
            "- Within each dynamics case, every policy faces the same "
            "reference and scoring rule."
        ),
        (
            "- No fixed-weight training ablation was run; adaptive-weight "
            "superiority is not established."
        ),
        "",
        "## Fits",
        "",
        (
            "| Family | Seed | Decisions | Updates | Best decisions | Seconds"
            " | Initial policy best? |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for fit in fits:
        lines.append(
            f"| {fit['family']} | {fit['seed']} | {fit['decisions']} | "
            f"{fit['updates']} | {fit['best_decisions']} | "
            f"{fit['elapsed_seconds']:.1f} | {fit['initial_untrained']} |"
        )
    lines += [
        "",
        "## Common-dynamics outcomes",
        "",
        *rows,
        "",
        "## Primary paired difference",
        "",
        "New-trained minus old-trained cost: "
        f"{comparisons['new']['new_minus_old']['mean_difference']:+.6f}; "
        f"seven-day 95% interval [{primary['lower']:+.6f}, "
        f"{primary['upper']:+.6f}].",
        "",
        (
            "Primary and reserved intervals must both exclude zero in the "
            "improving direction before claiming a robust cost improvement."
        ),
        (
            "Raw grid stress, energy and charge outcomes remain separate: "
            "better scalar cost alone does not prove every component "
            "improved."
        ),
        (
            "A switching benefit additionally requires beating the 48-hour "
            "constant-price oracle. This does not establish that hourly "
            "changes outperform daily changes."
        ),
        "",
        "## Limitations and files",
        "",
        (
            "- Counterfactual demand is a frozen observational-model "
            "prediction, not measured response."
        ),
        "- The paper's simulated backlog/minimum-consumption rules were not imported.",
        (
            "- Later survey descriptors are assumed stable; weather is a "
            "persisted regional reanalysis proxy."
        ),
        (
            "- Spot-energy charges exclude complete retail bills, supplier "
            "profit and measured consumer welfare."
        ),
        (
            "- No uncertainty penalty or monotonic price response was "
            "invented. Sparse-price occupancy is reported."
        ),
        (
            "- diagnostics/q_values.json compares Q values to same-policy "
            "model continuations, not optimal Q."
        ),
        (
            "- diagnostics/continuation.json appends a diagnostic 24 hours "
            "beyond consumer training; it is not observed rebound."
        ),
        (
            "- comparison.json contains paired 7/3/14-day intervals; "
            "evaluation/ contains all seeds, baselines and components."
        ),
        (
            "- Episode databases remain available in the Streamlit inspector."
            " No extra fits are scheduled."
        ),
    ]
    (Path(output) / "REPORT.md").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for family in ("old", "new"):
        for seed in (42, 7, 2026):
            history = read_json(Path(output) / f"{family}_seed{seed}/history.json")
            axes[0].plot(
                [h["seconds"] / 60 for h in history],
                [h["mean_cost"] for h in history],
                label=f"{family}/{seed}",
            )
            axes[1].plot(
                [h["decisions"] for h in history],
                [h["mean_cost"] for h in history],
                label=f"{family}/{seed}",
            )
    axes[0].set(xlabel="Consumed fit minutes", ylabel="Own-simulator validation cost")
    axes[1].set(xlabel="Decisions", ylabel="Own-simulator validation cost")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(output) / "learning_curves.png", dpi=160)
    plt.close(fig)
    outcomes = results["new"]
    middle = sorted({r["origin"] for r in outcomes["policies"]["new_seed42"]["days"]})
    origin = middle[len(middle) // 2]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for name in (
        "old_seed42",
        "new_seed42",
        "fixed",
        "market",
        "capacity_rule",
        "greedy",
    ):
        result = outcomes["policies"].get(name, outcomes["baselines"].get(name))
        row = next(
            r
            for r in result["days"]
            if r["origin"] == origin and r["scenario"] == "severe"
        )
        regional = np.asarray(row["regional_kw"])
        for region in range(2):
            axes[region].plot(np.arange(1, 49), regional[:, region], label=name)
        axes[2].step(np.arange(1, 49), row["prices"], where="post", label=name)
    settings = read_json(Path(output) / "settings.json")
    severe = next(s for s in settings["scenarios"] if s["name"] == "severe")
    for region in range(2):
        axes[region].axhline(
            severe["capacity_kw"][region],
            color="black",
            linestyle="--",
            label="constructed capacity",
        )
        axes[region].set(ylabel=f"{'NO1' if region==0 else 'NO5'} average kW")
    axes[0].legend(fontsize=7, ncol=3)
    axes[2].set(xlabel="Episode hour", ylabel="NOK/kWh")
    fig.suptitle(f"Common upgraded dynamics, severe scarcity, {origin[:10]}")
    fig.tight_layout()
    fig.savefig(Path(output) / "demand_and_prices.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    names = list(outcomes["policies"])
    bottom = np.zeros(len(names))
    for term in ("grid", "energy", "charge"):
        values = np.array(
            [outcomes["policies"][name]["terms"][term] / 7 for name in names]
        )
        ax.bar(names, values, bottom=bottom, label=term)
        bottom += values
    ax.set(
        ylabel="Normalized mean cost contribution",
        title="Actual objective components under common upgraded dynamics",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / "reward_components.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    outcomes = results["new"]
    combined = {**outcomes["baselines"], **outcomes["policies"]}
    ax.bar(list(combined), [r["mean_cost"] for r in combined.values()])
    ax.tick_params(axis="x", rotation=55)
    ax.set(
        ylabel="Common upgraded-simulator cost",
        title="Retrospective follow-up: baselines and all DQN seeds",
    )
    fig.tight_layout()
    fig.savefig(Path(output) / "policy_comparison.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 4))
    for name in ("old_seed42", "new_seed42", "fixed", "market"):
        result = combined[name]
        prices = np.asarray([day["prices"] for day in result["days"]]).ravel()
        grid = np.r_[0.05, np.arange(1, 21) / 10]
        frequency = np.array([np.isclose(prices, p).mean() for p in grid])
        ax.plot(grid, frequency, marker=".", label=name)
    ax.set(
        xlabel="Price NOK/kWh",
        ylabel="Fraction of decisions",
        title="Action occupancy under common upgraded dynamics",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(Path(output) / "action_occupancy.png", dpi=160)
    plt.close(fig)

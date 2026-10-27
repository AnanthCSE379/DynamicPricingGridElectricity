"""Measured replay, deterministic comparisons and paired retrospective inference."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from norway_energy_crisis.clean_data import require, save_json
from norway_energy_crisis.dqn.environment import append_history, atomic_npz, fingerprint
from norway_energy_crisis.dqn.evaluation import bootstrap, plotting
from .allocation import allocate, jain, service
from .environment import CapacityPriceEnv, arrival_order
from .policies import choose_action

SUMMARY_KEYS = (
    "loss",
    "fairness",
    "flexible_fairness",
    "forced_unserved_kwh",
    "essential_unserved_kwh",
    "requested_change_kwh",
    "requested_change_fraction",
    "requested_household_abs_change_kwh",
    "requested_kwh",
    "delivered_kwh",
    "reference_kwh",
    "charge_nok",
    "reference_charge_nok",
    "floor_binding_fraction",
    "preallocation_overload_region_hours",
    "postallocation_violation_region_hours",
    "essential_infeasible_region_hours",
    "essential_infeasible_hours",
    "flexible_eligible_households",
    "internal_switches",
    "mean_jump_nok_kwh",
    "sparse_price_fraction",
)


def read_json(path):
    return json.loads(Path(path).read_text())


def summarize(days):
    require(bool(days), "No eligible evaluated days.")
    means = {
        key: (
            float(np.mean([day[key] for day in days if day.get(key) is not None]))
            if any(day.get(key) is not None for day in days)
            else None
        )
        for key in SUMMARY_KEYS
    }
    regional = {}
    for area in range(2):
        key = f"region_{area}"
        if key in days[0]:
            regional[key] = {
                metric: (
                    float(
                        np.mean(
                            [
                                day[key][metric]
                                for day in days
                                if day[key][metric] is not None
                            ]
                        )
                    )
                    if any(day[key][metric] is not None for day in days)
                    else None
                )
                for metric in days[0][key]
            }
    by_scenario = {
        name: {
            metric: (
                float(
                    np.mean(
                        [
                            day[metric]
                            for day in days
                            if day["scenario"] == name and day.get(metric) is not None
                        ]
                    )
                )
                if any(
                    day["scenario"] == name and day.get(metric) is not None
                    for day in days
                )
                else None
            )
            for metric in SUMMARY_KEYS
        }
        for name in sorted({day["scenario"] for day in days})
    }
    prices = [day["prices"] for day in days if "prices" in day]
    grid = np.r_[0.05, np.arange(1, 21) / 10]
    counts = (
        {
            f"{price:.2f}": int(np.sum(abs(np.asarray(prices) - price) < 1e-8))
            for price in grid
        }
        if prices
        else {}
    )
    return dict(
        means=means,
        by_scenario=by_scenario,
        regional=regional,
        action_counts=counts,
        cases=len(days),
        days=days,
    )


def evaluate(env, policy, alpha=None, trace_loader=None, trace_recorder=None):
    """Enumerate every day/scenario once; no validation sampling with replacement."""
    days = []
    for index in range(len(env.provider.episodes)):
        for scenario_index in range(len(env.scenarios)):
            options = dict(episode_index=index, scenario_index=scenario_index)
            if alpha is not None:
                options["alpha"] = alpha
            observation, _ = env.reset(options=options)
            if trace_loader is not None:
                env.attach_trace(*trace_loader(env))
            rewards = 0.0
            for _ in range(24):
                action = choose_action(policy, env, observation)
                observation, reward, done, truncated, info = env.step(action)
                rewards += reward
            require(
                done and not truncated and abs(rewards + 100 * info["loss"]) < 1e-7,
                "Terminal reward must equal -100 times service loss.",
            )
            if trace_recorder is not None:
                trace_recorder(env)
            days.append(info)
    return summarize(days)


def prefer(candidate, incumbent, tolerance=1e-6):
    """Validation-only lexicographic ranking with an explicit numerical tolerance."""
    if incumbent is None:
        return True
    for key, direction in (
        ("loss", 1),
        ("fairness", -1),
        ("charge_nok", 1),
        ("mean_jump_nok_kwh", 1),
    ):
        left, right = candidate[key], incumbent[key]
        left, right = (0.0 if left is None else left), (0.0 if right is None else right)
        difference = direction * (left - right)
        if abs(difference) > tolerance:
            return difference < 0
    return False


class TrajectoryCache:
    """Cache model-generated fixed-price requests; capacity does not affect history."""

    def __init__(self, folder):
        self.folder = Path(folder)

    def constant(self, env, action):
        price = float(env.grid[action])
        identity = fingerprint(
            dict(
                engine=env.engine.signature,
                data=env.provider.manifest_hash,
                origin=env.episode.origin,
                alpha=env.scenario.alpha,
                price=price,
                recursion="protected_requests_no_allocation_feedback_v1",
            )
        )
        path = self.folder / f"{identity}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                return saved["prices"], saved["raw"]
        history = env.provider.features(env.episode)
        raw = []
        for hour in range(24):
            predicted = env.engine.predict(
                history, price, env.episode.future_calendar[hour]
            )
            raw.append(predicted)
            requested = np.maximum(predicted, env.scenario.alpha * env.reference[hour])
            history = append_history(
                history,
                requested,
                price,
                env.episode.future_calendar[hour],
                env.provider.preprocessor,
            )
        prices, raw = np.full(24, price), np.asarray(raw, dtype=np.float32)
        atomic_npz(path, prices=prices, raw=raw)
        return prices, raw


def cached_evaluate(
    env, policy, path, signature, alpha=None, trace_loader=None, trace_recorder=None
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        saved = read_json(path)
        require(
            saved["signature"] == signature,
            "Evaluation cache inputs changed; choose new output.",
        )
        return saved["result"]
    result = evaluate(env, policy, alpha, trace_loader, trace_recorder)
    save_json(path, dict(signature=signature, result=result))
    return result


def baselines(
    provider,
    engine,
    scenarios,
    cache,
    output,
    support,
    signature,
    fixed_action=None,
    alpha=0.5,
    allocation_comparison=True,
):
    """Enumerate the constant grid; fixed price is chosen on validation only."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = CapacityPriceEnv(provider, engine, scenarios, cache, support=support)
    trajectories = TrajectoryCache(Path(cache) / "constant_requests")
    identity = dict(
        protocol=signature,
        engine=engine.signature,
        split=provider.split,
        origins=[episode.origin for episode in provider.episodes],
        alpha=alpha,
    )
    constants = []
    for action in range(len(env.grid)):
        print(
            f"Constant-price evaluation {engine.label}/{provider.split}: "
            f"{action+1}/{len(env.grid)}",
            flush=True,
        )
        result = cached_evaluate(
            env,
            lambda current, observation, index=action: index,
            output / f"constant_{action:02}.json",
            identity,
            alpha,
            lambda current, index=action: trajectories.constant(current, index),
        )
        constants.append(result)
    if fixed_action is None:
        require(
            provider.split == "validation", "Select fixed pricing on validation only."
        )
        fixed_action = 0
        for action, result in enumerate(constants):
            if prefer(result["means"], constants[fixed_action]["means"]):
                fixed_action = action
    require(0 <= fixed_action < len(constants), "Invalid frozen fixed-price index.")
    oracle = [
        min(
            [result["days"][index] for result in constants],
            key=lambda day: (
                day["loss"],
                -(day["fairness"] or 0.0),
                day["charge_nok"],
                day["mean_jump_nok_kwh"],
            ),
        )
        for index in range(len(constants[0]["days"]))
    ]
    outcomes = dict(
        fixed=constants[fixed_action],
        minimum=constants[0],
        maximum=constants[-1],
        daily_constant_oracle=summarize(oracle),
    )
    for name, policy in (
        ("market", lambda current, observation: current.market_action()),
        ("rule", lambda current, observation: current.rule_action()),
    ):
        outcomes[name] = cached_evaluate(
            env,
            policy,
            output / f"{name}.json",
            identity,
            alpha,
            trace_recorder=(
                (lambda current: record_rule_trace(current, cache))
                if name == "rule"
                else None
            ),
        )
    if allocation_comparison:
        # Replay the exact protected requests and prices generated by the rule.
        # There is no allocation-to-request feedback, so no additional model calls
        # are needed to isolate how the allocator distributes the same shortage.
        outcomes["rule_proportional"] = reallocate(
            outcomes["rule"],
            provider,
            engine,
            scenarios,
            cache,
            "proportional",
            support,
            alpha,
        )
        orders = [
            reallocate(
                outcomes["rule"],
                provider,
                engine,
                scenarios,
                cache,
                "fcfs",
                support,
                alpha,
                seed,
            )
            for seed in (0, 1, 2)
        ]
        outcomes["rule_fcfs"] = average_results(orders)
    save_json(
        output / "baselines.json",
        dict(fixed_action=int(fixed_action), results=outcomes),
    )
    return outcomes, int(fixed_action)


def rule_trace_path(env, cache, prices):
    identity = fingerprint(
        dict(
            engine=env.engine.signature,
            origin=env.episode.origin,
            data=env.provider.manifest_hash,
            scenario=env.scenario.signature,
            alpha=env.scenario.alpha,
            prices=prices,
            recursion="protected_requests_no_allocation_feedback_v1",
        )
    )
    return Path(cache) / "rule_requests" / f"{identity}.npz"


def record_rule_trace(env, cache):
    atomic_npz(
        rule_trace_path(env, cache, env.prices),
        raw=np.asarray(env.raw_predictions, dtype=np.float32),
    )


def reallocate(
    result, provider, engine, scenarios, cache, method, support, alpha, order_seed=0
):
    """Reuse the original rule's exact numerical requests for every allocator."""
    env = CapacityPriceEnv(
        provider, engine, scenarios, cache, method, support, order_seed
    )
    days = []
    for index, day in enumerate(result["days"]):
        episode_index, scenario_index = divmod(index, len(scenarios))
        observation, _ = env.reset(
            options=dict(
                episode_index=episode_index, scenario_index=scenario_index, alpha=alpha
            )
        )
        path = rule_trace_path(env, cache, day["prices"])
        require(
            path.exists(), "Original rule trace is missing; recompute its evaluation."
        )
        with np.load(path, allow_pickle=False) as saved:
            predictions = saved["raw"]
        env.attach_trace(day["prices"], predictions)
        for price in day["prices"]:
            _, _, _, _, info = env.step(int(np.argmin(abs(env.grid - price))))
        require(
            np.array_equal(
                [row["requested"] for row in info["hourly"]],
                [row["requested"] for row in day["hourly"]],
            ),
            "An allocation comparison changed the original requests.",
        )
        days.append(info)
    return summarize(days)


def average_results(results):
    """Average numerical outcomes inside each paired day, never concatenate seeds."""
    require(bool(results), "No results to average.")
    days = []
    for index, first in enumerate(results[0]["days"]):
        rows = [result["days"][index] for result in results]
        require(
            all(
                (row["origin"], row["scenario"], row["alpha"])
                == (first["origin"], first["scenario"], first["alpha"])
                for row in rows
            ),
            "Unpaired days.",
        )
        day = dict(first)
        # Curves and schedules belong to the first replicate; only the numeric
        # endpoints are averaged. Preserve that distinction in machine outputs.
        day["trace_interpretation"] = (
            "First replicate trace; scalar metrics averaged within day."
        )
        for key in SUMMARY_KEYS:
            values = [row.get(key) for row in rows if row.get(key) is not None]
            if values:
                day[key] = float(np.mean(values))
        for area in range(2):
            key = f"region_{area}"
            if key in first:
                day[key] = {
                    metric: (
                        float(
                            np.mean(
                                [
                                    row[key][metric]
                                    for row in rows
                                    if row[key][metric] is not None
                                ]
                            )
                        )
                        if any(row[key][metric] is not None for row in rows)
                        else None
                    )
                    for metric in first[key]
                }
        days.append(day)
    result = summarize(days)
    if results[0].get("action_counts"):
        result["action_counts"] = {
            key: float(np.mean([result["action_counts"][key] for result in results]))
            for key in results[0]["action_counts"]
        }
    result["replicates"] = len(results)
    return result


class HistoricalReplay:
    """Private measured-data cursor; only the current hour reaches the allocator."""

    def __init__(self, data, origin):
        self._data, self._origin, self._hour = data, int(origin), 0

    def next_hour(self):
        require(self._hour < 24, "Historical day is complete.")
        index = self._origin + self._hour
        self._hour += 1
        return (
            np.array(self._data.demand[:, index], dtype=np.float64, copy=True),
            np.array(self._data.prices[:, index], dtype=np.float64, copy=True),
        )


def measured_replay(data, provider, scenarios, alpha=0.5):
    outcomes = {}
    masks = [data.area_index == area for area in range(2)]
    for method in ("fair", "proportional", "fcfs"):
        orders = (0, 1, 2) if method == "fcfs" else (0,)
        replicated = []
        for order_seed in orders:
            days = []
            for episode in provider.episodes:
                for base in scenarios:
                    scenario = replace(base, alpha=alpha)
                    cursor = HistoricalReplay(data, episode.origin)
                    requested = np.zeros(data.households)
                    delivered = requested.copy()
                    prior_flexible = requested.copy()
                    served_flexible = requested.copy()
                    essential_shortage = requested.copy()
                    charge = 0.0
                    reference_charge = 0.0
                    hourly = []
                    for hour in range(24):
                        demand, prices = cursor.next_hour()
                        essential = alpha * demand
                        allocated = np.zeros_like(demand)
                        infeasible = []
                        for area, mask in enumerate(masks):
                            result = allocate(
                                demand[mask],
                                essential[mask],
                                scenario.budget_kwh[area],
                                prior_flexible[mask],
                                served_flexible[mask],
                                method,
                                arrival_order(
                                    int(mask.sum()),
                                    episode.origin,
                                    hour,
                                    area,
                                    order_seed,
                                ),
                            )
                            allocated[mask] = result.delivered
                            essential_shortage[mask] += result.essential_unserved
                            served_flexible[mask] += result.flexible_delivered
                            infeasible.append(result.essential_infeasible)
                        prior_flexible += demand - essential
                        requested += demand
                        delivered += allocated
                        charge += float((allocated * prices[data.area_index]).sum())
                        reference_charge += float(
                            (demand * prices[data.area_index]).sum()
                        )
                        hourly.append(
                            dict(
                                requested=[float(demand[mask].sum()) for mask in masks],
                                delivered=[
                                    float(allocated[mask].sum()) for mask in masks
                                ],
                                essential_infeasible=infeasible,
                            )
                        )
                    metrics = service(requested, delivered)
                    metrics.pop("service_ratios")
                    active = prior_flexible > 1e-6
                    for area, mask in enumerate(masks):
                        regional = service(requested[mask], delivered[mask])
                        regional.pop("service_ratios")
                        metrics[f"region_{area}"] = regional
                    metrics.update(
                        forced_unserved_kwh=float((requested - delivered).sum()),
                        essential_unserved_kwh=float(essential_shortage.sum()),
                        requested_change_kwh=0.0,
                        requested_change_fraction=0.0,
                        flexible_eligible_households=int(active.sum()),
                        requested_household_abs_change_kwh=0.0,
                        reference_kwh=float(requested.sum()),
                        requested_kwh=float(requested.sum()),
                        delivered_kwh=float(delivered.sum()),
                        flexible_fairness=jain(
                            served_flexible[active] / prior_flexible[active]
                        ),
                        charge_nok=charge,
                        reference_charge_nok=reference_charge,
                        preallocation_overload_region_hours=int(
                            (
                                np.array([row["requested"] for row in hourly])
                                > scenario.budget_kwh + 1e-8
                            ).sum()
                        ),
                        postallocation_violation_region_hours=int(
                            (
                                np.array([row["delivered"] for row in hourly])
                                > scenario.budget_kwh + 1e-8
                            ).sum()
                        ),
                        essential_infeasible_region_hours=int(
                            np.asarray(
                                [row["essential_infeasible"] for row in hourly]
                            ).sum()
                        ),
                        essential_infeasible_hours=int(
                            np.any(
                                [row["essential_infeasible"] for row in hourly], axis=1
                            ).sum()
                        ),
                    )
                    days.append(
                        dict(
                            mode="historical_replay",
                            origin=episode.timestamp,
                            scenario=scenario.name,
                            alpha=alpha,
                            method=method,
                            hourly=hourly,
                            **metrics,
                        )
                    )
            replicated.append(summarize(days))
        outcomes[method] = average_results(replicated)
    return outcomes


def paired_daily(candidate, reference, metric="loss"):
    def grouped(result):
        groups = {}
        for day in result["days"]:
            if day.get(metric) is not None:
                groups.setdefault(day["origin"], {})[day["scenario"]] = day[metric]
        return groups

    left, right = grouped(candidate), grouped(reference)
    require(left.keys() == right.keys(), "Unpaired calendar days.")
    origins = sorted(left)
    for origin in origins:
        require(
            left[origin].keys() == right[origin].keys(), "Unpaired capacity scenarios."
        )
    differences = [
        np.mean([left[origin][key] - right[origin][key] for key in left[origin]])
        for origin in origins
    ]
    return {
        str(block): bootstrap(differences, np.zeros(len(differences)), block=block)
        for block in (3, 7, 14)
    }


def regional_result(result, area):
    """Retain calendar/scenario keys when estimating a regional endpoint."""
    return dict(
        days=[
            dict(
                origin=day["origin"], scenario=day["scenario"], **day[f"region_{area}"]
            )
            for day in result["days"]
        ]
    )


def allocation_intervals(outcomes, reference_name="fair"):
    reference = outcomes[reference_name]
    return {
        name: {
            metric: paired_daily(result, reference, metric)
            for metric in ("loss", "fairness", "flexible_fairness")
        }
        for name, result in outcomes.items()
        if name != reference_name
    }


def make_report(
    output, selection, primary, measured, sensitivities, durations, evidence_manifest
):
    output = Path(output)
    comparisons = {}
    claims = {}
    for configuration in primary["ensemble"]["policies"]:
        comparisons[configuration] = {}
        service_pass, fairness_pass, switching_pass = [], [], []
        for name in ("ensemble", "reserved", "control"):
            scenario = primary[name]
            candidate = average_results(scenario["policies"][configuration])
            baseline = scenario["baselines"][selection["baseline"]]
            service_interval = paired_daily(candidate, baseline)
            fairness_interval = paired_daily(candidate, baseline, "fairness")
            oracle_interval = paired_daily(
                candidate, scenario["baselines"]["daily_constant_oracle"]
            )
            comparisons[configuration][name] = dict(
                service_vs_baseline=service_interval,
                fairness_vs_baseline=fairness_interval,
                service_vs_daily_constant=oracle_interval,
                all_price_comparators={
                    controller: {
                        metric: paired_daily(candidate, result, metric)
                        for metric in ("loss", "fairness", "flexible_fairness")
                    }
                    for controller, result in scenario["baselines"].items()
                    if controller
                    in ("fixed", "market", "rule", "daily_constant_oracle")
                },
                regional={
                    str(area): {
                        metric: paired_daily(
                            regional_result(candidate, area),
                            regional_result(baseline, area),
                            metric,
                        )
                        for metric in ("loss", "fairness")
                    }
                    for area in range(2)
                },
                seed_mean_loss=float(
                    np.mean(
                        [
                            run["means"]["loss"]
                            for run in scenario["policies"][configuration]
                        ]
                    )
                ),
                seed_sd_loss=float(
                    np.std(
                        [
                            run["means"]["loss"]
                            for run in scenario["policies"][configuration]
                        ]
                    )
                ),
            )
            if name in ("ensemble", "reserved"):
                service_pass.append(service_interval["7"]["ci95"][1] < 0.0)
                fairness_pass.append(fairness_interval["7"]["ci95"][0] > 0.0)
                switching_pass.append(
                    oracle_interval["7"]["ci95"][1] < 0.0
                    and candidate["means"]["internal_switches"] > 0
                )
        claims[configuration] = dict(
            pricing_service_improvement=all(service_pass),
            joint_service_fairness_improvement=all(service_pass) and all(fairness_pass),
            switching_benefit=all(switching_pass),
        )
    result = dict(
        selection=selection,
        comparisons=comparisons,
        claims=claims,
        allocation_comparisons={
            name: allocation_intervals(
                {
                    "fair": case["baselines"]["rule"],
                    "proportional": case["baselines"]["rule_proportional"],
                    "fcfs": case["baselines"]["rule_fcfs"],
                }
            )
            for name, case in primary.items()
        },
        historical_allocation_comparisons={
            alpha: allocation_intervals(case) for alpha, case in measured.items()
        },
        sensitivity_comparisons={
            name: {
                controller: {
                    metric: paired_daily(
                        average_results(case["policies"][selection["configuration"]]),
                        baseline,
                        metric,
                    )
                    for metric in ("loss", "fairness")
                }
                for controller, baseline in case["baselines"].items()
            }
            for name, case in sensitivities.items()
        },
        durations=durations,
        evidence=evidence_manifest,
        interpretation=(
            "Retrospective simulated comparisons; no causal or fresh-holdout claim."
        ),
    )
    save_json(output / "comparison.json", result)
    save_json(
        output / "test_metrics.json",
        dict(primary=primary, historical_replay=measured, sensitivities=sensitivities),
    )
    lines = [
        "# Capacity and allocation comparison",
        "",
        (
            "Retrospective evidence-led stress tests. Capacities and essential"
            " fractions are constructed assumptions."
        ),
        (
            f"Selected family: {selection['configuration']}; representative policy"
            " seed: 42."
        ),
        "",
        (
            "| Response dynamics | Controller | Mean service loss | Reference fairness"
            " | Forced unserved kWh/day |"
        ),
        "|---|---|---:|---:|---:|",
    ]
    for name, scenario in primary.items():
        controllers = {
            **{
                key: average_results(runs) for key, runs in scenario["policies"].items()
            },
            **scenario["baselines"],
        }
        for controller, outcome in controllers.items():
            values = outcome["means"]
            fairness = (
                "n/a" if values["fairness"] is None else f"{values['fairness']:.6f}"
            )
            lines.append(
                f"| {name} | {controller} | {values['loss']:.6f} | {fairness} |"
                f" {values['forced_unserved_kwh']:.3f} |"
            )
    lines += [
        "",
        "## Claim checks",
        "",
        "```json",
        json.dumps(claims, indent=2),
        "```",
        "",
        (
            "Intervals use 5,000 paired moving-block resamples: seven-day primary,"
            " three/fourteen-day sensitivity."
        ),
        (
            "Policy/order seeds are averaged within each day. Response seeds are"
            " sensitivity cases, not independent observations."
        ),
        (
            "Historical replay uses measured requests and makes no alternative-price"
            " claim. Full replay, alpha and response sensitivities are in"
            " test_metrics.json."
        ),
        "",
        "## Evidence and limitations",
        "",
    ] + [f"- {item}" for item in evidence_manifest["limitations"]]
    selected_claims = claims[selection["configuration"]]
    lines += ["", "## Predeclared outcome", ""]
    if selected_claims["pricing_service_improvement"]:
        lines.append(
            "The selected family's service-loss improvement criterion was met "
            "under both ensemble and reserved response dynamics. This is a "
            "modeled retrospective comparison, not a causal pricing result."
        )
    else:
        lines.append(
            "The selected family did not meet the predeclared pricing-improvement"
            " criterion across ensemble and reserved dynamics. This negative result"
            " closes the planned comparison; further training is not automatically"
            " added."
        )
    if not selected_claims["joint_service_fairness_improvement"]:
        lines.append(
            "Joint service and fairness improvement is not established. "
            "Inspect the paired endpoints for equality or trade-offs."
        )
    if not selected_claims["switching_benefit"]:
        lines.append(
            "A switching benefit over the daily constant-price comparator "
            "is not established under both response scenarios."
        )
    lines.append(
        "The selected family is the predeclared primary comparison; "
        "other family and pairwise intervals are descriptive follow-up comparisons."
    )
    lines += [
        "",
        "## Paired primary intervals",
        "",
        (
            "Differences are policy minus the validation-selected"
            f" {selection['baseline']} baseline."
        ),
        (
            "Negative service-loss differences favor the policy; positive fairness"
            " differences favor it."
        ),
        "",
        (
            "| Family | Dynamics | Mean loss difference | 95% interval | Fairness"
            " interval |"
        ),
        "|---|---|---:|---|---|",
    ]
    for family, cases in comparisons.items():
        for name, case in cases.items():
            loss = case["service_vs_baseline"]["7"]
            fairness = case["fairness_vs_baseline"]["7"]
            lines.append(
                f"| {family} | {name} | {loss['mean_cost_difference']:.6f} | "
                f"{loss['ci95']} | {fairness['ci95']} |"
            )
    lines += [
        "",
        "## Protocol and execution",
        "",
        (
            "The primary endpoint is the household mean capped reference-service"
            " deficit. Reward is zero until the terminal -100 L. Capacities are fixed"
            " regional training-demand quantiles. The essential fraction is 0.50;"
            " 0.25/0.75 are sensitivity assumptions. Protected requests, not"
            " deliveries, advance the response model."
        ),
        "",
        (
            "No consumer response to outages is simulated. FCFS uses three synthetic"
            " order seeds averaged within day. The daily oracle enumerates only"
            " constant prices for each day and capacity scenario, not arbitrary"
            " schedules."
        ),
        "",
        "```json",
        json.dumps(durations, indent=2),
        "```",
        "",
        (
            "Nine fit summaries, parameter counts and validation checkpoints are in"
            " fit_summaries.json. Regional/scenario breakdowns, action counts, seed"
            " results, alpha sensitivities and three/seven/fourteen-day intervals are"
            " retained in the JSON outputs."
        ),
    ]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    plt = plotting(output)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels, losses, fairness = [], [], []
    for name, runs in primary["ensemble"]["policies"].items():
        means = average_results(runs)["means"]
        labels.append(name)
        losses.append(means["loss"])
        fairness.append(means["fairness"] or 0.0)
    for name in ("fixed", "market", "rule", "daily_constant_oracle"):
        means = primary["ensemble"]["baselines"][name]["means"]
        labels.append(name)
        losses.append(means["loss"])
        fairness.append(means["fairness"] or 0.0)
    for ax, values, label in zip(
        axes,
        (losses, fairness),
        ("Reference service loss", "Reference service fairness"),
    ):
        ax.bar(labels, values)
        ax.set_ylabel(label)
        ax.tick_params(axis="x", labelrotation=35)
    fig.tight_layout()
    fig.savefig(output / "service_and_fairness.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for method, style in (("fair", "-"), ("proportional", "--"), ("fcfs", ":")):
        names = sorted(measured)
        for ax, key in zip(axes, ("loss", "fairness")):
            ax.plot(
                [float(alpha) for alpha in names],
                [measured[alpha][method]["means"][key] for alpha in names],
                style,
                marker="o",
                label=method,
            )
            ax.set_xlabel("Assumed essential fraction")
            ax.set_ylabel(f"Measured replay: {key}")
            ax.legend()
    fig.tight_layout()
    fig.savefig(output / "historical_allocation.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for family, runs in primary["ensemble"]["policies"].items():
        counts = average_results(runs)["action_counts"]
        axes[0].plot(
            [float(price) for price in counts], list(counts.values()), label=family
        )
    axes[0].set_xlabel("NOK/kWh")
    axes[0].set_ylabel("Hourly decisions (mean across policy seeds)")
    axes[0].legend(fontsize=7)
    labels, request_change, unserved = [], [], []
    for family, runs in primary["ensemble"]["policies"].items():
        means = average_results(runs)["means"]
        labels.append(family)
        request_change.append(means["requested_change_kwh"])
        unserved.append(means["forced_unserved_kwh"])
    positions = np.arange(len(labels))
    axes[1].bar(
        positions - 0.18, request_change, 0.36, label="Request change vs reference"
    )
    axes[1].bar(positions + 0.18, unserved, 0.36, label="Forced unserved")
    axes[1].set_xticks(positions, labels, rotation=20)
    axes[1].set_ylabel("kWh per day, mean across scenarios and seeds")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "pricing_and_service.png", dpi=160)
    plt.close(fig)
    representative = primary["ensemble"]["policies"][selection["configuration"]][0][
        "days"
    ]
    day = representative[len(representative) // 2]
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    axes[0].step(np.arange(24), day["prices"], where="post")
    axes[0].set_ylabel("NOK/kWh")
    axes[0].set_title(f"Seed 42; {day['origin']}; {day['scenario']} capacity")
    for area in range(2):
        for key in ("reference", "requested", "delivered"):
            axes[area + 1].plot([row[key][area] for row in day["hourly"]], label=key)
        axes[area + 1].axhline(
            day["capacity_kw"][area],
            linestyle="--",
            color="black",
            label="constructed capacity",
        )
        axes[area + 1].set_ylabel(f"{evidence_manifest['areas'][area]} kWh/hour")
        axes[area + 1].legend(fontsize=7)
    axes[-1].set_xlabel("Dataset local hour")
    fig.tight_layout()
    fig.savefig(output / "example_day.png", dpi=160)
    plt.close(fig)
    return result

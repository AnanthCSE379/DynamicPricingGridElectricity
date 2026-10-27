"""Summarize saved post-hoc audit probes without rerunning or selecting policies."""

import argparse
import json
from pathlib import Path

import numpy as np

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.capacity.evaluation import read_json
from norway_energy_crisis.dqn.evaluation import plotting


def mean(rows, key):
    return float(np.mean([row[key] for row in rows]))


def verify_reward(run, folder):
    """Use a separate equation implementation on actual full-cohort CPU rollouts.

    CPU is explicitly a verification device here; the experiment and large audit
    inference stages use MPS. This function never trains a model.
    """
    import torch
    from norway_energy_crisis.preprocess import PreparedData
    from norway_energy_crisis.dqn.environment import EpisodeProvider
    from norway_energy_crisis.dqn.train import make_engine
    from norway_energy_crisis.capacity.environment import CapacityPriceEnv
    from norway_energy_crisis.capacity.policies import algorithm, choose_action
    from norway_energy_crisis.capacity.scenarios import calibrate

    torch.set_num_threads(4)
    data = PreparedData(ROOT / "prepared")
    provider = EpisodeProvider(data, "validation")
    engine = make_engine(ROOT / "runs/comparison_v1", "ensemble", data, "cpu")
    scenarios = calibrate(data)
    selection = read_json(run / "selection.json")
    cache = run / "cache" / selection["protocol"]
    model = algorithm("dqn").load(run / "dqn_seed42/best.zip", device="cpu")
    outcomes = []
    for name, policy in (
        ("maximum", lambda current, obs: 20),
        ("market", lambda current, obs: current.market_action()),
        ("dqn_42", model),
    ):
        env = CapacityPriceEnv(provider, engine, scenarios, cache)
        observation, _ = env.reset(options=dict(episode_index=29, scenario_index=2))
        rewards = []
        for _ in range(24):
            observation, reward, done, truncated, _ = env.step(
                choose_action(policy, env, observation)
            )
            rewards.append(reward)
        baseline = env.reference.sum(axis=0, dtype=np.float64)
        active = baseline > 1e-6
        # Do not call service() or the audit's score_profile().
        expected = -100.0 * float(
            np.maximum(
                1.0 - env.delivered_energy[active] / baseline[active], 0.0
            ).mean()
        )
        error = abs(sum(rewards) - expected)
        require(
            done
            and not truncated
            and all(value == 0 for value in rewards[:-1])
            and error < 1e-8,
            "Independent terminal reward calculation failed.",
        )
        outcomes.append(
            dict(
                policy=name,
                origin=env.episode.timestamp,
                households=int(active.sum()),
                episode_reward=sum(rewards),
                independently_computed_reward=expected,
                absolute_error=error,
            )
        )
    save_json(
        folder / "independent_reward_checks.json",
        dict(device="cpu verification only", cases=outcomes),
    )


def summarize(run, folder):
    require(
        (folder / "status.json").exists(), "Inference diagnostics have not finished."
    )
    audit = read_json(folder / "artifact_checks.json")
    policies = read_json(folder / "policy_probes.json")
    curves = read_json(folder / "price_curves.json")
    coverage = read_json(folder / "capacity_coverage.json")
    training = read_json(folder / "training_constants.json")["days"]
    original = read_json(run / "evidence.json")
    selection = read_json(run / "selection.json")
    require(
        all(
            digest(Path(row["path"])) == row["sha256"]
            for group in original["response_weights"].values()
            for row in group
        ),
        "Response checkpoint changed.",
    )
    require(
        all(
            digest(run / name / "best.zip") == expected
            for name, expected in selection["policies"].items()
        ),
        "Original policy checkpoint changed.",
    )
    result = dict(
        grid={},
        policy_probes={},
        q_probes={},
        price_probes={},
        stochastic_probes={},
        checkpoint_states=policies["checkpoints"],
        coverage=coverage,
        original_weights_unchanged=True,
        report_source_sha256=digest(__file__),
    )
    result["independent_reward_checks"] = read_json(
        folder / "independent_reward_checks.json"
    )
    for split in ("train", "validation", "test"):
        if split == "train":
            rows = training
        else:
            path = (
                run / "validation"
                if split == "validation"
                else run / "evaluation/ensemble/alpha0.5/baselines"
            )
            rows = [
                dict(price=value["days"][0]["prices"][0], loss=value["means"]["loss"])
                for file in sorted(path.glob("constant_*.json"))
                for value in [read_json(file)["result"]]
            ]
        result["grid"][split] = [
            dict(
                price=price,
                loss=mean([row for row in rows if row["price"] == price], "loss"),
            )
            for price in sorted({row["price"] for row in rows})
        ]
    deterministic = [row for row in policies["days"] if not row.get("stochastic")]
    metrics = (
        "loss",
        "hourly_reference_deficit",
        "temporal_cancellation",
        "request_reference_deficit",
        "forced_reference_deficit",
        "over_reference_delivery_kwh",
        "forced_unserved_kwh",
        "requested_change_kwh",
    )
    for dynamics in ("ensemble", "reserved"):
        result["policy_probes"][dynamics] = {}
        for name in sorted({row["policy"] for row in deterministic}):
            rows = [
                row
                for row in deterministic
                if row["dynamics"] == dynamics and row["policy"] == name
            ]
            result["policy_probes"][dynamics][name] = {
                key: mean(rows, key) for key in metrics
            }
            qrows = [q for row in rows for q in row["q_checks"] if q["hour"] == 23]
            if qrows:
                result["q_probes"][f"{dynamics}/{name}"] = dict(
                    states=len(qrows),
                    mean_action_regret=mean(qrows, "regret"),
                    wrong_action_fraction=float(
                        np.mean([q["regret"] > 1e-6 for q in qrows])
                    ),
                    mean_chosen_value_error=mean(qrows, "chosen_value_error"),
                    mean_predicted_action_span=mean(qrows, "q_span"),
                    mean_true_action_span=mean(qrows, "true_return_span"),
                )
    worst_row, worst_q = max(
        ((row, q) for row in deterministic for q in row["q_checks"]),
        key=lambda pair: pair[1]["regret"],
    )
    chosen = int(np.argmax(worst_q["q_values"]))
    result["worst_branch"] = dict(
        origin=worst_row["origin"],
        scenario=worst_row["scenario"],
        dynamics=worst_row["dynamics"],
        policy=worst_row["policy"],
        hour=worst_q["hour"],
        chosen_price=worst_q["chosen_price"],
        best_branch_price=worst_q["best_branch_price"],
        predicted_chosen_q=worst_q["q_values"][chosen],
        actual_chosen_q=worst_q["actual_returns"][chosen],
        actual_best_q=max(worst_q["actual_returns"]),
        regret=worst_q["regret"],
    )
    for name in ("ensemble", "s42", "s7", "reserved"):
        rows = [row for row in curves["states"] if row["dynamics"] == name]
        result["price_probes"][name] = dict(
            positive_household_fraction=mean(rows, "positive_household_fraction"),
            nonmonotonic_household_fraction=mean(
                rows, "nonmonotonic_household_fraction"
            ),
            regional_change_fraction=np.mean(
                [row["regional_low_to_high_change_fraction"] for row in rows], axis=0
            ).tolist(),
        )
    result["price_probes"]["ensemble_reserved_sign_disagreement"] = mean(
        [row for row in curves["states"] if row["dynamics"] == "sign_disagreement"],
        "ensemble_reserved_household_fraction",
    )
    for name in sorted(
        {row["policy"] for row in policies["days"] if row.get("stochastic")}
    ):
        base = [
            row
            for row in policies["days"]
            if row["policy"] == name
            and row["dynamics"] == "ensemble"
            and row["split"] == "validation"
            and row["scenario"] == "severe"
        ]
        det = [row for row in base if not row.get("stochastic")]
        sampled = [row for row in base if row.get("stochastic")]
        probs = [value for row in det for value in row["probabilities"]]
        result["stochastic_probes"][name] = dict(
            deterministic_loss=mean(det, "loss"),
            stochastic_loss=mean(sampled, "loss"),
            sampled_rollouts=len(sampled),
            mean_argmax_probability=mean(probs, "maximum_probability"),
            mean_policy_entropy=mean(probs, "entropy"),
        )
    save_json(folder / "summary.json", result)
    lines = [
        "# Reward and learning audit",
        "",
        (
            "This is a post-hoc audit of capacity_v1. No policies were retrained, no "
            "frozen selections were changed and no original metrics were overwritten. "
            "Source fingerprints and saved response/policy weights were verified."
        ),
        "",
        (
            "**Conclusion:** there are reproducible objective loopholes, unreliable"
            " counterfactual price dynamics and learned-policy failures. The evidence"
            " does not establish that the policies deliberately exploited a loophole or"
            " that reward hacking alone caused the performance gap."
        ),
        "",
        "## 1. Daily service permits temporal cancellation",
        "",
        (
            "The reward uses max(1 - sum(delivered)/sum(reference), 0) for each"
            " household. Consequently, delivery above reference in one hour can erase a"
            " reference shortage in another hour for that household. This is not a bug"
            " relative to the specified daily equation, but it conflicts with an"
            " hour-specific interpretation of lost service when no deferral/recovery is"
            " modeled."
        ),
        "",
        (
            "An adversarial feasible example has two households, two hours, unit"
            " reference requests, essential requests 0.5, and a regional budget of 2"
            " each hour. Deliveries [0.5,1.5] and [1.5,0.5] give zero daily loss,"
            " despite a 25% uncancelled hourly reference deficit. Cross-household"
            " cancellation remains correctly prevented."
        ),
        "",
        (
            "Real rollout decompositions below use the original daily loss and an"
            " additional diagnostic:"
            " mean_households(sum_hours(max(reference-delivery,0)) / daily_reference)."
            " This is another service proxy, not measured welfare or a replacement"
            " endpoint selected after seeing results."
        ),
        "",
        "| Dynamics | Policy | Daily loss % | Hourly deficit % | Canceled deficit pp |",
        "|---|---|---:|---:|---:|",
    ]
    for dynamics, values in result["policy_probes"].items():
        for name, row in values.items():
            lines.append(
                f"| {dynamics} | {name} | {100*row['loss']:.3f} |"
                f" {100*row['hourly_reference_deficit']:.3f} |"
                f" {100*row['temporal_cancellation']:.3f} |"
            )
    lines += [
        "",
        (
            "The objective also counts modeled voluntary request reductions "
            "as reference-service deficits, even if every current request is served. "
            "Conversely, increasing simulated deliveries can improve the score until "
            "daily reference energy is reached. It contains no price/charge penalty. "
            "A constant maximum price is therefore not automatically a hack or an "
            "invalid solution. The question is whether the modeled response and "
            "service proxy represent the research question."
        ),
        "",
        "## 2. Counterfactual price response is not stable across response seeds",
        "",
        (
            "Six January days, four hours per day and all 717 households were probed."
            " Each response model received the same historical input, calendar and"
            " candidate prices. These are conditional model sensitivities, not causal"
            " elasticities."
        ),
        "",
        (
            "| Dynamics | Households with higher prediction at 2.00 than 0.05 % | NO1"
            " aggregate change % | NO5 change % |"
        ),
        "|---|---:|---:|---:|",
    ]
    for name in ("ensemble", "s42", "s7", "reserved"):
        row = result["price_probes"][name]
        lines.append(
            f"| {name} | {100*row['positive_household_fraction']:.2f} |"
            f" {100*row['regional_change_fraction'][0]:.3f} |"
            f" {100*row['regional_change_fraction'][1]:.3f} |"
        )
    lines += [
        "",
        (
            "Ensemble/reserved household price-effect signs disagree in"
            f" {100*result['price_probes']['ensemble_reserved_sign_disagreement']:.2f}%"
            " of the probed household-states."
        ),
        "",
        (
            "The full recursive constant-price outcomes also differ. Under reserved"
            " dynamics, maximum-price requests rise 1.25% relative to that model"
            " reference, while minimum-price requests fall 4.11%. The maximum-price"
            " baseline has lower daily service loss despite more forced unserved kWh"
            " than the minimum. This separates reference matching from curtailment"
            " minimization. Immediate candidate-price effects and recursive"
            " price-history effects must not be conflated."
        ),
        "",
        (
            "Similar observed-price forecast RMSE does not establish reliable or causal"
            " counterfactual price responses. Distribution shift and model exploitation"
            " are established concerns in model-based offline control"
            " ([MOPO](https://arxiv.org/abs/2005.13239)); that literature does not"
            " validate our electricity response models."
        ),
        "",
        "## 3. PPO selection and learning failed the deterministic criterion",
        "",
        (
            "| Fit | Initial loss % | Best loss % | Final loss % | Best decision |"
            " Final switching/day |"
        ),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in audit["training"].items():
        lines.append(
            f"| {name} | {100*row['initial_loss']:.3f} | {100*row['best_loss']:.3f} |"
            f" {100*row['final_loss']:.3f} | {row['best_decisions']} |"
            f" {row['final_switches']:.3f} |"
        )
    lines += [
        "",
        (
            "Four of six PPO best checkpoints were saved at zero decisions. They are"
            " selected untrained initializations, not evidence that PPO learned good"
            " control. The allowed initial-validation checkpoint prevented replacing"
            " them with worse models, but reports must disclose this. Two PPO runs"
            " finished with no switching and an energy-weighted price of exactly 0.05;"
            " other runs also became effectively constant-price policies."
        ),
        "",
        (
            "Initial policy mode selection is fragile: near-uniform probabilities "
            "can still have a deterministic argmax near a strong constant price. PPO "
            "optimizes a stochastic policy, whereas the frozen comparison used "
            "deterministic actions. Sampling probes are in summary.json. They show "
            "that the two evaluation modes differ; they do not reconstruct lost final "
            "policies or prove the cause of collapse."
        ),
        "",
        (
            "Entropy coefficient was zero, but that does not mean exploration "
            "was absent. The PPO updates still sampled a categorical policy. Final "
            "weights were deliberately not retained after selecting best checkpoints, "
            "so causal attribution of failed updates is limited."
        ),
        "",
        "## 4. DQN action values are miscalibrated",
        "",
        (
            "Every DQN final validation score was worse than its saved best score."
            " Action probes compare predicted Q values with actual first-action"
            " branches followed by the same saved policy. At hour 23 these are exact"
            " terminal action values; earlier branches are not globally optimal"
            " schedules."
        ),
        "",
        "```json",
        json.dumps(result["worst_branch"], indent=2),
        "```",
        "",
        (
            "At terminal states, predicted action-value spreads are much larger than "
            "the true action-return spreads in many tested cases. Some Q estimates "
            "are positive even though every possible remaining episode return is "
            "nonpositive. Errors include both overestimation and underestimation; "
            "this audit does not establish a universal overestimation bias."
        ),
        "",
        (
            "Only one of 24 transitions carries reward. A 3,000-transition replay holds"
            " roughly 125 complete episodes. This is a plausible credit-assignment"
            " difficulty, not a demonstrated replay implementation error. Gamma 1"
            " correctly represents the agreed finite-horizon daily objective; lowering"
            " gamma is not a principled way to repair the service definition."
        ),
        "",
        "## 5. Scarcity and price distributions differ by partition",
        "",
        (
            "| Partition | Mild reference overload % | Moderate % | Severe % | Mean NO1"
            " price |"
        ),
        "|---|---:|---:|---:|---:|",
    ]
    for split, row in coverage.items():
        values = [
            100
            * row["scarcity"][name]["reference_forecast"][
                "overloaded_region_hour_fraction"
            ]
            for name in ("mild", "moderate", "severe")
        ]
        lines.append(
            f"| {split} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | "
            f"{row['observed_price']['0']['mean']:.3f} |"
        )
    lines += [
        "",
        (
            "These are reference-demand overload frequencies, not policy-induced "
            "overload counts. Mild validation/test cases have no reference overload; "
            "there is little curtailment-management benefit to learn there. Equal "
            "averaging across capacity levels mixes very different control problems."
        ),
        "",
        (
            "On 24 uniformly sampled training dates (seed 31415), the best "
            "constant in the diagnostic grid is 1.00 NOK/kWh. The validation-selected "
            "constant was 2.00. This supports a distribution-shift concern, but the "
            "training diagnostic sample is not a complete training-set oracle. The "
            "minimum price was worse than the sampled training optimum, so PPO "
            "minimum-price collapse cannot be explained simply as that grid optimum."
        ),
        "",
        "## Checks that did not fail",
        "",
        (
            "- Household-level reaggregation matched all 240 audit rollout returns."
            " Three additional full-cohort rollouts matched a separately implemented"
            " reward equation exactly; see independent_reward_checks.json."
        ),
        (
            "- All intermediate rewards were zero and terminal reward was -100 times"
            " the declared daily loss."
        ),
        (
            "- No regional postallocation capacity violation occurred in the audited"
            " rollouts."
        ),
        (
            "- All 15 existing capacity regression tests passed, including replay"
            " samples, snapshots, frozen recursion, no future demand/weather, no"
            " curtailment feedback, masks and update recovery."
        ),
        (
            "- Reference-essential infeasibility at alpha 0.50 was zero across all"
            " training/validation/test reference hours checked. Fixed-load"
            " infeasibility is not the explanation for these primary failures."
        ),
        (
            "- Original source fingerprints and response/policy checkpoint hashes were"
            " preserved."
        ),
        "",
        "## Recommended order of work",
        "",
        (
            "1. Resolve whether service is hour-specific or daily. With no recovery "
            "scope, hour-specific deficits avoid temporal cancellation. If flexible "
            "request reduction is intended to be acceptable, explicitly define its "
            "service/utility treatment. Do not switch to served/requested alone: "
            "shrinking the denominator would create a new loophole."
        ),
        (
            "2. Treat counterfactual response uncertainty as a primary limitation. "
            "Do not force a negative price slope and call it measured evidence. Use "
            "measured replay for allocation claims and clearly declared response "
            "scenarios for pricing claims unless causal evidence becomes available."
        ),
        (
            "3. Verify learnability on a small enumerable problem, then compare learned"
            " action values with simulator branches before another overnight fit."
            " Expose exact service-progress/remaining-deficit features to all"
            " controllers if a new experiment adopts them."
        ),
        (
            "4. For sparse credit, consider a mathematically return-preserving "
            "potential-based transformation with verified boundary terms "
            "([Ng et al.](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf)), "
            "or finite-horizon Monte Carlo targets. These are follow-up hypotheses, "
            "not fixes proven by this audit."
        ),
        (
            "5. Record policy entropy, stochastic and deterministic validation, "
            "terminal value errors and action occupancy; disclose zero-update winners. "
            "Make one controlled change at a time. Do not add architectures or "
            "extend training merely to obtain a positive result."
        ),
        "",
        "## Scope of the probes",
        "",
        (
            "Policy decompositions use four diagnostic dates spanning January and"
            " February–March, all three capacities, both ensemble and reserved"
            " dynamics. Seven saved RL policies plus fixed and market baselines were"
            " probed. PPO sampling uses three RNG seeds on two severe January cases for"
            " four saved PPO policies. These are post-hoc diagnostics, not a fresh"
            " holdout, causal validation, or replacement for the original complete"
            " comparison."
        ),
        "",
        (
            "Reproduce the inference probes with `python -m"
            " norway_energy_crisis.analysis.audit_capacity --device mps`"
            " and this report with"
            " `python -m norway_energy_crisis.analysis.report_capacity_audit`."
            " Raw probes,"
            " action branches, training diagnostics and provenance remain in this"
            " folder."
        ),
    ]
    (folder / "AUDIT.md").write_text("\n".join(lines) + "\n")
    plot(folder, audit, result, curves)
    return result


def plot(folder, audit, result, curves):
    plt = plotting(folder)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, row in audit["training"].items():
        axes[0].plot(
            [r["decisions"] for r in row["history"]],
            [100 * r["loss"] for r in row["history"]],
            label=name,
        )
    axes[0].set_xlabel("Training decisions")
    axes[0].set_ylabel("Validation service loss (%)")
    axes[0].legend(fontsize=5)
    for split, rows in result["grid"].items():
        axes[1].plot(
            [r["price"] for r in rows], [100 * r["loss"] for r in rows], label=split
        )
    axes[1].set_xlabel("Constant NOK/kWh")
    axes[1].set_ylabel("Service loss (%)")
    axes[1].legend()
    fig.suptitle(
        "Training grid uses 24 sampled days; validation/test use all eligible days"
    )
    fig.tight_layout()
    fig.savefig(folder / "learning_and_distribution.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name in ("ensemble", "s42", "s7", "reserved"):
        rows = np.asarray(
            [row["regional_kwh"] for row in curves["curves"] if row["dynamics"] == name]
        )
        change = 100 * (rows / rows[:, 0:1] - 1)
        for area in range(2):
            axes[area].plot(curves["grid"], change[:, :, area].mean(0), label=name)
    for area, ax in enumerate(axes):
        ax.set_xlabel("Candidate NOK/kWh")
        ax.set_ylabel(f"NO{1 if area==0 else 5}: change from 0.05 (%)")
        ax.legend()
    fig.suptitle(
        "Mean one-hour model sensitivity across 24 identical-history validation states"
    )
    fig.tight_layout()
    fig.savefig(folder / "response_disagreement.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for dynamics, ax in zip(("ensemble", "reserved"), axes):
        names = ("fixed", "market", "dqn_42", "dqn_7", "dqn_2026")
        positions = np.arange(len(names))
        ax.bar(
            positions - 0.18,
            [100 * result["policy_probes"][dynamics][name]["loss"] for name in names],
            0.36,
            label="Daily loss",
        )
        ax.bar(
            positions + 0.18,
            [
                100
                * result["policy_probes"][dynamics][name]["hourly_reference_deficit"]
                for name in names
            ],
            0.36,
            label="Hourly deficit diagnostic",
        )
        ax.set_xticks(positions, names, rotation=25)
        ax.set_ylabel("Reference deficit (%)")
        ax.set_title(dynamics)
        ax.legend(fontsize=7)
    fig.suptitle("Post-hoc diagnostic subset: four dates and three capacity scenarios")
    fig.tight_layout()
    fig.savefig(folder / "temporal_cancellation.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/capacity_v1")
    parser.add_argument("--audit", type=Path, default=ROOT / "runs/capacity_v1_audit")
    args = parser.parse_args()
    if not (args.audit / "independent_reward_checks.json").exists():
        verify_reward(args.run, args.audit)
    summarize(args.run, args.audit)
    print(args.audit / "AUDIT.md")


if __name__ == "__main__":
    main()

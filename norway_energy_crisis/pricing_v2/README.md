# Pricing checkpoint

This experiment compares DQN trained against the old and upgraded frozen consumer
models. The primary comparison evaluates both policy groups under the **same**
upgraded dynamics, with reserved upgraded seed 2026 as a sensitivity case.
Own-simulator rewards are learning diagnostics, not a consumer-accuracy comparison.

- 717 spot-contract households, in the saved household order.
- 24 historical hours followed by 48 hourly decisions.
- One shared price: 0.05, 0.10, 0.20, …, 2.00 NOK/kWh.
- No allocation, consumption floor, recovery, backlog or consumer retraining.
- Separate NO1/NO5 limits use measured training aggregates' linear 95th, 90th
  and 75th percentiles. These are constructed stress scenarios, not feeder ratings.

## Reward and variables

- `regional_kw`: regional household hourly kWh summed and divided by one hour.
- `capacity_kw`: constructed regional average-power limit.
- `load_ratios`: regional power divided by regional capacity.
- `grid_weights`: weights chosen from preceding-hour demand, before acting.
- `energy`: household cumulative consumption, in kWh.
- `charge`: modeled spot-energy charge, in NOK; not a complete retail bill.
- Reference quantities come from an independent frozen-model rollout at recorded
  regional prices, starting from exactly the same measured history.

For `B(x) = (x / (1 + x))²`:

```text
g(z) = 0.5 × [B(max(z − 0.90, 0) / 0.10)
            + B(max(z − 1.00, 0) / 0.10)]
u = clip((preceding_load_ratio − 0.80) / 0.20, 0, 1)
grid_weight = 1 + 4 × u² × (3 − 2u)
```

- Weight is 1 at/below 80%, 3 at 90%, and 5 at/above capacity.
- Energy cost applies `B` to each household's absolute relative energy deviation
  beyond 5%, divided by 5%. Suppression and inflation are both penalized.
- Charge cost applies `B` to each household's relative charge increase beyond 5%,
  divided by 5%. Its denominator is the maximum of absolute reference charge,
  `0.01 × saved_training_price_median × reference_energy`, and `1e-6`.
- Reference energy denominators are at least `1e-6`. Negative reference charges
  remain valid. The saved training median is 0.51654 NOK/kWh.
- Cost is `(mean_hour_region(weight × g) + energy_cost + charge_cost) / 7`.
  The fixed divisor bounds cost in [0, 1] without weakening household penalties
  when grid weights increase.
- Grid costs are booked hourly; energy/charge costs only at hour 48. Gamma is 1,
  reward scale is 100, and episode reward equals `−100 × cost` exactly.
- Flattening, switching and jump penalties are absent; they remain diagnostics.
  Terminal penalties evaluate the whole episode, not the final action alone.

This is a soft objective, not masking or a capacity guarantee. Inaccurate consumer
dynamics may still be exploited. Constant prices, alternating extremes, pulses,
end effects and cross-household cancellation are audited without retuning weights.

## Information and dynamics

- Both adapters reuse saved preprocessing, static descriptors and frozen weights.
  Seeds 42/7 receive the same histories; their household mean advances the shared
  demand history. No monotonic response direction is imposed.
- Old weights come from `runs/comparison_v1/`; upgraded weights and price-free
  controls come from `runs/response_refinement_v2/selection.json`.
- Temperature persists from the origin. Future measured consumption and weather
  never enter simulation, reference prediction, policy observations or reward.
- At reset, only the first market day and its independent reference forecast are
  visible. Day two is released after 24 decisions. Unreleased forecasts/totals are
  masked in observations and blank in the dashboard.
- Training ends December 2021. January has 29 eligible validation origins;
  February–March has 57 follow-up origins. History and all targets remain inside
  each partition. Training days and capacity scenarios are sampled uniformly.

## Commands

From the repository root, with local prepared data and consumer checkpoints:

```sh
make check PYTHON=python
make pricing-preflight PYTHON=python
make train-pricing-awake PYTHON=python BUDGET_HOURS=10
```

The last command runs preflight, audits, baseline selection, training and final
evaluation under `caffeinate -d -i -s -m`, with MPS fallback disabled. Run the
identical command to resume; it does not allocate another ten hours. Override
`PRICING_OUTPUT` for a new experiment. MPS is required. CPU is used for tests,
parity verification and replay storage.

```sh
# Optional separate setup, outside the training budget.
make pricing-setup PYTHON=python
# Evaluate already-completed, frozen fits without new training.
make evaluate-pricing PYTHON=python
# Open the viewer independently; refresh while training runs.
make dashboard PYTHON=python
```

Preflight checks both 717-household engines against CPU, completes DQN updates,
verifies recovery and checks 48-hour price-free physical invariance.

## Training and recovery

- Six paired slots: old/new for seed 42, then 7, then 2026; 6,000 seconds each.
  Setup and final evaluation are measured separately.
- MultiInputPolicy, 256 → 128 ReLU, batch 64, CPU replay 3,000, learning starts
  2,400, one update per four decisions, target update 1,000, gradient clip 10,
  and four supporting CPU threads.
- Learning rate falls linearly `1e-4 → 1e-5` with consumed slot time. Epsilon
  falls `1 → 0.05` over its first 30%.
- Initial, ten-minute and final validation enumerate 29 origins × three scenarios.
  Validation, recording and saving count inside each slot. Measured reserves
  protect final validation and complete learner blocks; actual time/overrun are
  reported.
- Minimum mean validation cost selects the checkpoint, including the initial
  untrained policy if it stays best. Seed 42 is representative; all seeds appear.
- Atomic recovery states are saved about every five minutes after complete
  episodes and updates. They include optimizer, replay, environment, last
  observation, RNGs, counters, validation history and consumed budget.
- Only the active resumable state is retained. Data, preprocessing, weights,
  scenarios, objective, dependencies and source are fingerprinted. Incompatible
  resumes fail. MPS recovery is not promised bitwise identical. Discarded recorded
  interactions remain labelled rather than silently entering reward trends.

## Deliverables and interpretation

New outputs live in ignored `runs/pricing_checkpoint_v1/`:

- `settings.json`, `preflight.json`, `audit.json`: provenance/correctness checks.
- `status.json`, per-fit progress/history and `episodes.sqlite3`: live inspection.
- Six best policies, fit summaries and a selection manifest frozen before testing.
- `evaluation/`: all six policies and baselines under old, upgraded, reserved
  upgraded and upgraded price-free dynamics.
- `comparison.json`: 5,000 paired moving-block resamples, seven-day primary with
  three-/fourteen-day sensitivity. Scenarios and policy seeds are averaged within
  day before sampling; simulator seeds remain sensitivity cases.
- `diagnostics/`: sampled Q values versus same-policy model continuations, and a
  labelled 24-hour continuation beyond the consumer training horizon.
- Learning, component, demand/price and comparison plots plus `REPORT.md`.

Baselines are minimum/maximum constants, validation-selected constants per
scenario, rounded cohort-weighted market prices, a capacity rule, one-hour greedy
control and best constant per episode. The latter is a model-based oracle only
within the constant-price family.

A reliable training improvement requires a negative cost-difference interval under
both common upgraded and reserved dynamics. A switching claim also requires
beating the constant-price oracle. Stress, peaks, household energy/charges,
occupancy, jumps and switching are reported separately. Negative results close
this predefined study too; no extra fits are automatically scheduled.

The reviewed `Papers/1812.09567v1.pdf` supports history-dependent consumer states;
its simulated backlog/minimum-consumption assumptions are not transferred. Later
survey answers are assumed stable retrospectively; weather is regional reanalysis.
Previously viewed months/seeds remain retrospective evidence. Results establish
neither causal response, measured welfare, retail viability nor adaptive-weight
superiority over fixed weights.

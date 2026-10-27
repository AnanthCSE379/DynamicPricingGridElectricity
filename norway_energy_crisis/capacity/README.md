# Capacity and allocation simulator

This package compares delivered service and fairness under constructed electricity
shortages in two independent regional pools: Oslo/NO1 and Bergen/NO5. It preserves
the existing response models and pricing experiments. A completed experiment does
not require reinforcement learning to beat the baselines.

Historical replay allocates measured household consumption. Experimental prices
cannot change that consumption. The separate learned-response mode averages frozen
Transformer seeds 42 and 7 before advancing shared household histories. A reference
forecast under recorded regional prices defines protected requests and daily service.
Only requested demand goes back to the response model; curtailment has no effect on
later requests and is never queued for recovery.

`scenarios.py` calibrates regional 95th/90th/75th percentile capacities from training
hourly demand using linear quantiles. These are scenario assumptions, not measured
feeder ratings. Every interval is one hour, so a capacity in kW gives the same numeric
energy budget in kWh. Essential fractions are 0.50 in training and 0.25/0.75 in
sensitivity evaluation. All allocation methods serve essentials first, or proportionally
curtail essentials when they alone exceed the regional budget. Flexible allocation is
FCFS with synthetic order, proportional, or cumulative fair-priority water filling.
There is no transfer between the regions.

`environment.py` exposes safe 24-hour-history snapshots and known price/calendar
schedules. Future measured demand and temperature are absent. Regional temperature
persists at its last origin value. The 21 prices are 0.05, then 0.10 through 2.00 in
0.10 steps. Every action is valid. PPO receives all-true masks during training and
evaluation, so this study cannot attribute any benefit to invalid-action masking.

The objective is the household mean positive daily reference-service deficit,
excluding households with zero reference energy. Overserving one household cannot
offset another household's deficit. Intermediate reward is zero; terminal reward is
`-100 * loss`, with gamma 1. Charges, switching and fairness are reported separately.
Fairness uses Jain's index of capped household reference-service ratios and, separately,
conditional flexible-service ratios. Empty eligible groups have undefined fairness;
an eligible group receiving no service has index zero.

All controllers receive identical observations. A shared household MLP (64, 32) is
mean/max pooled separately by region. DQN and MLP MaskablePPO use a temporal MLP
(128, 64). Transformer MaskablePPO instead uses two width-64 temporal attention
layers, four heads, feed-forward width 128, fixed positions and no dropout. The
combined features have width 128; Q or actor/critic heads have widths 128 and 64.
Parameter counts are saved with every best checkpoint. The Transformer is a custom
feature extractor inside PPO, not a separately trained response model.

The nine fits run in seed order 42, 7, 2026, each with DQN, MLP PPO and Transformer
PPO. Each gets 4,000 seconds. Learning rates decrease linearly with consumed time:
1e-4 to 1e-5 for DQN and 3e-4 to 3e-5 for PPO. DQN uses 3,000 CPU replay transitions,
batch 64, warmup 2,400, one update every four decisions, target update 1,000 and
gradient clip 10. Repeated immutable observation arrays share CPU storage without
discarding transitions or features. Epsilon falls from 1 to 0.05 in the first 30%
of the slot. PPO uses 192-step rollouts, five epochs, batch 64, GAE 0.95, clip 0.2,
value coefficient 0.5, entropy coefficient 0, gradient clip 0.5 and target KL 0.03.
There is one active fit and four supporting CPU threads. All learners and response
networks use explicit MPS; no automatic CPU fallback is allowed.

Run from the repository root in the installed Python environment:

```sh
make install-dev PYTHON=python
make check PYTHON=python
make capacity-preflight PYTHON=python
make train-capacity-awake PYTHON=python
```

The equivalent unattended command performs setup, baselines, training and evaluation:

```sh
caffeinate -d -i -s -m python -u -m norway_energy_crisis.capacity.train \
  --budget-hours 10 --device mps --stage all
```

Use the same command to resume. `--stage audit`, `baselines`, `train` and `evaluate`
also expose individual stages. `make evaluate-capacity` evaluates completed fits.
`DATA_DIR`, `RESPONSE_ROOT` and `CAPACITY_OUTPUT` are Makefile path overrides.
Outputs default to the ignored `norway_energy_crisis/runs/capacity_v1/` directory.
Raw downloads, prepared data and response checkpoints must already exist locally.

Setup, baseline enumeration and final evaluation are outside the ten-hour fit phase
and measured separately. Every fit validates all 30 January days and three capacity
levels initially, every ten minutes and finally. Periodic validation/checkpointing
counts against its slot; measured validation and block durations reserve stopping
time. Blocks end at episode and learner-update boundaries. Actual durations and
any overruns are reported, rather than promising an exact wall-clock termination.

`status.json`, fit `progress.json` and `progress.csv` provide progress. Redirect
stdout to `run.log` when launching outside the terminal. `active_resume.pkl` contains
only the active fit: optimizer/model, replay or completed PPO rollout, environment,
last observations, RNGs, counters, consumed time and validation history. It is saved
atomically about every five minutes and removed when its fit finishes. Best policies
and metrics remain. Data, preprocessing, response weights, source, dependencies,
scenarios and objective are fingerprinted. Changed inputs reject resumption; choose
a new output directory. Local pickle checkpoints must be trusted. MPS resumption
is not promised bitwise identical.

Validation selects the model family by three-seed mean loss, with the declared
fairness/charge/jump/configuration tie-breakers. Seed 42 is always the representative
policy. `selection.json` is frozen before new final comparisons. February–March and
response seed 2026 were already examined earlier: every new comparison is a
retrospective follow-up, not a fresh holdout.

Final evaluation compares all RL families/seeds with fixed, market and capacity-rule
pricing, constant minimum/maximum diagnostics, and each day's best constant price.
It also isolates allocation methods under the same rule schedule. Reserved response
seed 2026 and matched price-free models are sensitivity scenarios. Alpha and individual
response-seed sensitivities evaluate the selected family without retraining. FCFS
order seeds 0, 1 and 2 are averaged within day. Paired moving-block intervals use
5,000 resamples, seven-day primary blocks and three/fourteen-day sensitivities;
policy seeds and capacity scenarios stay paired by calendar day.

Completion produces `fit_summaries.json`, `evidence.json`, `comparison.json`, full
`test_metrics.json`, diagnostic PNGs and `REPORT.md`. Pricing improvement requires
a negative service-loss difference interval under both training and reserved dynamics.
Switching additionally has to beat the daily constant-price comparator. Failures
and service/fairness trade-offs are valid outcomes and close this experiment.

Consumption and spot prices are measured. Survey descriptors were collected later
and assumed stable retrospectively. Weather is regional ERA5-Land reanalysis.
Capacities, essential fractions and FCFS arrival order are declared assumptions.
Alternative-price demand, delivery and energy charges are modeled quantities.
The simulator does not validate causal response, measured welfare, complete retail
bills, profitability, real feeder security, voltage, line flows or instantaneous peaks.
PV, batteries, device-level EV control and deferred service are outside this study.

The implementation uses SB3's [custom feature extractor interface](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html)
and [SB3-Contrib MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)
with compatible 2.9 releases. CPU regression tests generate their own fixtures;
full-cohort MPS parity, gradient and recovery checks are a separate mandatory preflight.

## Optional minimum supply

`norway_energy_crisis.minimum_supply.MinimumSupply` fits a reservation from each
household's positive training-load 1st percentile. This is a declared lower-tail
support rule, not an identified essential requirement or a validated accuracy
threshold. A fixed 0.1 kW reservation is available as a comparator.

Run the low-load diagnostics without retraining:

```sh
make minimum-supply-stress PYTHON=python DEVICE=mps
```

Outputs go to `runs/minimum_supply_v1/`, including `minimum_supply.json`, measured
low-load errors, artificial six-hour curtailment probes followed by 66 hours of
recursion, allocator stress results, a plot and `REPORT.md`. Use a new
`MINIMUM_SUPPLY_OUTPUT` on rerun.

Activate the reservation explicitly when constructing an environment:

```python
from norway_energy_crisis.minimum_supply import MinimumSupply

baseline = MinimumSupply.load("path/to/minimum_supply.json")
env = CapacityPriceEnv(provider, engine, scenarios, cache_folder,
                       minimum_supply=baseline)
```

Protect the larger of the assumed essential requirement and the demand-capped
reservation. Actual consumption never exceeds requests. Insufficient regional
capacity raises an error instead of promising an infeasible guarantee. Requests
still feed the consumer history; allocations do not. The ordinary training CLI
continues to use the original allocation settings unless explicitly integrated
with this constructor option. New baseline-aware training requires its own run
and experiment configuration; existing results and checkpoints are preserved.

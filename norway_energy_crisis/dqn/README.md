# Norwegian hourly DQN

Implemented a separate pricing experiment using the prepared Norwegian panel and frozen 24-hour Transformers. Raw downloads and Norwegian response checkpoints are preserved. Retired LCL summaries remain in the ignored local archive; its old code and checkpoints were removed.

One price applies to all 717 households each hour for 24 hours. The action grids contain 0.05 and increments of 0.10 NOK/kWh up to 2.00 or 3.00. Transformer seeds 42 and 7 predict from the same controlled histories; their mean demand is appended recursively. Seed 2026 is reserved until policy and cap selection are frozen.

The objective combines demand flattening, energy preservation, price-jump size and modeled energy-charge increases, with weights 1, 1, 0.1 and 1. Switching frequency has no penalty. Rewards sum to minus 100 times the normalized daily cost. Temperature stays at its last known value; future measured demand and weather never enter an episode.

Both the response ensemble and SB3 DQN run on MPS. Replay stays in CPU memory. Learning rate decreases linearly from 0.0001 to 0.00001; exploration decreases from 1 to 0.05 over the first 30% of each slot.

From the parent project directory:

```sh
caffeinate -d -i -s -m /Users/ananth/anaconda3/bin/python -u -m norway_energy_crisis.dqn.train --budget-hours 10 --device mps --output norway_energy_crisis/runs/dqn_v2
```

Training runs six sequential 100-minute slots: caps 2 and 3 for seeds 42, 7 and 2026. Periodic validation and saving count toward each slot; measured validation time is reserved for its final evaluation. Setup and final testing add time outside the ten-hour training phase. Keep the laptop open and connected to power.

The default output is `norway_energy_crisis/runs/dqn_v1/`. `status.json` shows the stage; each active fit's `progress.json` records decisions, updates, reward, learning rate and budget. Diagnostics contain price support, response curves, constant-price outcomes and validation baselines. The final `REPORT.md`, `comparison.json`, `test_metrics.json` and plots are written after all fits and held-out evaluations finish.

The command above uses a fresh output folder to preserve the completed experiment. Source fingerprints intentionally prevent resuming training after code changes. Existing policies and response checkpoints remain loadable for evaluation; `--response-root` selects the response comparison folder when needed.

After interruption, run the identical command. One local `resume.pkl` stores the active model, optimizer, replay, environment, CPU/MPS random states and consumed budget. Inputs/settings are fingerprinted. Completed fits keep only their best policy; checkpoint recovery preserves state without promising bitwise-identical MPS training.

Final comparisons include daily constant-price oracles, validation-selected fixed pricing, projected market schedules and greedy control. Policies are also evaluated on individual response models, reserved dynamics and a matched price-free ensemble. Confidence intervals use paired seven-day moving blocks, with three- and fourteen-day sensitivity. Cost differences smaller than 0.000001 are treated as numerical noise when declaring benefits.

Tests and a full-household GPU smoke check:

```sh
python -m unittest norway_energy_crisis.dqn.test_dqn -v
python -m norway_energy_crisis.dqn.train --check-only --device mps
```

Existing policies can be evaluated through the same entry point:

```sh
python -m norway_energy_crisis.dqn.train --evaluate norway_energy_crisis/runs/dqn_v1/cap3_seed42/best.zip --split test --engine reserved --device mps
```

Test access and reserved dynamics require the frozen `selection.json`. Prices model spot-energy charges rather than complete retail bills. Results do not establish causal demand response, supplier profitability or behavior beyond the 24-hour episode.

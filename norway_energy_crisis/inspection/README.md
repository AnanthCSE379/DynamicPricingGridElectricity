# Training inspector

The Streamlit viewer reads local files and does no training or inference. New DQN
and capacity DQN/MaskablePPO fits record every episode and every household. Existing
raw data, response checkpoints, objectives and saved research outputs are preserved.

From the repository root:

```sh
make install-dashboard PYTHON=python
make dashboard PYTHON=python
```

Open http://127.0.0.1:8501. The server binds only to this computer. Override
`DASHBOARD_PORT` if needed. The Runs folder field can point to another output root.

- **Reward trends:** every completed episode, rolling mean, highest reward so far,
  and a separate validation curve. Optimizer losses, learning rate and exploration
  are available where the training logs contain them. Rewards are maximized, so a
  less-negative reward is better. Different objectives are not mixed across runs.
- **Episode inspector:** select any recorded episode, its regional aggregate and
  up to eight households. View the initial history and complete simulated demand,
  selected and recorded regional prices, hourly reward components, cumulative
  reward, all-household heatmap and downloadable action/household CSV files.
  The **Household step details** panel selects one household and offers previous/
  next household, first/previous/next/last recorded hour and a direct hour input.
  Each step shows its power, raw prediction, reference, selected price, cumulative
  energy and modeled charge. Capacity runs also show delivered/essential power
  and unserved energy. The curve marks the inspected hour. Expand the exact
  sequence to see the incoming demand window before that action and download the
  individual household's CSV. Reward details remain labelled as cohort rewards.
  Each episode remembers its inspected hour when switching episodes or refreshing;
  navigation changes the view and never rewinds or advances the training process.
- **Latest household state:** the latest recorded episode, with an hourly slider
  and a sortable table for every household's requested/delivered/reference power,
  preceding-hour change and cumulative energy. Values are simulator outputs, not
  live household meter readings.
- **Diagnostics:** reward-accounting mismatches, nonfinite/negative values,
  constant actions, flat demand trajectories, cost-component domination and action
  occupancy. These are inspection signals, not proof that a policy is faulty.

Keep training running in one terminal and open the dashboard in another. Select
the actual training run rather than a checkpoint replay. The top **Refresh now**
button reads newly committed steps immediately. **Automatic refresh** defaults to
every 30 seconds and supports 5, 15, 30, 60 or 120 seconds. The last-refresh time is
displayed in IST. New run folders are discovered on refresh, including when the
viewer was opened before training started. Training continues independently.

**Latest household state** always follows the newest recording. Enable **Follow
newest episode** in the inspector to advance to each new episode automatically;
turn it off to inspect an older episode without refresh changing your selection.
Completed-episode reward curves and partially recorded episodes update as the
trainer commits data. Reading the dashboard never starts or resumes training.

The starting history consists of historical consumption. The recorded-price
reference is a frozen-model prediction, not a measured counterfactual. kWh is
interval energy; dividing by the interval duration gives hourly average kW.
Capacity experiments retain separate regional pools and explicitly distinguish
requested from delivered energy. No allocator is added to the pricing experiment.

The 48-hour [pricing checkpoint](../pricing_v2/README.md) records regional stress
ratios, pre-action grid weights and the actual hourly grid/terminal household
reward contributions. Constructed regional limits are shown without implying
allocation. Unreleased day-two prices and reference forecasts stay blank until
midnight. Selecting its top-level run follows the active fit on refresh.

## Recording and recovery

New fits automatically write `episodes.sqlite3` beside their checkpoints. SQLite
transactions keep each step's prices, reward, components and household arrays
consistent. WAL mode allows simultaneous read-only dashboard access. Arrays are
compressed; repeated starting histories and references are shared. No episodes
are sampled away or silently deleted. Storage grows with the number of episodes;
the viewer loads only the selected/latest household trajectories.

The wrapper makes no random draws, extra response-model calls, reward changes or
policy-observation changes. Recording overhead is included in the training budget.
At checkpoint recovery, interactions absent from the recovered learner are retained
and marked discarded. They are hidden from default reward curves but remain
inspectable. Partial episodes are retained and labelled interrupted.

Reward bars show the contributions actually booked by the environment. Daily
energy/charge penalties and terminal service loss evaluate the whole episode.
They are not causally attributed to the final price just because they are charged
at termination. No intermediate reward trajectory is invented.

Instrumentation is fingerprinted for new fits. Older training resumes with a
different source fingerprint are rejected; use a new output directory for new
training. Do not edit historical experiment manifests to bypass that check.

## Older checkpoints

Old runs saved validation summaries, not complete exploratory training histories.
The dashboard shows those summaries without fabricating individual episodes.
Create clearly labelled deterministic validation replays with:

```sh
make dashboard-replay PYTHON=python DEVICE=cpu REPLAY_DAYS=3
# Or select another DQN / MLP PPO / Transformer PPO fit:
python -m norway_energy_crisis.inspection.replay \
  --run norway_energy_crisis/runs/capacity_v1/ppo_mlp_seed42 \
  --days 3 --device cpu
```

These commands use the original experiment's response checkpoints, not the revised
consumer model, and never retrain a policy. Outputs go to separate
`runs/inspection_replays/` folders. Existing outputs are protected; specify a new
`--output` to repeat an export. Data/weight provenance is checked and the current
simulation-code identity is recorded. A replay is not claimed to reproduce the
original exploratory interactions, MPS arithmetic or wall-clock schedules.

All recordings are local and ignored by Git under `runs/`. Tests generate their own
small fixtures and exercise Streamlit controls, observer neutrality, terminal
accounting, partial reads and completed DQN/PPO update recovery.
Core checks retain pandas 2/3 coverage. The optional Streamlit UI test runs in the
pandas 2 CI job, keeping framework dependencies out of the pandas 3 pipeline job.

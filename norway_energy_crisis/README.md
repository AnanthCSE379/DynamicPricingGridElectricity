# Norwegian data and models

The active workflow is **data → frozen consumer model → hourly pricing policy →
comparison and dashboard inspection**.

## Data pipeline

- `clean_data.py` validates the six raw CSVs, household/hour keys and survey codes;
  joins prices by region, date and hour; and records source hashes.
- Demand stays in kWh per one-hour interval. Price converts precisely from
  NOK/MWh to NOK/kWh. Genuine zero demand and negative prices are preserved.
- Regional ERA5-Land temperature is cached and aligned to the dataset's 24-hour
  day convention. It is a regional proxy, not household measurement.
- `preprocess.py` fits scales and category encodings on training data only and
  saves the inference contract. It preserves unknown survey answers explicitly.
- The primary cohort contains 717 spot-contract households: 487 in NO1 and 230
  in NO5. The full cleaned panel retains all 1,136 households.

Training ends December 2021; January 2022 is validation; February–March is
retrospective follow-up evaluation. History and target windows stay within each
partition. Consumer rollouts persist origin temperature and never use future
measured demand or weather. Later survey descriptors are assumed stable over the
observation period.

Initial data preparation, when outputs do not already exist:

```sh
make clean-data PYTHON=python
make preprocess PYTHON=python
```

Raw CSVs remain in this directory to preserve existing provenance paths.
`cleaned/`, `prepared/`, `weather_cache/` and `runs/` are local ignored folders.
Existing prepared data can be reused directly.

## Models and inspection

| Component | What it contains |
| --- | --- |
| `train_response.py` | Original RNN/GRU/LSTM/Transformer comparison and checkpoint loading |
| [response_v2/](response_v2/README.md) | Upgraded Transformer, extra-household pretraining, price-free controls and recursive evaluation |
| [pricing_v2/](pricing_v2/README.md) | Matched old/upgraded DQN training with 48-hour episodes and state-dependent stress weights |
| [inspection/](inspection/README.md) | Rewards, actions, household traces, refresh and checkpoint-recovery diagnostics |
| [analysis/](analysis/README.md) | Price/load support, curtailment probes and 72-hour sliding forecasts |
| [capacity/](capacity/README.md) | Separate regional allocation study and tested allocators |
| [dqn/](dqn/README.md) | Original pricing comparison and shared utilities used by later experiments |

Current pricing has no allocation, consumption floor or recovery queue. Capacity
limits are declared stress-test assumptions. Spot-energy charges are accounting
proxies, not complete bills or measured welfare.

```sh
make check PYTHON=python
make dashboard PYTHON=python
make pricing-status PYTHON=python
```

The main Makefile lists current workflows. `make help-legacy` lists earlier
training and diagnostic commands. Those commands retain their existing names;
standalone investigation modules now live under `norway_energy_crisis.analysis`.

## Results and reproducibility

- Original consumer results: `runs/comparison_v1/REPORT.md`.
- Upgraded consumer results: `runs/response_refinement_v2/REPORT.md`.
- Current pricing outputs: `runs/pricing_checkpoint_v1/`; its final report is
  written after evaluation finishes. Read `status.json` for the actual stage.
- Earlier pricing/allocation results: `runs/dqn_v1/` and `runs/capacity_v1/`.
- The [historical pipeline report](docs/pipeline_report.md) records the initial
  data checks and original benchmarks; it is not the current pricing result.

Data, weights and source fingerprints remain intact for the active experiment.
Only obsolete process markers, failed-start logs and regenerable completed-run
caches were removed. Unique metrics and checkpoints remain available. The
superseded refinement plan and original investigation source snapshots are in the
ignored `local_archive/norway_cleanup_2026_10_27/` folder.

Forecast improvements and simulator optimisation do not establish causal demand
response. Previously examined dates and seeds remain retrospective evidence.

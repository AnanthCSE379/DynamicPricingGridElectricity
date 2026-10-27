# Norwegian electricity pricing research

[![Code checks](https://github.com/AnanthCSE379/DynamicPricingGridElectricity/actions/workflows/ci.yaml/badge.svg)](https://github.com/AnanthCSE379/DynamicPricingGridElectricity/actions/workflows/ci.yaml)

The project models household consumption and learns hourly electricity prices
using the Norwegian energy-crisis dataset.

- Validated hourly readings for 1,136 households; 717 spot-contract households
  form the pricing cohort.
- A frozen Transformer predicts household consumption from consumption/price
  history, calendar, temperature and household descriptors.
- The current DQN experiment uses 48-hour episodes and compares training on the
  original and upgraded consumer models across three seeds.
- Its reward combines regional grid stress, household energy deviation and
  modeled spot-energy charges. Grid weights rise with preceding demand.
- A Streamlit dashboard shows training rewards, price actions, reward components
  and individual household trajectories, including partially recorded episodes.

Pricing currently models consumption without allocation or forced consumption
floors. Earlier allocation experiments remain available for further research.
All counterfactual price responses are model predictions; they do not establish
causal consumer response or actual electrical-grid stability.

## Organisation

| Location | Purpose |
| --- | --- |
| [Data pipeline](norway_energy_crisis/README.md) | Cleaning, preprocessing and original consumer models |
| [response_v2/](norway_energy_crisis/response_v2/README.md) | Upgraded consumer models and sequence evaluation |
| [pricing_v2/](norway_energy_crisis/pricing_v2/README.md) | Current 48-hour DQN comparison and shared reward accounting |
| [inspection/](norway_energy_crisis/inspection/README.md) | Training and household dashboard |
| [analysis/](norway_energy_crisis/analysis/README.md) | Dataset audits and consumer-behaviour investigations |
| [capacity/](norway_energy_crisis/capacity/README.md) | Earlier regional allocation experiments |
| [dqn/](norway_energy_crisis/dqn/README.md) | Earlier pricing study and shared simulation/training utilities |
| [make/legacy.mk](make/legacy.mk) | Commands for earlier studies and diagnostics |

The shared utilities retain their original module paths so frozen experiment
fingerprints and checkpoint loading remain valid.

## Commands

Run from this directory using the intended Python environment:

```sh
make install-dev PYTHON=python
make check PYTHON=python
make install-dashboard PYTHON=python
make dashboard PYTHON=python
make pricing-status PYTHON=python
```

For the bounded pricing experiment:

```sh
make pricing-preflight PYTHON=python
make train-pricing-awake PYTHON=python BUDGET_HOURS=10
make evaluate-pricing PYTHON=python
```

Training uses explicit MPS and `caffeinate -d -i -s -m`. The identical command
resumes a compatible interrupted run. Choose a fresh `PRICING_OUTPUT` after
changing fingerprinted code. Evaluation requires completed fits and frozen
selection. Use `make help` for current commands and `make help-legacy` for earlier
experiments. The [pricing guide](norway_energy_crisis/pricing_v2/README.md) explains
settings, information boundaries, checkpoints and interpretation.

Raw downloads, prepared arrays, weather, papers and `runs/` stay local and are
ignored by Git. Consumer checkpoints and prepared data are needed for real-data
inference/training; cloning the code does not download them. Completed reports,
metrics and unique checkpoints are retained. Regenerable caches from completed
older experiments were removed. Historical LCL material remains in the ignored
`local_archive/lcl/` directory.

GitHub Actions runs formatting, lint and every discovered fixture test using
Python 3.13 with pandas 2 and pandas 3 on CPU. Optional dashboard tests run in the
pandas 2 job. Tests create their own data/weather/checkpoints and need no private
downloads. Full-cohort MPS checks run locally.

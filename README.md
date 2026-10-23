# Household demand response and tariff selection

[![Code checks](https://github.com/AnanthCSE379/DynamicPricingGridElectricity/actions/workflows/ci.yaml/badge.svg)](https://github.com/AnanthCSE379/DynamicPricingGridElectricity/actions/workflows/ci.yaml)

The active implementation uses the Norwegian energy-crisis dataset.

- [Data and response models](norway_energy_crisis/README.md): validated hourly data, training-only preprocessing, and RNN/GRU/LSTM/Transformer comparisons for 717 spot-contract households. The selected 24-hour Transformer achieved mean test RMSE of 0.83645 kWh.
- [Hourly DQN experiment](norway_energy_crisis/dqn/README.md): completed six MPS training runs comparing 2.00 and 3.00 NOK/kWh price caps. Validation selected the 2.00 cap. DQN did not outperform the stronger market-tracking and greedy baselines under the agreed four-term objective.

Raw downloads, reference papers, cleaned/prepared data, weather caches and model runs stay local and are excluded from Git. Existing local checkpoints are required to evaluate the trained policies; cloning the source alone does not include them.

The retired LCL scripts, checkpoints and generated outputs were removed. Essential LCL reports, metrics and source fingerprints remain in the ignored `local_archive/lcl/` folder (about 100 KB). Original downloads remain in the ignored `raw_data/` folder.

Install dependencies from `requirements.txt`. Dataset preparation, training, evaluation and test commands are documented in the linked pipeline guides. Forecasting and simulated pricing results do not establish causal customer response.

Run the complete regression suite with `python -m unittest norway_energy_crisis.test_pipeline norway_energy_crisis.dqn.test_dqn -v`. The MPS test runs when the GPU is available. New training after source changes requires a fresh output folder; saved checkpoints remain compatible for inference and evaluation.

The [Makefile](Makefile) lists commands with `make help`. Use an activated Python environment, then:

```sh
make install-dev
make check
make train-dqn-awake BUDGET_HOURS=10
```

`make check` runs formatting checks, linting and tests. Tests generate temporary data, weather and checkpoints without the ignored downloads. [GitHub Actions](.github/workflows/ci.yaml) runs the same checks on pushes, pull requests and manual dispatches using Python 3.13 with both pandas 2 and pandas 3 on CPU runners. Full training and MPS checks use local data and hardware.

`make prepare` performs initial cleaning and preprocessing; existing output directories are protected. The training targets use fresh `comparison_v2` and `dqn_v2` output folders by default and resume from them when inputs and source match. DQN uses the completed `comparison_v1` response checkpoints unless `RESPONSE_ROOT` is overridden. Evaluation defaults to the completed `dqn_v1` experiment. Override paths or the interpreter as needed, for example `make check PYTHON=python` or `make train-dqn RESPONSE_ROOT=norway_energy_crisis/runs/comparison_v2 DQN_OUTPUT=norway_energy_crisis/runs/dqn_v3`.

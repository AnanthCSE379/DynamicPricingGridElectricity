# Household demand response and tariff selection

The active implementation uses the Norwegian energy-crisis dataset.

- [Data and response models](norway_energy_crisis/README.md): validated hourly data, training-only preprocessing, and RNN/GRU/LSTM/Transformer comparisons for 717 spot-contract households. The selected 24-hour Transformer achieved mean test RMSE of 0.83645 kWh.
- [Hourly DQN experiment](norway_energy_crisis/dqn/README.md): completed six MPS training runs comparing 2.00 and 3.00 NOK/kWh price caps. Validation selected the 2.00 cap. DQN did not outperform the stronger market-tracking and greedy baselines under the agreed four-term objective.

Raw downloads, reference papers, cleaned/prepared data, weather caches and model runs stay local and are excluded from Git. Existing local checkpoints are required to evaluate the trained policies; cloning the source alone does not include them.

The retired LCL scripts, checkpoints and generated outputs were removed. Essential LCL reports, metrics and source fingerprints remain in the ignored `local_archive/lcl/` folder (about 100 KB). Original downloads remain in the ignored `raw_data/` folder.

Install dependencies from `requirements.txt`. Dataset preparation, training, evaluation and test commands are documented in the linked pipeline guides. Forecasting and simulated pricing results do not establish causal customer response.

Run the complete regression suite with `python -m unittest norway_energy_crisis.test_pipeline norway_energy_crisis.dqn.test_dqn -v`. The MPS test runs when the GPU is available. New training after source changes requires a fresh output folder; saved checkpoints remain compatible for inference and evaluation.

# Norwegian demand-response models

Implemented separate cleaning, preprocessing and response-training scripts for the Norwegian energy-crisis dataset. Raw downloads are preserved. Retired LCL reports and metrics are kept locally in the ignored `local_archive/lcl/` folder; its old code and checkpoints were removed.

The real-data audit passed: 14,913,408 readings, 1,136 households and 1,469 genuine zeros. The primary cohort contains 717 households (487 NO1, 230 NO5). Cached weather has no unresolved gaps. Prepared evaluation uses 30 validation days and 58 test days, with identical origins for both history lengths.

- `clean_data.py` validates the six CSVs, joins prices by region/hour, preserves zero consumption and negative prices, and attaches cached ERA5-Land regional temperature. It records source hashes and daylight-saving adjustments.
- `preprocess.py` selects hourly spot-contract households, fits training-only scales and household encodings, and prepares identical forecast origins for five- and 24-hour histories.
- `train_response.py` compares shared RNN, GRU, LSTM and Transformer models. Training progresses from one-hour to six-hour to recursive 24-hour targets. Screening uses one seed; the two finalists and a matched price-free control are confirmed across three seeds before final testing.
- `test_pipeline.py` checks joins, units, missing-category handling, time alignment, split isolation, model interfaces, recursive information boundaries, checkpoints and the complete comparison workflow.

Training covers October 2020–December 2021; January 2022 is validation; February–March 2022 is the final test period. Every example keeps its full history and forecast inside one partition. Future measured demand and temperature are never supplied during rollouts.

Household contract type and dwelling/heating/EV descriptions come from a later survey and are assumed stable. Retrospective response/savings answers are excluded. Forecast accuracy does not establish causal price response. The response-model pipeline is separate from the new [Norwegian DQN experiment](dqn/README.md), which uses MPS and compares hourly price caps of 2.00 and 3.00 NOK/kWh.

From the parent project directory:

```sh
python -m norway_energy_crisis.clean_data
python -m norway_energy_crisis.preprocess
python -m norway_energy_crisis.train_response --output norway_energy_crisis/runs/comparison_v2
```

On macOS, keep the display and machine awake during training:

```sh
caffeinate -d -i -s -m python -u -m norway_energy_crisis.train_response --output norway_energy_crisis/runs/comparison_v2
```

Both comparisons are complete. The 24-hour Transformer achieved mean test RMSE 0.83645 kWh across three seeds, versus 0.84228 without prices and 1.11073 for previous-day demand. Response results are in `runs/comparison_v1/REPORT.md`. The ten-hour MPS DQN experiment selected the 2.00 NOK/kWh cap, but did not outperform market-tracking or greedy baselines under the total objective. Its results are in `runs/dqn_v1/REPORT.md`. See the DQN README for controls and outputs.

Cleaned arrays, prepared inputs and experiment runs are stored in `cleaned/`, `prepared/` and `runs/`. Source outputs are protected against replacement. Training resumes from the last completed epoch with the identical command. Each run retains its best checkpoint; only active runs retain a resumable checkpoint.

Saved checkpoints remain usable after the code cleanup. Training also fingerprints source files, so start new fits in a fresh `--output` folder after changing code. Existing results and their original provenance stay in the completed folders. Cleaning and preprocessing commands above are for initial setup; reuse the existing prepared data for new training runs.

The comparison writes `runs/comparison_v1/status.json`, per-run histories and metrics, frozen model selection, and a final `REPORT.md`. The report recommends later simulator use only if the held-out 24-hour comparison passes both baseline and price-free checks.

Verification: `python -m unittest norway_energy_crisis.test_pipeline -v`. Eight tests cover the full workflow and exact checkpoint recovery after interruption. Completed data findings and the current experiment stage are summarized in `REPORT.md`.

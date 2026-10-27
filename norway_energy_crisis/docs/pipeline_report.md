# Norwegian pipeline: completed work

Cleaning validated all 14,913,408 hourly observations for 1,136 households. There are no missing hours, conflicting household/hour records, missing consumption values or negative consumption. All 1,469 genuine zeros were retained. Raw files were preserved and their hashes recorded.

The primary cohort has 717 hourly spot-contract households: 487 in Oslo/NO1 and 230 in Bergen/NO5. Regional prices preserve negative values and use the precise NOK/MWh divided by 1,000 conversion. ERA5-Land temperature was cached and aligned to the panel's 24-hour-day convention; no unresolved weather gaps remain.

Training-only preprocessing is complete. Five- and 24-hour histories share 10,921 eligible training origins, 30 validation days and 58 held-out test days. Every history and complete 24-hour target stays within its chronological partition. Calendar encoding, household descriptors and normalization are saved for inference.

Eight verification tests passed, including all four architectures, recursive information boundaries, price-free invariance, model checkpoint loading, interrupted-run recovery and a temporary-data execution of the complete 15-fit comparison. All four architectures also passed a real-data rollout check.

The response comparison is complete. The selected 24-hour Transformer achieved mean test RMSE 0.83645 kWh, compared with 0.84228 for the matched price-free models and 1.11073 for previous-day demand. Prices improved forecasting modestly. Full response results are in `runs/comparison_v1/REPORT.md` and `comparison.json`.

A separate Norwegian DQN experiment completed six MPS fits, 1,204,632 decisions and approximately ten training hours. It compared 2.00 and 3.00 NOK/kWh caps across three policy seeds, training against the shared-history mean of response seeds 42 and 7 and reserving response seed 2026 until selection. Switching frequency was excluded from the four-term reward. Validation selected the 2.00 cap and policy seed 2026. On the ensemble test scenario, three-seed mean costs were 0.0008231 for cap 2 and 0.0010664 for cap 3, versus 0.0001197 for projected market pricing and 0.0001845 for greedy control. Neither the wider-range nor switching benefit passed the agreed checks. Full results are in `runs/dqn_v1/REPORT.md` and its accompanying metrics.

Household contract and static survey descriptors are assumed stable across the observation period. Temperature is a regional reanalysis proxy. Forecast improvements and DQN simulator optimization cannot establish causal demand response to prices. Retired LCL reports and metrics remain in the ignored local archive; old LCL code and checkpoints were removed.

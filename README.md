# Household demand response and tariff selection

Built and tested a GRU demand predictor and a Stable Baselines3 DQN tariff selector using the LCL household dataset.

- The GRU uses five hours of separate readings for 1,117 households and the next tariff to predict each household's next half-hour consumption. It was retrained from scratch; validation selected epoch 27.
- Shared preprocessing fills missing demand with training-only household means, retains availability flags, scales demand and prices, converts units, and encodes time using sine/cosine features. Missing targets are excluded from GRU training and evaluation.
- DQN chooses one shared Low, Normal, or High tariff every two hours for 24 hours. There is no recovery period. It completed 100,000 training decisions.
- The reward combines demand flattening, energy preservation, tariff-change frequency, jump size, and bill-increase penalties. Demand, energy, and bill references come from a Normal-price GRU rollout starting from the same history.

## Validation results

Tested 147 tariff schedules on each of 21 validation days, shuffled price inputs, and seven objective variants. Normal was best among the candidate schedules on 19 days. The saved best DQN always selected Normal; the final DQN performed worse. Scoring demand flattening alone allowed alternatives to improve that penalty by about 18% on average.

The GRU's paired validation RMSE was 0.1822 kWh, compared with 0.1782 kWh for persistence. Its paired test RMSE was 0.1855 versus 0.1807 kWh. These results do not establish useful dynamic pricing or causal customer response.

## Files

- `household_response/`: GRU model, data features, inference, and tests.
- `price_selector/`: shared preprocessing, Gymnasium environment, DQN training, and validation diagnostics.
- `scripts/`: household data preparation, GRU training/reporting, and shared raw-data cleaning/splitting helpers.
- `raw_data/`: original source datasets and tariff workbook.
- `results/five_hour_response/`: prepared panel, retrained GRU, metrics, and predictions.
- `results/price_selector_dqn/`: best and final DQN models, training records, and validation history.
- [Diagnostic report](results/price_selector_diagnostics_validation/REPORT.md): schedule comparisons and price-response tests.
- `results/overnight_training.log`: completed data preparation, GRU training, and DQN training log.

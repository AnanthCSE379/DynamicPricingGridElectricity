# Price-response diagnostics

Validation days: 21; schedules/day: 147; households: 1117.

## One-step forecast and price sensitivity

- Factual GRU paired RMSE: 0.182206 kWh.
- Persistence paired RMSE: 0.178170 kWh.
- Shuffled-price RMSE: 0.182361 kWh (10 permutations).
- Low vs Normal aggregate prediction change: 0.262%.
- High vs Normal aggregate prediction change: -1.144%.
- Households predicting increasing demand with price: 578/1117.

Price enters only the last affine layer before softplus; immediate price-response direction is fixed per household for every history.

## Same-day policy comparisons

| Policy | Mean cost (lower is better) |
|---|---:|
| normal | 0.0060519 |
| low | 0.0101789 |
| high | 0.2365818 |
| best | 0.0060519 |
| final | 0.0120695 |

## Objective ablations

| Objective | Normal cost | Best shared schedule cost | Per-day search cost | Days beating Normal |
|---|---:|---:|---:|---:|
| default | 0.0060519 | 0.0060519 | 0.0059533 | 2 |
| demand_only | 0.0171562 | 0.0143736 | 0.0139415 | 21 |
| no_bill | 0.0088027 | 0.0088027 | 0.0086568 | 3 |
| no_energy | 0.0088027 | 0.0088027 | 0.0084976 | 2 |
| no_stability | 0.0057187 | 0.0056762 | 0.0056140 | 21 |
| demand_weight_5 | 0.0122210 | 0.0122210 | 0.0121601 | 3 |
| bill_weight_0.1 | 0.0084200 | 0.0084200 | 0.0082829 | 2 |

## Interpretation limits

- Finite schedule search does not prove global optimality.
- Per-day candidate selection uses simulator forecasts; it is an exploratory oracle, not a deployable policy.
- Weight variants are evaluated on the same candidate pool without retraining DQN.
- One-step forecast diagnostics do not validate 24-hour autoregressive accuracy or causal price response.

Passed 86 numerical consistency checks. Detailed inputs, schedules, per-day results and metrics accompany this report.

Both non-Normal winning schedules were independently rerun through the serial Gymnasium environment and retained their improvement. Maximum batch-vs-serial relative RMS prediction difference was 6.91e-05.

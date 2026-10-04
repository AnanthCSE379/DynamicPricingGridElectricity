# GRU customer-response experimental report

## A. What the trained GRU actually predicts

The checkpoint is a two-layer 64-unit GRU with a 48 half-hour (24 h) input. Its nine inputs are prior standardized household-mean consumption, six cyclical calendar encodings, a weekend flag, and tariff price. It predicts the next **absolute household-mean demand** in standardized units; the saved scaler inverse-transforms this to kWh per half-hour. There are no weather, solar, appliance, or customer-identity inputs.

Experiments use 96 held-out historical seed windows, balanced across four time-of-day bands. For every intervention, the same seed is rolled under a baseline 11.76 p/kWh tariff and a counterfactual tariff path; no future observed load is fed into either rollout.

## B. Derived quantities

“Removed”, “recovered”, “net reduction”, elasticity, rebound, and flexible fraction are calculated from paired GRU trajectories. The GRU itself predicts demand only; it does not predict withdrawal, shifting, or energy conservation.

## C. Findings and limits

* The only observed training prices are 3.99, 11.76, and 67.20 p/kWh. Intermediate experiments are interpolation and are marked in the raw results; no price above 67.2 p/kWh is used.
* Solar, net-demand, and clustering claims cannot be made because solar is not a model input and the target is an aggregate feeder average rather than individual customers.
* Multi-step results are autoregressive GRU rollouts, so they are valid model probes but may compound one-step model error. Rebound or repeated-event patterns are therefore model behaviour, not evidence of psychology or physical energy conservation.
* The tolerance and threshold criteria in `behavioural_parameters.csv` are operational diagnostics with explicitly stated sensitivity rules, not universal customer limits.

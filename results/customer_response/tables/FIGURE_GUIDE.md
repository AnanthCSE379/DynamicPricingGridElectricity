# Figure interpretation guide

All means summarize **96 held-out GRU seed histories**; the price-response figures show 95% confidence intervals. Red points denote actual training price levels. Intermediate price levels are interpolation diagnostics.

1. **Price vs predicted demand** — The model’s four-half-hour mean demand falls from 0.210 kWh/hh at 11.76p to 0.193 kWh/hh at 67.20p.
2. **Price vs percentage reduction** — At the supported high price, the paired counterfactual reduction is 6.55% (95% CI approximately ±1.07 pp). Low price 3.99p increases demand by 4.85%.
3. **Price vs shifted/recovered energy** — This shows energy that appears above baseline after the event. It is a derived GRU-trajectory quantity, not an energy-conservation guarantee.
4. **Price vs net withdrawal** — The net change sums the full event-plus-recovery horizon. Positive values mean the intervention trajectory remains below its paired baseline overall.
5. **Price vs elasticity** — Elasticity is non-constant. At 67.2p it is -0.014; it is a finite event elasticity, not a structural demand parameter.
6. **Marginal response** — The plot is diagnostic only because its middle prices are interpolation; it does not establish a supported saturation point.
7. **Duration response** — Mean high-price reduction rises from 4.43% at 0.5 h to 10.63% at 8 h. No duration plateau is demonstrated in this tested horizon.
8. **Rebound magnitude** — At 67.2p the mean rebound peak is 0.0053 kWh/hh. It is a small model-predicted post-event overshoot, not proof of real-world load shifting.
9. **Removed vs recovered energy** — Points below the diagonal retain a net reduction; the aggregate recovery ratio at 67.2p is 0.55.
10. **Price-change magnitude** — Same response data as Figure 2 expressed as the high-minus-baseline price change.
11. **Sudden vs gradual paths** — Both are autoregressive GRU trajectories. The gradual path uses unsupported intermediate prices (20, 35, and 50p), so retain it as a scenario illustration rather than primary evidence.
12. **Time-of-day × price heatmap** — At 67.2p, response ranges from -0.92% (Night (00:00–05:30)) to 10.34% (Afternoon (11:00–16:30)); night does not show reliable high-price curtailment in this model.
13. **Repeated events** — Reduction rises from 5.84% at the first event to 7.78% at the third and ends at 6.90%. This is non-monotonic, so there is no evidence of monotonic response attenuation.
14. **Baseline-demand bands** — This is conditional heterogeneity across seed histories, not customer-level heterogeneity.
15. **Flexible/non-flexible components** — The “remaining” bar is only the demand remaining at the highest observed training price. It is an inferred non-responsive load at 67.2p, not essential load.
16. **Tolerance/saturation** — No saturation price is identified within observed training support: only baseline and one higher supported price exist.
17. **Price history** — At the same current 67.2p price, mean one-step reduction is 0.0122 kWh/hh after normal history versus 0.0077 after high history. This supports model memory of recent tariff inputs.

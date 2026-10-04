# CustomerTwin: high-level behavioural laws

Define the sudden cost spike as `x = p / 14.23 - 1` and the model-implied withdrawal fraction as `W = 1 - q(p) / q(14.23)`.

These equations describe controlled interventions on the saved GRU checkpoint. They are conditional model summaries, not causal estimates from a customer experiment.

## All held-out intervals

For price increases above 14.23 p/kWh, the best simple approximation is:

`W(x) = 0.026525 · x^0.533357`  (RMSE 0.18 percentage points)

Therefore `q(p) ≈ 0.212552 · [1 − W(p / 14.23 − 1)]` kWh/half-hour.

## Winter evening (Dec–Feb, 17:00–20:30)

For price increases above 14.23 p/kWh, the best simple approximation is:

`W(x) = 0.035416 · x^0.547068`  (RMSE 0.23 percentage points)

Therefore `q(p) ≈ 0.346241 · [1 − W(p / 14.23 − 1)]` kWh/half-hour.

## Winter overnight (Dec–Feb, 00:00–05:30)

No simple monotonic withdrawal equation is supported: the model's high-price response reverses sign in this context.

## Autumn evening (Oct–Nov, 17:00–20:30)

For price increases above 14.23 p/kWh, the best simple approximation is:

`W(x) = 0.033763 · x^0.582198`  (RMSE 0.17 percentage points)

Therefore `q(p) ≈ 0.315108 · [1 − W(p / 14.23 − 1)]` kWh/half-hour.

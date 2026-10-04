# Customer demand-response model: empirical interrogation

## What is deployed

For each next half-hour \(t\), the ConsumerTwin predicts normalized demand from a 48-step (24-hour) history:

\[z_t=f_\theta(X_{t-48:t-1}), \qquad \hat q_t=\max(0.01,\mu+\sigma z_t).\]

Each row in \(X\) contains previous normalized consumption, sine/cosine encodings of half-hour, weekday and month, a weekend indicator, and a tariff.  The deployed controller only overwrites the final tariff entry, \(p_{t-1}\), before predicting \(q_t\).  Thus the test measures a **one-half-hour-lagged, conditional association**, not a contemporaneous structural demand curve.

## Usable local equations

For each context below, use the reported surrogate over the tested 5–80 p/kWh range:

\[\bar q(p)\approx a+b(p-14.23)+c(p-14.23)^2.\]

The JSON file is the authoritative machine-readable coefficient source.  The linear derivative at the historical tariff is \(b\), and the corresponding point elasticity is \(b\,14.23/\bar q(14.23)\).

## Interpretation limit

The GRU is a nonlinear predictive model trained on observational trial data.  Its response curve should be treated as a conditional simulator under the original data distribution—not proof that changing prices causes the same effect in the real world, particularly at 40–80 p/kWh.  For a portable causal equation, validate the curve against randomized/experimental price variation or fit a demand model with explicit identification assumptions.

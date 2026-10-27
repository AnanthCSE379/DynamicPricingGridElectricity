# Consumer simulator refinement

The Transformer predicts next-hour household consumption from 24 hours of demand,
prices, regional temperature, calendar features and saved household descriptors.
Its output is consumption in kWh. Prices are supplied by the caller.

The comparison separates the old persistence head, a direct positive head, and
extra-household pretraining. All three share linear/log demand features and price
changes. Configuration A uses the persistence head and 717-household pretraining;
B uses the direct head and the same cohort; C uses the direct head and all 1,136
households during pretraining. Pretraining masks all price features because paid
tariffs are unknown for the extra households. Fine-tuning uses only the 717
hourly-spot households. Matched price-free controls use the same architecture and
training phases.

Natural zero/low-load transitions receive additional sampling coverage, with
importance weights preserving the uniform household/origin MSE objective. No
synthetic recovery targets, consumption floors or fixed price-response direction
are imposed. Forecasts append predicted requests, never forced allocations.

Training uses MPS explicitly. Eight 50-minute fit slots leave time within the
ten-hour budget for preflight, evaluation and reporting. All runs checkpoint
approximately every five minutes after a completed update. Resume the identical
command with unchanged source, dependencies and data. Resumption does not promise
bitwise-identical MPS optimization. The run lock identifies the owning PID; check
that process before removing a stale lock after an ungraceful shutdown.

From the repository root:

```sh
make check PYTHON=python
make response-v2-preflight PYTHON=python
make train-response-v2-awake PYTHON=python
```

The equivalent foreground training command is:

```sh
PYTORCH_ENABLE_MPS_FALLBACK=0 caffeinate -d -i -s -m \
  python -u -m norway_energy_crisis.response_v2.train \
  --stage all --device mps --budget-hours 10
```

Outputs are separate under `norway_energy_crisis/runs/response_refinement_v2/`:
status, budget, immutable provenance, feature schema, support diagnostics, fit
progress and histories, best checkpoints, a frozen selection, forecast arrays,
natural low-load errors, price curves, seed disagreement, paired sequence probes,
plots and `REPORT.md`. A completed run retains no resumable optimizer state.

The simulator API is `ConsumerSimulator(checkpoints, device).forecast(...)` in
`model.py`. Inputs use raw kWh, NOK/kWh, Celsius, consecutive local dataset hours,
known household IDs and a supplied future price matrix. It returns one forecast
per household/hour. An ensemble averages predictions before updating its shared
histories. Inference receives no measured future demand or temperature.

January is development data; February-March is retrospective follow-up because
these periods were previously examined. Forecast accuracy, price sensitivity and
artificial cap responses are reported separately. They do not establish causal
price effects or real consumer behaviour after an outage. Allocation and pricing
policy retraining are outside this experiment. Existing response and RL artifacts
remain untouched.

Longer retrospective forecasts are conditioned on recorded price paths. This
does not mean several future days of market prices were known at the origin;
the one-hour model receives only the candidate price for its current step.

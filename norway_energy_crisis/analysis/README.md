# Data and consumer investigations

These standalone diagnostics explain the dataset and the limits of the frozen
consumer models. They do not change the current pricing reward or retrain a
consumer model unless a separate training command is explicitly run.

| Modules | Purpose |
| --- | --- |
| `audit_training_data.py` | Full-panel coverage, survey and input-data checks |
| `audit_historical_prices.py` | Historical price support and distribution |
| `audit_load_support.py`, `report_load_support.py` | Consumption ranges and low-load model support |
| `consumer_recovery.py`, `report_consumer_recovery.py` | Sudden-curtailment histories and subsequent model forecasts |
| `stress_minimum_supply.py` | Model probes below proposed consumption baselines |
| `investigate_rollout.py`, `report_rollout_households.py` | 72-hour sliding histories, predicted versus measured feedback and household traces |
| `diagnose_price_channels.py` | Immediate and historical price-channel sensitivity |
| `audit_capacity.py`, `report_capacity_audit.py` | Earlier capacity-policy and reward diagnostics |
| `test_*.py` | Generated-fixture checks for the investigations |

For example, from the repository root:

```sh
python -m norway_energy_crisis.analysis.audit_training_data --help
python -m norway_energy_crisis.analysis.audit_historical_prices --help
make consumer-recovery PYTHON=python DEVICE=mps
make response-rollout-audit PYTHON=python
```

`make help-legacy` lists the retained investigation commands. Results remain in
separate ignored folders under `norway_energy_crisis/runs/`; most diagnostics
protect existing output directories. Use a fresh output path to rerun them.
Original source snapshots before this reorganisation are kept locally alongside
the cleanup manifest, so earlier source hashes are not misrepresented.

The shared `minimum_supply.py` remains at its original package path because the
older capacity experiment fingerprints it. It is a numerical scenario mechanism,
not a validated household comfort requirement. Current pricing does not use it.
Consumer recovery after altered histories remains extrapolation without measured
outage-response labels; it is not evidence of real consumer adaptation.

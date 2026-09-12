# Data Pipeline and Model Specifications

This document outlines the data processing pipeline, time series schemas, model architectures, and evaluation methods implemented across the project.

---

## 1. Dataset Overview

The project utilizes smart meter telemetry from the Low Carbon London (LCL) trial conducted by UK Power Networks:
- **Telemetry Scale**: Half-hourly power consumption readings across residential households.
- **Aggregated Feeder Dataset**: 39,727 continuous half-hour intervals spanning from November 2011 to February 2014.
- **Dynamic Tariff Records**: Half-hourly dynamic Time-of-Use (dToU) price schedules extracted from official trial schedules (ranging from 3.99 p/kWh to 67.20 p/kWh).
- **Substation Configuration**: Aggregated local distribution feeder serving a representative 1,000-household cluster with a nominal transformer stress threshold of 0.25 kWh/hh per half-hour.

---

## 2. Data Cleaning and Preprocessing Methodology

The raw telemetry data is processed through an out-of-core pipeline designed to operate with minimal memory overhead:

1. **Lazy Scanning and Type Casting**
   - Raw telemetry CSV records are parsed via Polars lazy frames.
   - Meter identifiers are indexed as categoricals; half-hourly consumption values are parsed as single-precision floats (`Float32`).
   - Timestamps are standardized to ISO datetime format (`%Y-%m-%d %H:%M:%S`).

2. **Feeder Aggregation and Cleansing**
   - Individual meter readings are grouped by half-hourly timestamp.
   - Metrics computed per interval: total load (kWh), mean load per household (`household_mean_kwh`), load standard deviation, and count of active reporting meters.
   - Rows with null readings or incomplete feeder reporting are filtered out.

3. **Dynamic Tariff Ingestion**
   - Trial tariff schedules stored as Excel serial timestamps are converted using the 1899-12-30 epoch baseline.
   - Tariff schedules are joined to the feeder timeline. Missing intervals during non-trial periods default to standard flat rates.

4. **Chronological Splitting (No Shuffling)**
   - Strict chronological splits preserve temporal causality and eliminate lookahead bias:
     - **Training Set**: Start through 2013-05-31 23:30 (approx. 70% of timeline).
     - **Validation Set**: 2013-06-01 00:00 through 2013-09-30 23:30 (approx. 15%).
     - **Test Set**: 2013-10-01 00:00 through 2014-02-28 23:30 (approx. 15%).
   - All scalers (StandardScaler) are fitted strictly on the training partition and applied to validation and test partitions.

---

## 3. Time Series Data Format and Feature Engineering

The input representations are structured around sliding windows of 48 half-hours (24 hours of history):

### A. Pure Forecasting Feature Set (8 Features)
Used to forecast future grid load without price signal exposure:
- `household_mean_kwh`: Historical scaled load values.
- `sin_hh`, `cos_hh`: Trigonometric encoding of the half-hour index (0–47).
- `sin_dow`, `cos_dow`: Trigonometric encoding of the day of the week (0–6).
- `sin_month`, `cos_month`: Trigonometric encoding of the calendar month (1–12).
- `is_weekend`: Binary indicator (1.0 for Saturday/Sunday, 0.0 otherwise).

### B. Consumer Twin Feature Set (9 Features)
Used to model residential demand response to pricing signals:
- All 8 features from the pure forecasting set.
- `Tariff`: Active half-hourly retail tariff (p/kWh) injected at step t.

### C. Wholesale Spot and Grid Economics
- Wholesale energy costs are derived from half-hourly spot price profiles calibrated against UK Day-Ahead power market clearing rates.
- Seasonal multipliers adjust wholesale price severity (Winter: 1.30–1.40x, Summer: 0.85–0.88x).
- Transformer congestion costs follow a quadratic penalty applied to all aggregate demand exceeding the 0.25 kWh/hh substation limit.

---

## 4. Machine Learning Model Specifications

### A. Recurrent Neural Networks (Load Forecasting & Consumer Twin)
Three recurrent architectures are implemented using standard hidden capacity for comparative evaluation:
- **Architectures**:
  - Vanilla RNN (Elman)
  - Long Short-Term Memory (LSTM)
  - Gated Recurrent Unit (GRU)
- **Hyperparameters**:
  - Hidden units: 64
  - Recurrent layers: 2
  - Dropout: 0.15
  - Projection Head: Linear(64, 32) -> ReLU -> Dropout(0.15) -> Linear(32, 1)
  - Optimization: AdamW (learning rate 1e-3, weight decay 1e-4)
  - Criterion: Mean Squared Error (MSE) / Smooth L1 Loss

### B. Fine-Tuned Agentic AI Pricing Controller
- **Base Architecture**: 0.5B parameter autoregressive language model.
- **Fine-Tuning Method**: Low-Rank Adaptation (LoRA) applied to attention projection layers (rank = 8, alpha = 16).
- **Inference Interface**: Generates structured JSON responses containing:
  - `reasoning`: Concise explanation of load expectation and pricing rationale.
  - `predicted_load`: Anticipated half-hourly aggregate consumption.
  - `tariff_p`: Bounded retail tariff in pence/kWh (clamped strictly within [3.99, 67.20]).
- **Behavioral Objective**: Balance operator margin targets (5–15%) against customer price shock thresholds (capped at 25.0 p/kWh for standard operations) while mitigating transformer congestion.

---

## 5. Pricing Controller Regimes

1. **Regime 1: Static Flat**
   - Fixed tariff of 14.23 p/kWh matching the historical baseline.

2. **Regime 2: Static Peak (Time-of-Use)**
   - Two-tier static tariff charging 28.00 p/kWh during evening peak hours (16:00–19:00) and 11.76 p/kWh during off-peak hours.

3. **Regime 3: Twin GRU Dynamic Controller**
   - Classical dual-recurrent setup: a PureGRU neural network forecasts unconstrained feeder load, and a dynamic control policy scales retail prices between 3.99 p/kWh (valley) and 67.20 p/kWh (peak congestion).

4. **Regime 4: GRU + Fine-Tuned Agentic AI Controller (Preemptive)**
   - The Agentic AI controller ingests high-precision neural forecasts from the PureGRU model.
   - Dispatches price signals preemptively (t - 1) before the anticipated demand surge, acting as a physical grid shock absorber to relieve transformer stress.

5. **Regime 5: Pure Agentic AI**
   - The Agentic AI controller infers trends directly from raw historical load observations without an external neural forecaster.

6. **Regime 6: GRU + Fine-Tuned Agentic AI Controller (Advance Broadcast)**
   - Uses the tariff generated from the GRU forecast within a formalized Hour-Ahead market broadcast where the price is announced one step in advance and enacted synchronously at step t.

---

## 6. Plot Data Basis and Simulation Assumptions

### A. Pure Empirical Data Plots (Zero Modeling Assumptions)
The five Exploratory Data Analysis (EDA) visualizations in `plots/eda/` are calculated directly from raw smart meter telemetry (`lcl_tou.csv`) and official trial tariff tables (`Tariffs.xlsx`):
- `01_diurnal_load_profile.png`: Arithmetic mean and standard deviation of actual half-hourly smart meter readings grouped by diurnal half-hour index.
- `02_weekly_seasonal_patterns.png`: Direct calendar groupby of historical consumption by weekday and month.
- `03_tariff_distribution_and_pricing.png`: Direct histogram and frequency counts of contractual trial rates.
- `04_demand_response_high_vs_normal.png`: Direct empirical consumption comparison on days with declared High tariff alerts versus Normal baseline days.
- `05_feeder_timeline_2year.png`: Continuous chronological trace of the raw 39,727 half-hourly observations over 2011–2014.

In the benchmark suite (`plots/6way_benchmark/`), **Regime 1 (Static Flat)** represents the unmanaged empirical baseline demand under the historical standard 14.23 p/kWh London flat tariff.

### B. Engineering Assumptions in the Benchmark Simulation Plots
The remaining trajectories in `plots/6way_benchmark/` simulate dynamic pricing interactions under four documented engineering assumptions:
1. **Substation Transformer Capacity**:
   - The aggregate feeder serves a 1,000-household cluster.
   - The substation transformer overload threshold is fixed at **0.25 kWh/hh per half-hour** (calibrated to the ~80th percentile of baseline feeder demand). Any aggregate demand above 0.25 kWh/hh constitutes equipment thermal overload.
2. **Substation Congestion Penalty Function**:
   - Modeled as a quadratic penalty function:
     $$C_{\text{congestion}} = \gamma \times \left(\max(0, D_t - 0.25)\right)^2 \times 1,000\text{ homes}$$
     where $\gamma = 2.0$ £ per $(\text{kWh/hh})^2$ per 1,000 homes, reflecting $I^2R$ resistive heating and DUoS red-band network congestion charges.
3. **Wholesale Procurement Price Profile**:
   - Modeled using a calibrated half-hourly diurnal profile (`WHOLESALE_HH_BASE`) matching the UK Day-Ahead Power Auction (N2EX / APX UK) for 2012–2014, with seasonal multipliers (Winter: 1.30–1.40x, Summer: 0.85–0.88x, Spring/Autumn: 0.95–1.20x) and Gaussian noise $\epsilon \sim \mathcal{N}(0, 0.05 \cdot W_{\text{base}})$.
4. **Simulated Consumer Demand Response**:
   - When dynamic tariffs are dispatched, the resulting customer demand is computed by the trained Consumer Twin GRU model (`consumer_predict_one()`), which captures real-world human demand inelasticity learned from the trial dataset.

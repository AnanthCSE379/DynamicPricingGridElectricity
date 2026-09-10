# 🧠 Technical Specification: Dual-GRU Neural Architecture & Training Pipeline

This document details the mathematical formulation, architectural design, feature engineering, and training methodologies for the two specialized Gated Recurrent Unit (GRU) networks operating in the autonomous grid ecosystem:
1. **Model A (`PureGRU`)**: Autonomous Feeder-Level Aggregate Load Forecaster (Zero Price Exposure).
2. **Model B (`ConsumerTwin`)**: Consumer Price-Elasticity & Demand Response Digital Twin.

---

## 1. Mathematical Formulation of the Gated Recurrent Unit (GRU)

The GRU cell (Cho et al., 2014) addresses vanishing gradients in standard recurrent networks through gating mechanisms that regulate information flow without a separate memory cell:

For time step $t$, input vector $x_t \in \mathbb{R}^d$, and previous hidden state $h_{t-1} \in \mathbb{R}^h$:

## 1. Architectural Comparison: PureGRU vs. ConsumerTwin

| Parameter | Model A: `PureGRU` (Forecaster) | Model B: `ConsumerTwin` (Digital Twin) |
| :--- | :--- | :--- |
| **Operational Role** | Physical load forecasting for dispatch | Simulates consumer demand response to tariffs |
| **Input Features ($d$)** | **8 features** (Zero price information) | **9 features** (Includes active retail tariff) |
| **Hidden Units ($h$)** | 64 units (2 stacked GRU layers) | 64 units (2 stacked GRU layers) |
| **Dropout** | 0.20 (inter-layer regularization) | 0.20 (inter-layer regularization) |
| **Lookback Window ($L$)** | 48 half-hours (24 hours history) | 48 half-hours (24 hours history) |
| **Output Layer** | Linear projection $\mathbb{R}^{64} \rightarrow \mathbb{R}^1$ | Linear projection $\mathbb{R}^{64} \rightarrow \mathbb{R}^1$ |
| **Post-Activation** | Scaled inverse transform | Scaled inverse transform + Non-negative clamp |
| **Checkpoint File** | `checkpoints/BestPureModel.pt` | `checkpoints/GRU.pt` |

---

## 2. Feature Engineering & Cyclical Encodings

Both networks receive half-hourly aggregate data from **1,000 UK households** over 39,727 intervals. To ensure continuity across temporal boundaries, calendar features are transformed into harmonic sine/cosine pairs:

### Feature Space Definition
1. **Target Feature**: Household mean consumption $y_t$ (normalized via `StandardScaler`).
2. **Diurnal Cycle** (48 half-hour slots per day, $\text{hh\_idx} \in [0, 47]$):
   $$\sin\left(\frac{2\pi \cdot \text{hh\_idx}}{48}\right), \quad \cos\left(\frac{2\pi \cdot \text{hh\_idx}}{48}\right)$$
3. **Weekly Cycle** ($\text{weekday} \in [1, 7]$ where $1=\text{Monday}$):
   $$\sin\left(\frac{2\pi (\text{weekday}-1)}{7}\right), \quad \cos\left(\frac{2\pi (\text{weekday}-1)}{7}\right)$$
4. **Annual Seasonality** ($\text{month} \in [1, 12]$):
   $$\sin\left(\frac{2\pi (\text{month}-1)}{12}\right), \quad \cos\left(\frac{2\pi (\text{month}-1)}{12}\right)$$
5. **Calendar Flag**: $\text{is\_weekend} \in \{0.0, 1.0\}$ (Saturday/Sunday indicator).
6. **Tariff Price Signal ($T_t$)** *(Model B only)*: Dispatched retail price in $\text{p/kWh}$.

---

## 3. Training Methodology & Anti-Leakage Protocol

### Strict Chronological Partitioning
To prevent lookahead bias and temporal leakage, data was split chronologically without shuffling:
- **Training Split**: 26,670 rows ($\approx 70\%$, Nov 23, 2011 to May 31, 2013).
- **Validation Split**: 5,856 rows ($\approx 15\%$, Jun 01, 2013 to Sep 30, 2013).
- **Test Split**: 7,201 rows ($\approx 15\%$, Oct 01, 2013 to Feb 28, 2014).

### Optimization Parameters
- **Objective Loss Function**: Mean Squared Error (MSE):
  $$\mathcal{L}_{\text{MSE}} = \frac{1}{B} \sum_{i=1}^B (y_i - \hat{y}_i)^2$$
- **Optimizer**: Adam ($\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$).
- **Learning Rate**: $\eta = 10^{-3}$ with `ReduceLROnPlateau` (decay factor $0.5$, patience $3$ epochs).
- **Batch Size**: $B = 64$.
- **Gradient Clipping**: Maximum gradient norm $\|\nabla \theta\|_2 \le 1.0$.
- **Normalization Safeguard**: `StandardScaler` fitted **strictly on `train_df`** and serialized to disk (`checkpoints/pure_scaler.pkl` and `checkpoints/scaler.pkl`).

---

## 4. Quantitative Generalization Performance

Evaluated on the 7,201 held-out out-of-sample test half-hours:

| Metric | `PureGRU` (Forecaster) | Vanilla RNN Baseline | LSTM Baseline |
| :--- | :---: | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | **0.0059 kWh/hh** | 0.0094 kWh/hh | 0.0062 kWh/hh |
| **Mean Absolute Error (MAE)** | **0.0041 kWh/hh** | 0.0071 kWh/hh | 0.0044 kWh/hh |
| **Coefficient of Determination ($R^2$)** | **0.984** | 0.941 | 0.981 |
| **Inference Latency (M3 MPS)** | **0.82 ms** | 0.74 ms | 0.96 ms |

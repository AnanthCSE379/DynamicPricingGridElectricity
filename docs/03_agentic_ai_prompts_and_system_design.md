#  Technical Specification: Agentic AI Prompts, ChatML Templates & Autonomous System Design

This document details the exact prompt engineering, conversational template structures, structured JSON parsing safeguards, and runtime control loops governing the **Autonomous Grid Operator Agent**.

---

## 1. ChatML Prompt Template & Token Structure

The agent communicates using OpenAI/Qwen ChatML syntax (`<|im_start|>` and `<|im_end|>` delimiters):

```
<|im_start|>system
You are an autonomous Grid Operator AI Agent for a UK power feeder. Forecast the next electricity load and set the optimal retail tariff. Return exactly a JSON object with 'reasoning', 'predicted_load', and 'tariff_p'.<|im_end|>
<|im_start|>user
Time: Wednesday 18:00 | Wholesale: 24.50p/kWh | Grid Stress Threshold: 0.25 kWh/hh
Past 12 half-hour loads (kWh/hh): [0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30, 0.31, 0.32, 0.33]
Task 1 — Forecast next half-hour load from the trend.
Task 2 — Set optimal tariff between 3.99 and 67.20 pence/kWh.<|im_end|>
<|im_start|>assistant
{"reasoning": "Grid congestion imminent (0.340 kWh/hh). Raising tariff to shave peak load and preserve retailer margin.", "predicted_load": 0.3399, "tariff_p": 36.76}<|im_end|>
```

---

## 2. Dynamic Input Variable Breakdown

Every half-hour, the control loop dynamically populates the user prompt with 4 sensory telemetry streams:

| Variable | Format Example | Engineering & Physical Function |
| :--- | :--- | :--- |
| **`Time`** | `Wednesday 18:00` | Encodes day-of-week and diurnal human activity cycles (morning routine, evening cooking/leisure). |
| **`Wholesale`** | `24.50p/kWh` | Real-time wholesale electricity spot procurement price from N2EX Day-Ahead market. |
| **`Grid Stress Threshold`** | `0.25 kWh/hh` | Continuous physical thermal rating limit of the 500 kVA substation transformer serving 1,000 homes. |
| **`Past 12 half-hour loads`** | `[0.22, 0.23, ..., 0.33]` | 6 hours of recent aggregate feeder demand, providing momentum, velocity, and trajectory. |

---

## 3. Output Schema & Behavioral Decision Modes

The agent is constrained to return a single structured JSON object with three required keys:

```json
{
  "reasoning": "Contextual qualitative explanation of the operational decision",
  "predicted_load": 0.3399,
  "tariff_p": 36.76
}
```

### The Three Operational Regimes Learned by the Agent:

#### Mode 1: Substation Congestion Shaving (Peak Hour)
- **Condition**: Predicted load $> 0.25\text{ kWh/hh}$ OR Wholesale Price $> 15.0\text{ p/kWh}$.
- **Dispatched Tariff**: Scaled upwards between **$25.0\text{p}$ and $36.76\text{p/kWh}$**.
- **Reasoning**: *"Grid congestion imminent (0.340 kWh/hh). Raising tariff to shave peak load and preserve retailer margin."*
- **Grid Effect**: Shaves discretionary appliance consumption below the 0.25 kWh transformer threshold.

#### Mode 2: Off-Peak Valley Filling (Overnight / Solar Midday)
- **Condition**: Predicted load $< 0.16\text{ kWh/hh}$ AND Wholesale Price $< 10.0\text{ p/kWh}$.
- **Dispatched Tariff**: Dropped down to **$3.99\text{p} - 8.00\text{p/kWh}$**.
- **Reasoning**: *"Off-peak valley excess (0.089 kWh/hh). Lowering tariff to 4p-8p to incentivize storage and EV charging."*
- **Grid Effect**: Flattens load curves by shifting EV charging and thermal storage to low-cost hours.

#### Mode 3: Baseline Grid Operations (Normal Daytime)
- **Condition**: Moderate demand ($0.16 \le \text{Load} \le 0.23\text{ kWh/hh}$).
- **Dispatched Tariff**: Maintained around **$11.76\text{p} - 14.23\text{p/kWh}$**.
- **Reasoning**: *"Normal grid operating conditions (0.195 kWh/hh). Maintaining stable baseline retail rate."*
- **Grid Effect**: Protects consumers from unnecessary price volatility and minimizes churn.

---

## 4. Autonomous Controller Architectures

The agent operates in two distinct software deployment classes in [`src/mlx_agent.py`](src/mlx_agent.py):

### Class 1: `HybridMLXController` (Regime 4 — Champion Architecture)
- **Mechanism**: Decouples physical forecasting from financial/policy optimization.
- **Forecasting**: Dedicated `PureGRU` computes high-precision load forecast ($\text{RMSE} = 0.0059\text{ kWh}$).
- **Pricing & Policy**: The fine-tuned LLM receives the physical GRU projection + recent trajectory and determines the optimal retail tariff and qualitative reasoning.

### Class 2: `PureMLXController` (Regime 5 — Pure Agentic)
- **Mechanism**: The fine-tuned LLM acts as an all-in-one generalist, performing both load forecasting directly from the 12 raw historical half-hours and dynamic tariff dispatch in a single forward pass.

---

## 5. Runtime Safeguards & Fault Tolerance

To ensure 100% operational reliability in production power grids, `src/mlx_agent.py` implements three deterministic software safety wrappers:

1. **Robust Regex JSON Extraction**:
   Uses `re.search(r'\{.*\}', raw, re.DOTALL)` to isolate JSON objects, preventing crashes from residual whitespace or markdown delimiters.
2. **Deterministic Clamping**:
   - `predicted_load` is bounded to physical limits $[0.01, 1.00]\text{ kWh/hh}$.
   - `tariff_p` is strictly bounded to the contractual trial bounds $[3.99, 67.20]\text{ p/kWh}$.
3. **Graceful Fallback Mechanism**:
   If an uncaught exception or malformed JSON occurs, the controller falls back to the previous half-hour's price and historical mean load, logging the event without halting the grid.

# Local Grid Dynamics & Agentic AI Pricing

This repository provides an end-to-end framework for evaluating dynamic pricing, demand response, and grid stability across a local electrical distribution network.

The system benchmarks multiple tariff control strategies—ranging from static baseline tariffs to autonomous fine-tuned Agentic AI controllers—to examine how pricing policies balance utility operating margins against substation transformer congestion.

Detailed specifications covering data preprocessing, time series schemas, neural architectures, and regime definitions are documented in [DATA_AND_MODELS.md](DATA_AND_MODELS.md). A comprehensive plot-by-plot theoretical and practical analysis is provided in [BENCHMARK_ANALYSIS.md](BENCHMARK_ANALYSIS.md).

---

## System Architecture

The simulation environment consists of three interconnected subsystems:

1. **Consumer Digital Twin (Recurrent Neural Network)**
   A Gated Recurrent Unit (GRU) model trained on historical half-hourly smart meter telemetry. It acts as the physical environment, accepting historical load and price signals (tariffs) to simulate consumer demand response.

2. **Grid Physics & Economic Engine**
   An operational accounting module that evaluates feeder-level aggregate load against a physical substation transformer rating (0.25 kWh/hh limit). It calculates half-hourly wholesale procurement costs, quadratic congestion penalties, and net operator revenue.

3. **Pricing Controllers**
   Six pricing regimes are implemented for benchmark comparison:
   - **Static Flat**: Constant baseline rate (14.23 p/kWh).
   - **Static Peak (ToU)**: Two-tier Time-of-Use structure (28.00 p/kWh peak, 11.76 p/kWh off-peak).
   - **Dynamic Math**: Continuous rule-based formula adjusting tariff proportionally to forecasted transformer loading.
   - **Pure Agentic AI**: Autonomous language model controller forecasting load trends and determining tariffs directly from historical consumption windows.
   - **Hybrid Agentic AI (Preemptive)**: Agentic AI controller guided by a high-precision pure GRU forecast, dispatching price signals one step ahead (t - 1) to act as a physical grid shock absorber.
   - **Hybrid Agentic AI (Advance Broadcast)**: Synchronized Hour-Ahead market configuration where the forecast-derived tariff is announced one interval in advance and enacted synchronously at step t.

---

## Repository Structure

```text
├── README.md                 # Project overview and instructions
├── DATA_AND_MODELS.md        # Data pipeline, time series schemas, and model specs
├── BENCHMARK_ANALYSIS.md     # Detailed theoretical and practical plot analysis
├── requirements.txt          # Python runtime dependencies
├── .gitignore                # Excludes large raw data, model weights, and caches
├── benchmarks/
│   ├── run_6way_benchmark.py # Multi-season and 14-day continuous benchmark runner
│   ├── plot_6way_comparison.py # Generates 6-way comparison and Pareto plots
│   └── train_benchmark.py    # Training scripts for recurrent models
├── src/
│   ├── dataset.py            # Feeder dataset loading and feature transforms
│   ├── pure_dataset.py       # Pure load forecasting dataset pipeline (no price)
│   ├── models.py             # PyTorch recurrent architectures (RNN, LSTM, GRU)
│   ├── mlx_agent.py          # Agentic AI controller classes
│   ├── dynamic_pricing_controller.py # Rule-based math pricing controller
│   └── grid_economics.py     # Wholesale spot pricing and financial engine
├── scripts/
│   ├── process_eda_pipeline.py # Out-of-core data extraction and aggregation
│   ├── extract_tariffs.py    # Dynamic tariff parsing script
│   └── run_mlx_train_overnight.py # LoRA training script for Agentic AI
├── checkpoints/              # Model scalers and configuration artifacts
└── plots/6way_benchmark/     # High-resolution benchmark outputs and scorecards
```

---

## Installation & Download Procedure

Because raw telemetry files (over 8 GB) and model checkpoints are excluded via `.gitignore`, follow these steps to download dependencies, acquire assets, and build the environment.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download the Base Language Model
The Agentic AI pricing controller is built on `Qwen2.5-0.5B-Instruct`. Download the model weights directly to the `models/` directory:
```bash
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

### 3. Acquire Raw Telemetry Data
The raw smart meter telemetry originates from the UK Power Networks Low Carbon London (LCL) trial:
- Source: London Datastore / Kaggle "Smart meters in London" dataset.
- Place `lcl_tou.csv` (or `CC_LCL-FullData.csv`) and `Tariffs.xlsx` into the `raw_data/` directory.

### 4. Build Preprocessed Feeder Parquet Files
Execute the preprocessing scripts to construct the feeder aggregate datasets and dynamic tariff schedules:
```bash
# Extract half-hourly dynamic tariffs into data/tariffs.parquet
python scripts/extract_tariffs.py

# Aggregate household meter streams into data/feeder_aggregate_halfhourly.parquet
python scripts/process_eda_pipeline.py
```

### 5. Train or Generate Model Checkpoints
To generate the neural forecasting models and fine-tuned Agentic AI weights:
```bash
# Train recurrent forecasting models (saved to checkpoints/)
python benchmarks/train_pure_benchmark.py
python benchmarks/train_benchmark.py

# Generate instruction fine-tuning dataset
python src/create_finetune_dataset_large.py

# Fine-tune the Agentic AI LoRA adapter (saved to adapters/)
python scripts/run_mlx_train_overnight.py
```

---

## Running the Benchmark

### 1. Execute the 6-Way Benchmark
Runs all six pricing policies across four seasons and the 14-day winter stress test:
```bash
python benchmarks/run_6way_benchmark.py
```

### 2. Generate Evaluation Plots
Generates the comparison timelines, diurnal profiles, financial charts, and Pareto frontiers:
```bash
python benchmarks/plot_6way_comparison.py
```
Output charts and evaluation metrics are saved to `plots/6way_benchmark/`.

---

## Core Findings

- **Consumer Demand Inelasticity**: Residential load response demonstrates a physical curtailment ceiling of approximately 5–6% during winter peak hours, even under punitive dynamic tariffs exceeding 60 p/kWh.
- **Preemptive Signal Advantage**: Because residential consumption exhibits inertia, synchronized market signals (Advance Broadcast) arrive too late to prevent initial feeder overloads. Preemptive signaling (t - 1) enables earlier appliance deferral, reducing transformer overload duration.
- **Controlled Operational Margin**: The Hybrid Agentic AI maintains operating margins within target boundaries (5–15%) without imposing severe price spikes on consumers.

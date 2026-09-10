# 📊 Model Leaderboard — LCL Load Forecasting Benchmark

## Overall Test Metrics

| Rank | Model | Params | RMSE (kWh) | MAE (kWh) | MAPE (%) | R² |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **1** | **GRU** | 41,473 | 0.00806 | 0.00604 | 2.91% | 0.98950 |
| **2** | **VanillaRNN** | 15,233 | 0.00893 | 0.00654 | 3.06% | 0.98712 |
| **3** | **LSTM** | 54,593 | 0.01069 | 0.00837 | 4.49% | 0.98152 |

## High Tariff Period Performance (16:00–20:00, 67.20 p/kWh)

| Model | RMSE (kWh) | MAE (kWh) | MAPE (%) | R² |
| :--- | ---: | ---: | ---: | ---: |
| **GRU** | 0.01127 | 0.00885 | 3.69% | 0.98264 |
| **VanillaRNN** | 0.01298 | 0.01014 | 4.64% | 0.97696 |
| **LSTM** | 0.01645 | 0.01345 | 6.31% | 0.96299 |

## Normal Tariff Period Performance

| Model | RMSE (kWh) | MAE (kWh) | MAPE (%) | R² |
| :--- | ---: | ---: | ---: | ---: |
| **GRU** | 0.00771 | 0.00578 | 2.87% | 0.98973 |
| **VanillaRNN** | 0.00835 | 0.00615 | 2.99% | 0.98798 |
| **LSTM** | 0.01015 | 0.00802 | 4.41% | 0.98222 |
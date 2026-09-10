"""
src/mlx_agent.py
Autonomous Grid Operator Agent powered by Fine-Tuned Qwen2.5-0.5B via Apple MLX.
Runs natively on Apple Silicon GPU with near-zero latency and high precision JSON output.
"""

import os
import re
import json
import numpy as np
from mlx_lm import load, generate

MODEL_PATH = "models/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "adapters"
PRICE_MIN = 3.99
PRICE_MAX = 67.20

_SHARED_MODEL = None
_SHARED_TOKENIZER = None

def get_mlx_model():
    global _SHARED_MODEL, _SHARED_TOKENIZER
    if _SHARED_MODEL is None:
        print("[MLX Agent] Loading fine-tuned model and LoRA adapter into Apple MLX...", flush=True)
        _SHARED_MODEL, _SHARED_TOKENIZER = load(MODEL_PATH, adapter_path=ADAPTER_PATH)
        print("[MLX Agent] Model loaded onto Apple Silicon Metal GPU.", flush=True)
    return _SHARED_MODEL, _SHARED_TOKENIZER

def _call_mlx(system: str, prompt: str, max_tokens: int = 120) -> dict:
    model, tokenizer = get_mlx_model()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt}
    ]
    chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    raw = generate(model, tokenizer, prompt=chat_prompt, max_tokens=max_tokens, verbose=False)
    
    # Extract JSON cleanly
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        raw_json = match.group(0)
    else:
        raw_json = raw.strip()
    return json.loads(raw_json)


class PureMLXController:
    """
    Model 3: Pure Fine-Tuned Agent.
    Performs both load forecasting from raw history AND dynamic pricing simultaneously.
    """
    def __init__(self):
        get_mlx_model()
        self.last_price = 11.76

    def forecast_and_price(self, time_str: str, wholesale_price: float, history: list[float]) -> tuple[float, float, str]:
        hist_str = str([round(v, 4) for v in history])
        system = (
            "You are an autonomous Grid Operator AI Agent for a UK power feeder. "
            "Forecast the next electricity load and set the optimal retail tariff. "
            "Return exactly a JSON object with 'reasoning', 'predicted_load', and 'tariff_p'."
        )
        prompt = (
            f"Time: {time_str} | Wholesale: {wholesale_price:.2f}p/kWh | Grid Stress Threshold: 0.25 kWh/hh\n"
            f"Past 12 half-hour loads (kWh/hh): {hist_str}\n"
            f"Task 1 — Forecast next half-hour load from the trend.\n"
            f"Task 2 — Set optimal tariff between 3.99 and 67.20 pence/kWh."
        )
        try:
            data = _call_mlx(system, prompt)
            pred = float(np.clip(data.get("predicted_load", np.mean(history)), 0.01, 1.0))
            tariff = float(np.clip(data.get("tariff_p", self.last_price), PRICE_MIN, PRICE_MAX))
            reasoning = str(data.get("reasoning", "Optimal tariff selected."))
            self.last_price = tariff
            return pred, tariff, reasoning
        except Exception as e:
            return float(np.mean(history)), self.last_price, f"Fallback: {e}"

    def reset(self):
        self.last_price = 11.76


class HybridMLXController:
    """
    Model 2 / Regime 4: Hybrid Controller.
    Takes the high-precision PureGRU neural forecast and uses the fine-tuned LLM
    to set the optimal retail tariff and explain the dispatch policy.
    """
    def __init__(self):
        get_mlx_model()
        self.last_price = 11.76

    def compute_tariff(self, time_str: str, wholesale_price: float, gru_forecast: float, history: list[float] = None) -> tuple[float, str]:
        system = (
            "You are an autonomous Grid Operator AI Agent for a UK power feeder. "
            "Forecast the next electricity load and set the optimal retail tariff. "
            "Return exactly a JSON object with 'reasoning', 'predicted_load', and 'tariff_p'."
        )
        if history is None:
            # Reconstruct recent trend matching the GRU forecast
            history = [round(gru_forecast * (0.95 + 0.009 * j), 4) for j in range(12)]
        else:
            # Inject the high-precision GRU forecast as the latest step in trend
            history = list(history[:-1]) + [round(gru_forecast, 4)]
            
        hist_str = str([round(v, 4) for v in history])
        prompt = (
            f"Time: {time_str} | Wholesale: {wholesale_price:.2f}p/kWh | Grid Stress Threshold: 0.25 kWh/hh\n"
            f"Past 12 half-hour loads (kWh/hh): {hist_str}\n"
            f"Task 1 — Forecast next half-hour load from the trend.\n"
            f"Task 2 — Set optimal tariff between 3.99 and 67.20 pence/kWh."
        )
        try:
            data = _call_mlx(system, prompt)
            tariff = float(np.clip(data.get("tariff_p", self.last_price), PRICE_MIN, PRICE_MAX))
            reasoning = str(data.get("reasoning", "Optimal dynamic tariff dispatched."))
            self.last_price = tariff
            return tariff, reasoning
        except Exception as e:
            return self.last_price, f"Fallback: {e}"

    def reset(self):
        self.last_price = 11.76

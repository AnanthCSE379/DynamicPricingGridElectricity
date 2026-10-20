"""Predict every saved household's next demand from a ten-row demand matrix."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import HISTORY_STEPS, PRICES, make_features
from .model import HouseholdGRU


ROOT = Path(__file__).resolve().parents[1]


def predict(input_values, checkpoint):
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=True)
    household_ids = bundle["household_ids"]
    if input_values.get("household_ids") != household_ids:
        raise ValueError("household_ids must exactly match the checkpoint's saved household order.")
    history = np.asarray(input_values["demand_history_kwh"], dtype=np.float32)
    if history.shape != (HISTORY_STEPS, len(household_ids)):
        raise ValueError(f"Require 10 rows of {len(household_ids)} half-hour kWh readings, oldest first.")
    if np.isinf(history).any() or (history[np.isfinite(history)] < 0).any():
        raise ValueError("Demand readings must be nonnegative; use null for missing inputs.")
    price = float(input_values["next_price_pence_per_kwh"])
    if not np.isfinite(price) or not np.isclose(PRICES, price, rtol=0, atol=1e-5).any():
        raise ValueError("Next price must be an observed LCL tariff: 3.99, 11.76 or 67.20 p/kWh.")
    if "history_timestamps" in input_values:
        timestamps = pd.DatetimeIndex(input_values["history_timestamps"])
        if len(timestamps) != 10 or timestamps.hasnans or not np.all(np.diff(timestamps.asi8) == 30 * 60 * 1_000_000_000):
            raise ValueError("History timestamps must be ten consecutive half-hours, oldest first.")
        if "next_timestamp" in input_values and pd.Timestamp(input_values["next_timestamp"]) != timestamps[-1] + pd.Timedelta(minutes=30):
            raise ValueError("next_timestamp must immediately follow the final history half-hour.")
    means = bundle["household_means"].numpy()
    maxima = bundle["household_maxima"].numpy()
    features = make_features(history.T, means, maxima)
    normalized_price = torch.tensor([(price - float(PRICES[0])) / float(PRICES[-1] - PRICES[0])], dtype=torch.float32)
    model = HouseholdGRU(len(household_ids), bundle["width"])
    model.load_state_dict(bundle["weights"])
    model.eval()
    with torch.no_grad():
        prediction = model(features[None], normalized_price)[0].numpy() * maxima
    return {"household_ids": household_ids, "next_price_pence_per_kwh": price,
            "predicted_next_demand_kwh": prediction.tolist(),
            "predicted_next_average_power_kw": (prediction * 2).tolist(),
            "observed_history_readings_per_household": np.isfinite(history).sum(axis=0).tolist(),
            "total_predicted_next_demand_kwh": float(prediction.sum(dtype=np.float64)),
            "interval_minutes": 30}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "results/five_hour_response/model/gru.pt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = predict(json.loads(args.input.read_text()), args.checkpoint)
    if args.output:
        # Exclusive creation protects a previous custom prediction.
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({"output": str(args.output.resolve()), "households": len(result["household_ids"]),
                          "total_predicted_next_demand_kwh": result["total_predicted_next_demand_kwh"]}, indent=2))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

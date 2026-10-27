"""Transformer household forecasts with explicit input and checkpoint contracts."""

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from norway_energy_crisis.clean_data import require
from norway_energy_crisis.response_v2.data import FEATURES, VERSION, FeatureBuilder


class ConsumerModel(nn.Module):
    """Predict scaled consumption; price dependence has no imposed direction."""

    def __init__(self, static_size, head="direct", use_price=True, initial_mean=0.3):
        super().__init__()
        require(head in ("direct", "persistence"), "Unknown output head.")
        self.config = dict(
            static_size=static_size,
            head=head,
            use_price=use_price,
            initial_mean=float(initial_mean),
        )
        self.projection = nn.Linear(len(FEATURES), 64)
        layer = nn.TransformerEncoderLayer(
            64,
            4,
            128,
            0.1,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        position = torch.arange(24, dtype=torch.float32)[:, None]
        frequency = torch.exp(torch.arange(0, 64, 2) * (-math.log(10000) / 64))
        encoding = torch.zeros(24, 64)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("position", encoding)
        self.head = nn.Sequential(
            nn.Linear(64 + static_size + 8, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        require(initial_mean > 0, "Positive initial mean required.")
        bias = initial_mean + math.log(-math.expm1(-initial_mean))
        nn.init.constant_(self.head[-1].bias, bias if head == "direct" else 0)

    def encode(self, history, price_enabled=True):
        require(history.shape[1:] == (24, len(FEATURES)), "History feature shape.")
        enabled = self.config["use_price"] and price_enabled
        if not enabled:
            mask = torch.ones(len(FEATURES), device=history.device)
            mask[1], mask[10] = 0, 0
            history = history * mask
        return self.encoder(self.projection(history) + self.position[None])[:, -1]

    def predict_encoded(
        self, encoded, history, static, price, target_calendar, price_enabled=True
    ):
        enabled = self.config["use_price"] and price_enabled
        change = price - history[:, -1, 1]
        if not enabled:
            price, change = torch.zeros_like(price), torch.zeros_like(change)
        output = self.head(
            torch.cat(
                (encoded, static, price[:, None], change[:, None], target_calendar), -1
            )
        )[:, 0]
        if self.config["head"] == "persistence":
            previous = history[:, -1, 0].clamp_min(1e-6)
            output = previous + torch.log(-torch.expm1(-previous)) + output
        return F.softplus(output)

    def forward(self, history, static, price, target_calendar, price_enabled=True):
        return self.predict_encoded(
            self.encode(history, price_enabled),
            history,
            static,
            price,
            target_calendar,
            price_enabled,
        )


def append_prediction(history, scaled, price, temperature, target_calendar):
    """Rebuild derived channels from the controlled request, never a delivery."""
    row = torch.cat(
        (
            scaled[:, None],
            price[:, None],
            temperature[:, None],
            target_calendar,
            torch.log1p(scaled)[:, None],
            (price - history[:, -1, 1])[:, None],
            torch.ones_like(scaled[:, None]),
        ),
        -1,
    )
    shifted = torch.cat((history[:, 1:], row[:, None]), 1)
    # The first visible row has no preceding price inside its visible window.
    first = torch.cat((shifted[:, :1, :10], torch.zeros_like(shifted[:, :1, 10:])), -1)
    return torch.cat((first, shifted[:, 1:]), 1)


def rollout(models, inputs, horizon=None, price_enabled=True):
    """Average household requests before advancing one shared household history."""
    models = list(models) if isinstance(models, (list, tuple)) else [models]
    horizon = inputs.prices.shape[1] if horizon is None else horizon
    require(1 <= horizon <= inputs.prices.shape[1], "Invalid rollout length.")
    history = inputs.history
    temperature = history[:, -1, 2]
    predictions = []
    for hour in range(horizon):
        price, target_calendar = inputs.prices[:, hour], inputs.calendar[:, hour]
        scaled = torch.stack(
            [
                model(history, inputs.static, price, target_calendar, price_enabled)
                for model in models
            ]
        ).mean(0)
        predictions.append(scaled * inputs.scales)
        history = append_prediction(
            history, scaled, price, temperature, target_calendar
        )
    return torch.stack(predictions, 1)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def bundle(model, data, **metadata):
    return dict(
        format=VERSION,
        model_config=model.config,
        weights=model.state_dict(),
        preprocessing=data.state,
        data_fingerprint=data.fingerprint,
        feature_order=FEATURES,
        **metadata,
    )


def load_model(checkpoint, device="cpu"):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require(saved.get("format") == VERSION, "Not a v2 consumer checkpoint.")
    require(saved["feature_order"] == FEATURES, "Checkpoint feature mismatch.")
    FeatureBuilder(saved["preprocessing"])
    model = ConsumerModel(**saved["model_config"])
    model.load_state_dict(saved["weights"])
    return model.to(device).eval().requires_grad_(False), saved


class ConsumerSimulator:
    """Infer household consumption in kWh from raw-unit histories and price paths.

    Multiple checkpoints must share the same schema and scaling. Each hourly
    ensemble average is appended before the next prediction is computed.
    """

    def __init__(self, checkpoints, device="cpu"):
        self.models, self.bundles = [], []
        for checkpoint in checkpoints:
            model, saved = load_model(checkpoint, device)
            if self.bundles:
                require(
                    saved["preprocessing"] == self.bundles[0]["preprocessing"],
                    "Ensemble preprocessing mismatch.",
                )
            self.models.append(model)
            self.bundles.append(saved)
        require(bool(self.models), "No response checkpoints.")
        self.features = FeatureBuilder(self.bundles[0]["preprocessing"])
        self.device = device

    @torch.inference_mode()
    def forecast(
        self,
        demand,
        prices,
        temperature,
        times,
        household_ids,
        candidate_prices,
        batch_size=128,
    ):
        inputs = self.features.inputs(
            demand,
            prices,
            temperature,
            times,
            household_ids,
            candidate_prices,
        )
        predictions = []
        for start in range(0, len(household_ids), batch_size):
            values = rollout(
                self.models,
                inputs.slice(start, start + batch_size).to(self.device),
            )
            predictions.append(values.cpu().numpy())
        result = np.concatenate(predictions)
        require(np.isfinite(result).all() and (result >= 0).all(), "Invalid forecast.")
        return result

"""Shared household features and MLP/Transformer temporal encoders for SB3."""

import math
import random
import time

import numpy as np
import torch
from torch import nn
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib import MaskablePPO

from norway_energy_crisis.clean_data import require
from .replay import CompactReplayBuffer


class CapacityFeatures(BaseFeaturesExtractor):
    """All policies receive the same observations; only the temporal encoder differs."""

    def __init__(self, observation_space, area_index, transformer=False):
        super().__init__(observation_space, features_dim=128)
        self.register_buffer("areas", torch.as_tensor(area_index, dtype=torch.long))
        household_size = 48 + observation_space["static"].shape[1] + 6
        self.household = nn.Sequential(
            nn.Linear(household_size, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU()
        )
        self.transformer = transformer
        if transformer:
            self.projection = nn.Linear(12, 64)
            layer = nn.TransformerEncoderLayer(
                64, 4, 128, dropout=0.0, batch_first=True
            )
            self.temporal = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
            positions = torch.arange(24, dtype=torch.float32)[:, None]
            frequencies = torch.exp(torch.arange(0, 64, 2) * (-math.log(10000.0) / 64))
            encoding = torch.zeros(24, 64)
            encoding[:, 0::2] = torch.sin(positions * frequencies)
            encoding[:, 1::2] = torch.cos(positions * frequencies)
            self.register_buffer("positions", encoding, persistent=False)
        else:
            self.temporal = nn.Sequential(
                nn.Flatten(),
                nn.Linear(24 * 12, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
        self.combine = nn.Sequential(nn.Linear(128 + 64 + 240 + 8, 128), nn.ReLU())

    def forward(self, observations):
        inputs = torch.cat(
            [
                observations[name]
                for name in ("demand", "reference", "static", "accounts")
            ],
            dim=-1,
        )
        household = self.household(inputs)
        pooled = []
        for area in range(2):
            chosen = household[:, self.areas == area]
            pooled.extend((chosen.mean(1), chosen.amax(1)))
        history = observations["temporal"]
        if self.transformer:
            temporal = self.temporal(self.projection(history) + self.positions[None])[
                :, -1
            ]
        else:
            temporal = self.temporal(history)
        return self.combine(
            torch.cat(
                (
                    *pooled,
                    temporal,
                    observations["known"].flatten(1),
                    observations["context"],
                ),
                dim=1,
            )
        )


class SlotClock:
    """Consumed compute includes evaluation/serialization and survives resumption."""

    def __init__(self, seconds, configuration, consumed=0.0):
        require(seconds > 0, "Positive training slot required.")
        self.seconds, self.configuration, self.consumed = (
            float(seconds),
            configuration,
            float(consumed),
        )
        self.started = time.monotonic()

    @property
    def elapsed(self):
        return self.consumed + time.monotonic() - self.started

    def learning_rate(self, _=None):
        start = 1e-4 if self.configuration == "dqn" else 3e-4
        fraction = np.clip(self.elapsed / self.seconds, 0.0, 1.0)
        return float(start * (1 - 0.9 * fraction))

    def epsilon(self, _=None):
        return float(1 - 0.95 * np.clip(self.elapsed / (0.3 * self.seconds), 0.0, 1.0))


def algorithm(configuration):
    require(
        configuration in ("dqn", "ppo_mlp", "ppo_transformer"),
        "Unknown RL configuration.",
    )
    return DQN if configuration == "dqn" else MaskablePPO


def attach_schedules(model, clock):
    model.lr_schedule = clock.learning_rate
    if isinstance(model, DQN):
        model.exploration_schedule = clock.epsilon
        model.exploration_rate = clock.epsilon()


def make_agent(env, configuration, seed, device, clock, overrides=None):
    require(device in ("cpu", "mps"), "Explicit cpu/mps device required.")
    if device == "mps":
        require(torch.backends.mps.is_available(), "MPS unavailable; no CPU fallback.")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "mps":
        torch.mps.manual_seed(seed)
    policy_kwargs = dict(
        features_extractor_class=CapacityFeatures,
        features_extractor_kwargs=dict(
            area_index=env.area_index.tolist(),
            transformer=configuration == "ppo_transformer",
        ),
        net_arch=[128, 64],
        activation_fn=nn.ReLU,
    )
    settings = dict(
        learning_rate=clock.learning_rate,
        batch_size=64,
        gamma=1.0,
        policy_kwargs=policy_kwargs,
        seed=seed,
        device=device,
        verbose=0,
    )
    if configuration == "dqn":
        settings.update(
            replay_buffer_class=CompactReplayBuffer,
            buffer_size=3000,
            learning_starts=2400,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=1000,
            max_grad_norm=10.0,
        )
    else:
        settings.update(
            n_steps=192,
            n_epochs=5,
            gae_lambda=0.95,
            clip_range=0.2,
            vf_coef=0.5,
            ent_coef=0.0,
            max_grad_norm=0.5,
            target_kl=0.03,
        )
    settings.update(overrides or {})
    model = algorithm(configuration)("MultiInputPolicy", env, **settings)
    attach_schedules(model, clock)
    return model


def choose_action(policy, env, observation):
    """Pass masks at inference as well as during SB3-Contrib rollout collection."""
    if isinstance(policy, MaskablePPO):
        return int(
            policy.predict(
                observation, deterministic=True, action_masks=env.action_masks()
            )[0]
        )
    if hasattr(policy, "predict"):
        return int(policy.predict(observation, deterministic=True)[0])
    return int(policy(env, observation))

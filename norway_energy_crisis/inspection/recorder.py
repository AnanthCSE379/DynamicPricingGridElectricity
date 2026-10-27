"""Observe Gymnasium episodes without changing observations, actions or rewards."""

import gymnasium as gym
import numpy as np

from norway_energy_crisis.dqn.environment import bounded, REWARD_SCALE, JUMP_RANGE
from .store import EpisodeStore


class EpisodeRecorder(gym.Wrapper):
    """Record every household/action to SQLite before VecEnv can reset the day.

    Training recovery retains discarded interactions but excludes them from the
    default learning curves. Recording consumes time inside the training budget.
    No model inference, random draws, reward shaping or action selection occurs.
    """

    def __init__(self, env, folder, manifest, kind="training"):
        super().__init__(env)
        if kind not in ("training", "checkpoint_replay"):
            raise ValueError("Unknown episode provenance.")
        self.store = EpisodeStore(folder, {"schema": 1, "kind": kind, **manifest})
        self.kind = kind
        self.model = self.clock = None
        self.episode_id = None
        self.local_decisions = 0
        self.ready = False

    def __getattr__(self, name):
        # Existing training/checkpoint code accesses the base environment's hour,
        # feature_shapes and service accounts. Keep that interface unchanged.
        return getattr(self.env, name)

    def bind(self, model, clock=None):
        self.model, self.clock = model, clock

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        self._begin()
        return result

    def _begin(self):
        self.episode_id = None
        self.ready = True
        env = self.env
        preprocessor = env.provider.preprocessor
        if hasattr(env, "inspection_context"):
            self.episode_metadata, self.context_arrays = env.inspection_context()
            self.capacity = bool(self.episode_metadata["allocation_enabled"])
            return
        self.context_arrays = dict(
            household_ids=np.asarray(preprocessor.ids).astype(str),
            regions=np.asarray(env.provider.area_index, dtype=np.int16),
            initial_history=np.asarray(
                env.history[:, :, 0] * preprocessor.scales[:, None], dtype=np.float32
            ),
            reference=np.asarray(env.reference, dtype=np.float32),
            market_prices=np.asarray(env.episode.future_prices, dtype=np.float32),
            grid=np.asarray(env.grid),
            capacity_kw=np.asarray(
                getattr(getattr(env, "scenario", None), "capacity_kw", [])
            ),
        )
        self.capacity = bool(
            getattr(env, "allocation_enabled", hasattr(env, "scenario"))
        )
        self.episode_metadata = dict(
            origin=env.episode.timestamp,
            objective=(
                "terminal service loss" if self.capacity else "four-term demand shaping"
            ),
            reward_timing=(
                "Terminal whole-day service cost"
                if self.capacity
                else "Hourly demand/jump; terminal whole-day energy/charge"
            ),
            scenario=env.scenario.name if self.capacity else "no allocation",
            alpha=env.scenario.alpha if self.capacity else None,
            interval_hours=env.scenario.interval_hours if self.capacity else 1.0,
            source="Frozen response-model predictions; reference is also simulated.",
            regions=list(preprocessor.state["areas"]),
            model_engine=getattr(env.engine, "signature", "unknown"),
            horizon=len(env.reference),
            allocation_enabled=self.capacity,
        )

    def action_masks(self):
        if hasattr(self.env, "action_masks"):
            return self.env.action_masks()
        return np.ones(self.action_space.n, dtype=bool)

    def step(self, action):
        if not self.ready:
            raise RuntimeError("Reset the recorder before stepping.")
        env = self.env
        previous_price = env.previous_regional.copy()
        if self.capacity:
            prior_delivery = env.delivered_energy.copy()
        result = env.step(action)
        observation, reward, terminated, truncated, info = result
        finished = terminated or truncated
        price = float(env.prices[-1])
        if "telemetry" in info:
            telemetry = info["telemetry"]
            raw = np.asarray(telemetry["arrays"]["raw"])
            requested = np.asarray(telemetry["arrays"]["requested"])
            delivered = np.asarray(telemetry["arrays"]["delivered"])
            essential = np.asarray(telemetry["arrays"]["essential"])
            components = dict(info["reward_components"])
        elif self.capacity:
            raw = np.asarray(env.raw_predictions[-1])
            essential = env.scenario.alpha * env.reference[env.hour - 1]
            requested = np.maximum(raw, essential)
            delivered = env.delivered_energy - prior_delivery
            components = {"service": float(reward)}
        else:
            raw = requested = delivered = np.asarray(env.demands[-1])
            essential = np.zeros_like(requested)
            deviation = max(
                abs(np.asarray(requested, dtype=np.float64).sum() - env.target)
                / max(env.target, 1e-6)
                - 0.1,
                0,
            )
            jump = (
                np.minimum(
                    np.abs(price - previous_price[env.provider.area_index])
                    / JUMP_RANGE,
                    1,
                )
                ** 2
            )
            components = dict(
                demand=-REWARD_SCALE * float(bounded(deviation)) / (24 * 3.1),
                jump=-REWARD_SCALE * 0.1 * float(jump.mean()) / (24 * 3.1),
                energy=0.0,
                charge=0.0,
            )
            if terminated:
                components["energy"] = (
                    -REWARD_SCALE
                    * float(
                        bounded(
                            np.abs(env.energy - env.energy0)
                            / np.maximum(env.energy0, 1e-6)
                        ).mean()
                    )
                    / 3.1
                )
                components["charge"] = (
                    -REWARD_SCALE
                    * float(
                        bounded(
                            np.maximum((env.charge - env.charge0) / env.denom, 0)
                        ).mean()
                    )
                    / 3.1
                )
        if not np.isclose(sum(components.values()), reward, atol=1e-6, rtol=1e-6):
            raise ValueError(
                "Recorded component accounting does not match environment reward."
            )
        if self.episode_id is None:
            self.episode_id = self.store.start(
                self.episode_metadata, self.context_arrays, self.kind
            )
        self.local_decisions += 1
        decision = (
            int(self.model.num_timesteps) + 1
            if self.model is not None
            else self.local_decisions
        )
        debug = dict(
            updates=int(self.model._n_updates) if self.model is not None else None,
            epsilon=(
                float(self.model.exploration_rate)
                if self.model is not None and hasattr(self.model, "exploration_rate")
                else None
            ),
            seconds=float(self.clock.elapsed) if self.clock is not None else None,
            learning_rate=(
                float(self.clock.learning_rate()) if self.clock is not None else None
            ),
            floor_binding_fraction=(
                float(np.mean(raw < essential)) if self.capacity else 0.0
            ),
            action_valid=bool(self.action_masks()[int(action)]),
        )
        debug.update(info.get("telemetry", {}).get("diagnostics", {}))
        arrays = {
            name: np.asarray(value, dtype=np.float32)
            for name, value in dict(
                requested=requested, delivered=delivered, raw=raw, essential=essential
            ).items()
        }
        arrays.update(
            {
                name: np.asarray(value, dtype=np.float32)
                for name, value in info.get("telemetry", {}).get("arrays", {}).items()
            }
        )
        self.store.append(
            self.episode_id,
            int(env.hour),
            decision,
            int(action),
            price,
            float(reward),
            components,
            debug,
            arrays,
            finished,
            context_arrays=info.get("telemetry_context"),
        )
        return result

    def snapshot(self):
        if self.env.hour != 0:
            raise ValueError(
                "Recorded training checkpoints require an episode boundary."
            )
        state = self.env.snapshot()
        state["telemetry_cursor"] = self.store.cursor
        return state

    def restore(self, state):
        self.env.restore(state)
        if "telemetry_cursor" in state:
            self.store.rollback_after(state["telemetry_cursor"])
        self._begin()

    def close(self):
        self.store.close()
        return self.env.close()

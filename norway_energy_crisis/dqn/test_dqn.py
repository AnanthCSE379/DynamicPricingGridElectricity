"""Test information boundaries, accounting, ensemble dynamics and DQN recovery."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env

from norway_energy_crisis.clean_data import save_json
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.test_pipeline import fixture
from norway_energy_crisis.train_response import (
    ResponseModel,
    atomic_torch_save,
    checkpoint,
)
from .environment import (
    EpisodeProvider,
    PriceEnv,
    TERM_NAMES,
    price_grid,
    price_transitions,
    score,
)
from .evaluation import bootstrap, evaluate
from . import train


class DQNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, prepared = fixture(cls.root)
        cls.data = PreparedData(prepared)
        cls.responses = cls.root / "responses"
        for seed in (42, 7, 2026):
            for kind in (["price", "no_price"] if seed != 2026 else ["price"]):
                torch.manual_seed(seed)
                model = ResponseModel(
                    "Transformer", 24, cls.data.static.shape[1], kind == "price"
                )
                torch.nn.init.normal_(model.head[-1].weight, std=0.008)
                folder = cls.responses / f"transformer_24h_seed{seed}_{kind}"
                folder.mkdir(parents=True)
                atomic_torch_save(
                    checkpoint(model, cls.data, {}, 6, 1.0), folder / "best.pt"
                )
        cls.engine = train.make_engine(cls.responses, "ensemble", cls.data, "cpu")
        cls.provider = EpisodeProvider(cls.data, "train")
        cls.cache = cls.root / "cache"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def env(self, cap=3.0, engine=None, provider=None):
        return PriceEnv(
            provider or self.provider, engine or self.engine, self.cache, cap
        )

    def test_action_grids_bounds_and_regional_entry(self):
        self.assertEqual(len(price_grid(2.0)), 21)
        self.assertEqual(len(price_grid(3.0)), 31)
        np.testing.assert_array_equal(price_grid(2.0), price_grid(3.0)[:21])
        self.assertEqual(price_grid(3.0)[-1], 3.0)
        with self.assertRaises(ValueError):
            price_grid(4.0)
        jumps, switches = price_transitions(
            np.full(24, 0.5), [-1.0, 4.0], np.array([0, 0, 1])
        )
        expected = ((1.5 / 2.95) ** 2 * 2 + 1) / 3
        self.assertAlmostEqual(jumps[0], expected)
        self.assertEqual(switches["internal_switches"], 0)
        self.assertEqual(switches["run_durations_hours"], [24])
        self.assertEqual(switches["entry_switch_fraction"], 1.0)

    def test_partition_and_sealed_response_boundaries(self):
        for split, code in [("train", 0), ("validation", 1)]:
            provider = EpisodeProvider(self.data, split)
            for episode in provider.episodes:
                self.assertTrue(
                    np.all(
                        self.data.partitions[episode.origin - 24 : episode.origin + 24]
                        == code
                    )
                )
            self.assertFalse(hasattr(provider, "data"))
            self.assertFalse(hasattr(provider.episodes[0], "target"))
            self.assertEqual(provider.episodes[0].temperature_history.shape, (2, 24))
        with self.assertRaises(ValueError):
            EpisodeProvider(self.data, "test")
        with self.assertRaises(ValueError):
            train.make_engine(self.responses, "reserved", self.data, "cpu")
        seal = self.root / "test_selection.json"
        save_json(seal, dict(frozen_before_test=True))
        self.assertTrue(EpisodeProvider(self.data, "test", seal).episodes)
        self.assertEqual(
            len(
                train.make_engine(
                    self.responses, "reserved", self.data, "cpu", seal
                ).models
            ),
            1,
        )

    def test_future_demand_and_weather_are_not_read(self):
        class Guard:
            def __init__(self, array):
                self.array = array

            def __getitem__(self, key):
                hour_indices = key[1]
                if np.any(np.asarray(hour_indices) >= 48):
                    raise AssertionError("Future measurement accessed")
                return self.array[key]

        origin = next(
            index
            for index in self.data.origins["train"]
            if index % 24 == 0 and index >= 48
        )
        # This construction allows history 24:48 but rejects every target 48:72.
        self.assertEqual(origin, 48)
        with patch.object(self.data, "demand", Guard(self.data.demand)), patch.object(
            self.data, "temperature", Guard(self.data.temperature)
        ), patch.object(
            self.data, "origins", {"train": np.array([origin])}
        ), patch.object(
            self.data, "batch", side_effect=AssertionError("Unsafe batch helper called")
        ):
            provider = EpisodeProvider(self.data, "train")
            env = self.env(provider=provider)
            observation, _ = env.reset(options={"episode_index": 0})
            temperature = env.history[:, -1, 2].copy()
            for _ in range(24):
                observation, _, _, _, _ = env.step(4)
                np.testing.assert_array_equal(env.history[:, -1, 2], temperature)

    def test_shared_mean_recursion_frozen_models_and_candidate_heads(self):
        episode = self.provider.episodes[0]
        history = self.provider.features(episode)
        first = train.make_engine(self.responses, "s42", self.data, "cpu")
        second = train.make_engine(self.responses, "s7", self.data, "cpu")
        expected = (
            first.predict(history, 1.0, episode.future_calendar[0])
            + second.predict(history, 1.0, episode.future_calendar[0])
        ) / 2
        actual = self.engine.predict(history, 1.0, episode.future_calendar[0])
        np.testing.assert_allclose(actual, expected, rtol=1e-6)
        candidates = self.engine.predict_candidates(
            history, price_grid(3.0), episode.future_calendar[0]
        )
        for action_index in (0, 10, 30):
            np.testing.assert_allclose(
                candidates[action_index],
                self.engine.predict(
                    history, price_grid(3.0)[action_index], episode.future_calendar[0]
                ),
                rtol=1e-5,
            )
        before = [
            {name: tensor.clone() for name, tensor in model.state_dict().items()}
            for model in self.engine.models
        ]
        env = self.env()
        env.reset(options={"episode_index": 0})
        env.step(10)
        np.testing.assert_allclose(
            env.history[:, -1, 0], actual / self.engine.scales, rtol=1e-5
        )
        for model, weights in zip(self.engine.models, before):
            self.assertFalse(model.training)
            self.assertTrue(
                all(not parameter.requires_grad for parameter in model.parameters())
            )
            for name, tensor in model.state_dict().items():
                torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)

    def test_gym_exact_rewards_zero_negative_reference_and_no_switch_cost(self):
        env = self.env()
        check_env(env, warn=True, skip_render_check=True)
        env.reset(options={"episode_index": 1})
        rewards = []
        for hour in range(24):
            observation, reward, done, truncated, info = env.step((hour * 7) % 31)
            rewards.append(reward)
            self.assertTrue(env.observation_space.contains(observation))
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertAlmostEqual(sum(rewards), -100 * info["cost"], places=6)
        self.assertEqual(set(info["terms"]), set(TERM_NAMES))
        manual = (
            info["terms"]["demand"]
            + info["terms"]["energy"]
            + 0.1 * info["terms"]["jump"]
            + info["terms"]["charge"]
        ) / 3.1
        self.assertAlmostEqual(manual, info["cost"])
        self.assertGreater(info["stats"]["internal_switches"], 0)
        episode = self.provider.episodes[0]
        zero = score(
            np.zeros((24, 2)),
            np.full(24, 0.05),
            np.zeros((24, 2)),
            episode,
            self.provider.area_index,
            0.5,
        )
        self.assertEqual(zero["terms"]["demand"], 0.0)
        self.assertTrue(np.isfinite(zero["cost"]))
        negative = episode.future_prices.copy()
        negative[:] = -0.1
        from dataclasses import replace

        result = score(
            np.ones((24, 2)),
            np.full(24, 0.05),
            np.ones((24, 2)),
            replace(episode, future_prices=negative),
            self.provider.area_index,
            0.5,
        )
        self.assertTrue(0 <= result["cost"] <= 1)

    def test_same_trajectory_scores_identically_across_caps(self):
        first, second = self.env(2.0), self.env(3.0)
        first.reset(options={"episode_index": 0})
        second.reset(options={"episode_index": 0})
        for hour in range(24):
            action = hour % 21
            left = first.step(action)
            right = second.step(action)
            self.assertAlmostEqual(left[1], right[1], places=8)
        self.assertAlmostEqual(left[-1]["cost"], right[-1]["cost"], places=10)

    def test_price_free_physics_and_candidate_invariance(self):
        engine = train.make_engine(self.responses, "control", self.data, "cpu")
        first, second = self.env(engine=engine), self.env(engine=engine)
        first.reset(options={"episode_index": 0})
        second.reset(options={"episode_index": 0})
        for _ in range(24):
            first.step(0)
            second.step(30)
        np.testing.assert_allclose(first.demands, second.demands, rtol=0, atol=0)
        self.assertNotEqual(first.charge.sum(), second.charge.sum())

    def test_explicit_day_enumeration_and_paired_blocks(self):
        provider = EpisodeProvider(self.data, "validation")
        provider.episodes = provider.episodes[:3]
        env = self.env(provider=provider)
        result = evaluate(env, lambda episode, obs: 5)
        self.assertEqual(
            [day["origin"] for day in result["days"]],
            [episode.timestamp for episode in provider.episodes],
        )
        outcome = bootstrap(np.arange(58) + 1, np.arange(58) + 2, block=7)
        np.testing.assert_allclose(outcome["ci95"], [-1.0, -1.0])
        self.assertEqual(outcome["days"], 58)

    def test_clock_endpoints_and_resume_after_real_dqn_updates(self):
        clock = train.SlotClock(100)
        with patch(
            "norway_energy_crisis.dqn.train.time.monotonic", return_value=clock.started
        ):
            self.assertAlmostEqual(clock.learning_rate(), 1e-4)
            self.assertAlmostEqual(clock.epsilon(), 1.0)
        with patch(
            "norway_energy_crisis.dqn.train.time.monotonic",
            return_value=clock.started + 100,
        ):
            self.assertAlmostEqual(clock.learning_rate(), 1e-5)
            self.assertAlmostEqual(clock.epsilon(), 0.05)
        env = self.env()
        clock = train.SlotClock(1000)
        model = train.make_agent(
            env, "cpu", 42, clock, dict(buffer_size=64, learning_starts=8, batch_size=8)
        )
        model.learn(total_timesteps=24)
        self.assertGreater(model._n_updates, 0)
        observation = env.observation()
        expected = model.predict(observation, deterministic=True)[0]
        path = self.root / "resume.pkl"
        state = dict(signature={"test": 1})
        train.save_resume(path, model, env, clock, state, "cpu")
        restored_env = self.env()
        restored, resumed_clock, _ = train.load_resume(
            path, restored_env, 1000, {"test": 1}, "cpu"
        )
        self.assertEqual(restored.num_timesteps, model.num_timesteps)
        self.assertEqual(restored._n_updates, model._n_updates)
        self.assertEqual(restored.replay_buffer.pos, model.replay_buffer.pos)
        self.assertEqual(
            int(restored.predict(restored_env.observation(), deterministic=True)[0]),
            int(expected),
        )
        for key in observation:
            np.testing.assert_array_equal(
                observation[key], restored_env.observation()[key]
            )
        self.assertGreaterEqual(resumed_clock.elapsed, clock.elapsed - 0.5)
        restored.learn(total_timesteps=24, reset_num_timesteps=False)
        self.assertEqual(restored.num_timesteps, 48)
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for parameter in restored.policy.parameters()
            )
        )
        with self.assertRaises(ValueError):
            train.load_resume(path, self.env(), 1000, {"different": 1}, "cpu")

    def test_fit_interruption_recovers_budget_and_finishes(self):
        provider = EpisodeProvider(self.data, "validation")
        provider.episodes = provider.episodes[:2]
        output = self.root / "interrupted_fit"
        original = train.DQN.learn
        calls = 0

        def interrupt(model, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise KeyboardInterrupt("Simulated interrupted training")
            return original(model, *args, **kwargs)

        options = dict(
            validation_seconds=1.0,
            checkpoint_seconds=0.001,
            overrides=dict(buffer_size=64, learning_starts=8, batch_size=8),
        )
        with patch.object(train.DQN, "learn", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                train.fit(
                    self.provider,
                    provider,
                    self.engine,
                    output,
                    3.0,
                    42,
                    2.0,
                    self.cache,
                    "cpu",
                    **options,
                )
        self.assertTrue((output / "resume.pkl").is_file())
        result = train.fit(
            self.provider,
            provider,
            self.engine,
            output,
            3.0,
            42,
            2.0,
            self.cache,
            "cpu",
            **options,
        )
        self.assertGreaterEqual(result["decisions"], 24)
        self.assertGreater(result["elapsed_seconds"], 1.0)
        self.assertFalse((output / "resume.pkl").exists())

    def test_full_six_fit_workflow_seals_before_reserved_and_test(self):
        output = self.root / "full_run"
        original = train.make_engine

        def guarded(root, name, data, device, selection_file=None):
            if name == "reserved":
                self.assertTrue((output / "selection.json").is_file())
            return original(root, name, data, device, selection_file)

        with patch.object(train, "make_engine", guarded):
            train.run(
                self.data.folder,
                self.responses,
                output,
                budget_hours=0.003,
                device="cpu",
                validation_seconds=2.0,
                checkpoint_seconds=1.0,
                overrides=dict(buffer_size=64, learning_starts=8, batch_size=8),
                limit_days=2,
            )
        self.assertEqual(
            json.loads((output / "status.json").read_text())["state"], "complete"
        )
        self.assertEqual(len(list(output.glob("cap*/best.zip"))), 6)
        self.assertEqual(len(list(output.glob("cap*/resume.pkl"))), 0)
        self.assertTrue((output / "comparison.json").is_file())
        train.run(
            self.data.folder,
            self.responses,
            output,
            budget_hours=0.003,
            device="cpu",
            validation_seconds=2.0,
            checkpoint_seconds=1.0,
            overrides=dict(buffer_size=64, learning_starts=8, batch_size=8),
            limit_days=2,
        )

    @unittest.skipUnless(
        torch.backends.mps.is_available(), "MPS not visible in this process"
    )
    def test_mps_parity_and_dqn_update(self):
        engine = train.make_engine(self.responses, "ensemble", self.data, "mps")
        train.parity(self.data, engine, self.provider, self.responses)
        env = self.env(engine=engine)
        model = train.make_agent(
            env,
            "mps",
            42,
            train.SlotClock(60),
            dict(buffer_size=64, learning_starts=8, batch_size=8),
        )
        model.learn(total_timesteps=24)
        self.assertGreater(model._n_updates, 0)
        self.assertTrue(
            all(
                parameter.device.type == "mps"
                for parameter in model.policy.parameters()
            )
        )
        path = self.root / "mps_resume.pkl"
        state = dict(signature={"device": "mps"})
        train.save_resume(path, model, env, train.SlotClock(600), state, "mps")
        restored_env = self.env(engine=engine)
        restored, _, _ = train.load_resume(
            path, restored_env, 600, state["signature"], "mps"
        )
        restored.learn(total_timesteps=24, reset_num_timesteps=False)
        self.assertEqual(restored.num_timesteps, 48)
        self.assertTrue(
            all(
                torch.isfinite(parameter).all()
                for parameter in restored.policy.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()

"""Conservation, information boundaries and learner recovery on generated data."""

import contextlib
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.logger import configure

from norway_energy_crisis.clean_data import save_json
from norway_energy_crisis.dqn.environment import EpisodeProvider, append_history
from norway_energy_crisis.dqn.train import make_engine
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.test_pipeline import fixture
from norway_energy_crisis.train_response import (
    ResponseModel,
    checkpoint,
    atomic_torch_save,
)
from .allocation import allocate, service, jain
from .environment import CapacityPriceEnv, arrival_order
from .evaluation import (
    HistoricalReplay,
    average_results,
    evaluate,
    paired_daily,
    prefer,
    read_json,
)
from .policies import SlotClock, algorithm, choose_action, make_agent
from .scenarios import CONFIGURATIONS, RUN_ORDER, CapacityScenario, calibrate
from .replay import CompactReplayBuffer
from . import train


class CapacityTests(unittest.TestCase):
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
            for kind in (("price", "no_price") if seed != 2026 else ("price",)):
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
        cls.engine = make_engine(cls.responses, "ensemble", cls.data, "cpu")
        cls.provider = EpisodeProvider(cls.data, "train")
        cls.provider.episodes = cls.provider.episodes[:2]
        cls.validation = EpisodeProvider(cls.data, "validation")
        cls.validation.episodes = cls.validation.episodes[:2]
        cls.scenarios = calibrate(cls.data)
        cls.cache = cls.root / "cache"

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def env(self, scenarios=None, engine=None, provider=None, method="fair"):
        return CapacityPriceEnv(
            provider or self.provider,
            engine or self.engine,
            scenarios or self.scenarios,
            self.cache,
            method,
        )

    def test_allocation_conservation_and_essential_first(self):
        rng = np.random.default_rng(91)
        for method in ("fair", "proportional", "fcfs"):
            for _ in range(30):
                requests = rng.uniform(0, 10, 30)
                essential = 0.5 * requests
                budget = float(rng.uniform(0, requests.sum() * 1.2))
                result = allocate(
                    requests,
                    essential,
                    budget,
                    method=method,
                    order=rng.permutation(len(requests)),
                )
                self.assertTrue((result.delivered >= 0).all())
                self.assertTrue((result.delivered <= requests + 1e-8).all())
                self.assertLessEqual(result.delivered.sum(), budget + 1e-8)
                self.assertAlmostEqual(
                    result.delivered.sum() + result.remaining_kwh, budget
                )
                self.assertAlmostEqual(
                    result.delivered.sum(), min(budget, requests.sum())
                )
                if essential.sum() <= budget:
                    np.testing.assert_array_equal(result.essential_delivered, essential)
                else:
                    self.assertTrue(result.essential_infeasible)
                    np.testing.assert_allclose(
                        result.essential_delivered / essential, budget / essential.sum()
                    )
                    self.assertEqual(result.flexible_delivered.sum(), 0)

    def test_water_filling_and_no_backlog(self):
        # Household one was fully served yesterday; household two was curtailed.
        result = allocate([4, 4, 0], [0, 0, 0], 4, [4, 4, 100], [4, 0, 0])
        np.testing.assert_allclose(result.delivered, [0, 4, 0], atol=1e-8)
        # Once second household's current 1 kWh request fills, the remainder is
        # available to the other household; old curtailment is never queued.
        result = allocate([4, 1, 0], [0, 0, 0], 4, [4, 4, 100], [4, 0, 0])
        np.testing.assert_allclose(result.delivered, [3, 1, 0], atol=1e-8)
        result = allocate([4, 4], [1, 1], 4, method="proportional")
        np.testing.assert_allclose(result.delivered, [2, 2])
        result = allocate([4, 4], [1, 1], 4, method="fcfs", order=[1, 0])
        np.testing.assert_allclose(result.delivered, [1, 3])

    def test_zero_inputs_undefined_ratios_and_invalid_records(self):
        for method in ("fair", "proportional", "fcfs"):
            result = allocate([0, 0], [0, 0], 0, method=method, order=[0, 1])
            self.assertEqual(result.delivered.sum(), 0)
            self.assertFalse(result.essential_infeasible)
            result = allocate([2, 3], [1, 1.5], 0, method=method, order=[0, 1])
            self.assertEqual(result.delivered.sum(), 0)
            self.assertTrue(result.essential_infeasible)
        self.assertIsNone(jain([]))
        self.assertEqual(jain([0, 0]), 0)
        self.assertEqual(service([0, 0], [0, 0])["loss"], 0)
        self.assertIsNone(service([0, 0], [0, 0])["fairness"])
        # Excess service to the first household cannot cancel the second deficit.
        self.assertEqual(service([1, 1], [5, 0])["loss"], 0.5)
        for request, essential, budget in (
            ([-1], [0], 1),
            ([1], [2], 1),
            ([np.nan], [0], 1),
            ([1], [0], -1),
        ):
            with self.assertRaises(ValueError):
                allocate(request, essential, budget)
        with self.assertRaises(ValueError):
            allocate([3, 3], [0, 0], 1, method="fcfs", order=[0, 0])
        with self.assertRaises(ValueError):
            allocate([3], [0], 1, [0], [1])

    def test_calibration_units_and_region_isolation(self):
        scenario = CapacityScenario("units", 0.9, (2.0, 4.0), interval_hours=0.5)
        np.testing.assert_array_equal(scenario.budget_kwh, [1.0, 2.0])
        for scenario in self.scenarios:
            for area in range(2):
                values = (
                    self.data.demand[self.data.area_index == area][
                        :, self.data.partitions == 0
                    ]
                    .astype(float)
                    .sum(0)
                )
                self.assertEqual(
                    scenario.capacity_kw[area],
                    np.quantile(values, scenario.quantile, method="linear"),
                )
        altered = copy.copy(self.data)
        altered.demand = self.data.demand.copy()
        altered.demand[:, self.data.partitions != 0] = 1e6
        self.assertEqual(calibrate(altered), self.scenarios)
        env = self.env((CapacityScenario("isolated", 0.9, (0.0, 1e6)),))
        env.reset(options=dict(episode_index=0, scenario_index=0))
        env.step(0)
        self.assertEqual(env.delivered_energy[env.area_masks[0]].sum(), 0)
        self.assertGreater(env.delivered_energy[env.area_masks[1]].sum(), 0)

    def test_fcfs_determinism_and_synthetic_order_seeds(self):
        first = arrival_order(100, 24, 0, 0, 0)
        np.testing.assert_array_equal(first, arrival_order(100, 24, 0, 0, 0))
        self.assertFalse(np.array_equal(first, arrival_order(100, 24, 0, 0, 1)))
        self.assertFalse(np.array_equal(first, arrival_order(100, 24, 0, 1, 0)))

    def test_safe_snapshots_and_current_hour_replay(self):
        self.assertFalse(hasattr(self.provider, "data"))
        episode = self.provider.episodes[0]
        self.assertFalse(hasattr(episode, "target"))
        for origin in self.provider.episodes:
            self.assertTrue(
                (
                    self.data.partitions[origin.origin - 24 : origin.origin + 24] == 0
                ).all()
            )
        with self.assertRaises(ValueError):
            EpisodeProvider(self.data, "test")

        class CurrentOnly:
            def __init__(self, values, origin):
                self.values, self.origin = values, origin

            def __getitem__(self, key):
                if key[1] != self.origin:
                    raise AssertionError("Future demand was requested")
                return self.values[key]

        safe = copy.copy(self.data)
        safe.demand = CurrentOnly(self.data.demand, episode.origin)
        cursor = HistoricalReplay(safe, episode.origin)
        requests, _ = cursor.next_hour()
        np.testing.assert_array_equal(requests, self.data.demand[:, episode.origin])
        with self.assertRaises(AssertionError):
            cursor.next_hour()
        with patch.object(
            PreparedData, "batch", side_effect=AssertionError("Forbidden target loader")
        ):
            env = self.env()
            env.reset(options=dict(episode_index=0, scenario_index=0))
            temperature = env.history[:, -1, 2].copy()
            for _ in range(24):
                env.step(0)
                np.testing.assert_array_equal(env.history[:, -1, 2], temperature)

    def test_requested_feedback_ensemble_and_floor(self):
        scarce = self.env((CapacityScenario("scarce", 0.9, (0.0, 0.0)),))
        ample = self.env((CapacityScenario("ample", 0.9, (1e6, 1e6)),))
        for env in (scarce, ample):
            env.reset(options=dict(episode_index=0, scenario_index=0))
        np.testing.assert_array_equal(scarce.reference, ample.reference)
        for action in (0, 20, 5, 15):
            history = scarce.history.copy()
            ep = scarce.episode
            engines = [
                make_engine(self.responses, name, self.data, "cpu")
                for name in ("s42", "s7")
            ]
            predicted = np.mean(
                [
                    engine.predict(
                        history, scarce.grid[action], ep.future_calendar[scarce.hour]
                    )
                    for engine in engines
                ],
                axis=0,
            )
            actual = self.engine.predict(
                history, scarce.grid[action], ep.future_calendar[scarce.hour]
            )
            np.testing.assert_allclose(actual, predicted, rtol=1e-6)
            protected = np.maximum(actual, 0.5 * scarce.reference[scarce.hour])
            expected = append_history(
                history,
                protected,
                scarce.grid[action],
                ep.future_calendar[scarce.hour],
                self.provider.preprocessor,
            )
            scarce.step(action)
            ample.step(action)
            np.testing.assert_allclose(scarce.history, expected, rtol=1e-6)
            np.testing.assert_array_equal(scarce.history, ample.history)
        self.assertEqual(scarce.delivered_energy.sum(), 0)
        self.assertGreater(ample.delivered_energy.sum(), 0)
        # The external protection is explicitly counted when a raw forecast falls.
        env = self.env()
        env.reset(options=dict(episode_index=0, scenario_index=0))
        with patch.object(
            self.engine, "predict", return_value=np.zeros(self.data.households)
        ):
            env.step(0)
        np.testing.assert_allclose(env.requested_energy, 0.5 * env.reference[0])
        self.assertEqual(env.floor_bindings, self.data.households)
        self.assertTrue(
            all(
                not model.training
                and not any(p.requires_grad for p in model.parameters())
                for model in self.engine.models
            )
        )

    def test_reward_masks_observations_and_snapshot(self):
        env = self.env()
        check_env(env, warn=False)
        obs, _ = env.reset(seed=5, options=dict(episode_index=0, scenario_index=0))
        self.assertTrue(env.action_masks().all())
        self.assertEqual(env.action_masks().shape, (21,))
        self.assertEqual(env.grid[0], 0.05)
        self.assertEqual(env.grid[-1], 2.0)
        for _ in range(5):
            _, reward, _, _, _ = env.step(3)
            self.assertEqual(reward, 0)
        state = env.snapshot()
        first = env.step(10)
        env.restore(state)
        second = env.step(10)
        for key in obs:
            np.testing.assert_array_equal(first[0][key], second[0][key])
        for _ in range(18):
            _, reward, done, _, info = env.step(3)
        self.assertTrue(done)
        self.assertEqual(reward, -100 * info["loss"])
        self.assertEqual(info["postallocation_violation_region_hours"], 0)

    def test_price_free_physical_invariance(self):
        engine = make_engine(self.responses, "control", self.data, "cpu")
        env = self.env(engine=engine)
        low = evaluate(env, lambda current, obs: 0)
        high = evaluate(env, lambda current, obs: 20)
        for left, right in zip(low["days"], high["days"]):
            self.assertEqual(left["loss"], right["loss"])
            self.assertEqual(left["delivered_kwh"], right["delivered_kwh"])
            self.assertNotEqual(left["charge_nok"], right["charge_nok"])
        # Also verify direct network predictions ignore candidate/history prices.
        history = self.provider.features(self.provider.episodes[0])
        changed = history.copy()
        changed[:, :, 1] += 100
        np.testing.assert_array_equal(
            engine.predict(history, 0.05, np.zeros(6)),
            engine.predict(changed, 2.0, np.zeros(6)),
        )

    def test_deterministic_enumeration_and_pairing(self):
        env = self.env()
        first = evaluate(env, lambda current, obs: current.market_action())
        second = evaluate(env, lambda current, obs: current.market_action())
        self.assertEqual(first, second)
        self.assertEqual(first["cases"], 6)
        averaged = average_results([first, second, first])
        self.assertEqual(averaged["means"], first["means"])
        intervals = paired_daily(averaged, first)
        self.assertEqual(intervals["7"]["ci95"], [0.0, 0.0])
        self.assertEqual(intervals["7"]["days"], 2)
        self.assertEqual(intervals["7"]["repetitions"], 5000)
        bad = copy.deepcopy(first)
        bad["days"][0]["origin"] = "wrong day"
        with self.assertRaises(ValueError):
            paired_daily(bad, first)

    def test_three_policy_gradients_and_completed_update_recovery(self):
        with tempfile.TemporaryDirectory() as temporary:
            for configuration in CONFIGURATIONS:
                env = self.env()
                clock = SlotClock(100, configuration)
                overrides = (
                    dict(buffer_size=48, learning_starts=0, batch_size=8)
                    if configuration == "dqn"
                    else dict(n_steps=24, batch_size=8, n_epochs=1)
                )
                model = make_agent(env, configuration, 42, "cpu", clock, overrides)
                model.set_logger(configure(None, []))
                model.learn(total_timesteps=24, log_interval=None)
                self.assertGreater(model._n_updates, 0)
                self.assertEqual(env.hour, 0)
                self.assertTrue(
                    all(
                        p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.policy.parameters()
                    )
                )
                observation = env.observation()
                action = choose_action(model, env, observation)
                path = Path(temporary) / f"{configuration}.pkl"
                train.save_resume(
                    path, model, env, clock, dict(signature="fixture"), "cpu"
                )
                restored_env = self.env()
                restored, restored_clock, _ = train.load_resume(
                    path, restored_env, 100, "fixture", configuration, "cpu"
                )
                self.assertGreaterEqual(restored_clock.elapsed, clock.elapsed - 0.1)
                self.assertEqual(
                    choose_action(restored, restored_env, restored_env.observation()),
                    action,
                )
                for key in observation:
                    np.testing.assert_array_equal(
                        observation[key], restored_env.observation()[key]
                    )
                previous_updates = restored._n_updates
                restored.set_logger(configure(None, []))
                restored.learn(
                    total_timesteps=24, reset_num_timesteps=False, log_interval=None
                )
                self.assertEqual(restored.num_timesteps, 48)
                self.assertGreater(restored._n_updates, previous_updates)
                with self.assertRaises(ValueError):
                    train.load_resume(
                        path, restored_env, 100, "changed", configuration, "cpu"
                    )
                archive = Path(temporary) / f"{configuration}.zip"
                train.atomic_model(restored, archive)
                loaded = algorithm(configuration).load(archive, device="cpu")
                self.assertEqual(
                    choose_action(loaded, restored_env, restored_env.observation()),
                    choose_action(restored, restored_env, restored_env.observation()),
                )
                model.logger.close()
                restored.logger.close()

    def test_wall_clock_schedules_and_selection_ties(self):
        for configuration in CONFIGURATIONS:
            clock = SlotClock(100, configuration, 50)
            high = 1e-4 if configuration == "dqn" else 3e-4
            self.assertAlmostEqual(clock.learning_rate(), high * 0.55, places=7)
            self.assertAlmostEqual(clock.epsilon(), 0.05)
        row = dict(loss=0.1, fairness=0.9, charge_nok=10.0, mean_jump_nok_kwh=0.2)
        self.assertTrue(prefer(dict(row, loss=0.099), row))
        self.assertTrue(prefer(dict(row, loss=0.10000001, fairness=0.95), row))
        self.assertFalse(prefer(dict(row), row))

    def test_compact_replay_preserves_uniform_samples_and_bounds_storage(self):
        env = self.env()
        observation, _ = env.reset(options=dict(episode_index=0, scenario_index=0))
        compact = CompactReplayBuffer(
            5, env.observation_space, env.action_space, device="cpu"
        )
        ordinary = DictReplayBuffer(
            5, env.observation_space, env.action_space, device="cpu"
        )
        for index in range(12):
            following, reward, done, truncated, _ = env.step(index % 21)
            args = (
                {key: value[None] for key, value in observation.items()},
                {key: value[None] for key, value in following.items()},
                np.array([index % 21]),
                np.array([reward]),
                np.array([done]),
                [{"TimeLimit.truncated": truncated}],
            )
            compact.add(*args)
            ordinary.add(*args)
            observation = following
        np.random.seed(81)
        expected = ordinary.sample(64)
        np.random.seed(81)
        actual = compact.sample(64)
        for name in expected.observations:
            torch.testing.assert_close(
                actual.observations[name], expected.observations[name], rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.next_observations[name],
                expected.next_observations[name],
                rtol=0,
                atol=0,
            )
        for name in ("actions", "rewards", "dones"):
            torch.testing.assert_close(
                getattr(actual, name), getattr(expected, name), rtol=0, atol=0
            )
        references = sum(record[1] for record in compact.pool.values())
        self.assertEqual(references, 5 * 2 * len(observation))
        unshared_bytes = sum(value.nbytes for value in observation.values()) * 2 * 5
        self.assertLess(compact.bytes_stored, unshared_bytes)
        self.assertEqual(compact.size(), 5)
        compact.reset()
        self.assertEqual(compact.bytes_stored, 0)
        self.assertEqual(compact.size(), 0)

    def test_interrupted_validation_preserves_best_and_consumed_budget(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_json(
                output / "best.json",
                dict(
                    signature="fixture",
                    checkpoint_seconds=600.0,
                    validation=dict(means=dict(loss=0.08)),
                ),
            )
            save_json(
                output / "history.json",
                [dict(seconds=20.0, decisions=0), dict(seconds=600.0, decisions=1200)],
            )
            save_json(output / "progress.json", dict(seconds=610.0, decisions=1224))
            clock = SlotClock(4000, "dqn", 300)
            state = dict(best=0.1, history=[dict(seconds=20.0, decisions=0)])
            train.reconcile_progress(
                output, SimpleNamespace(num_timesteps=600), clock, state, "fixture"
            )
            self.assertEqual(state["best"], 0.08)
            self.assertGreaterEqual(clock.elapsed, 610)
            self.assertEqual(state["discarded_decisions"], 624)
            self.assertEqual(len(state["history"]), 2)

    def test_generated_end_to_end_baselines_replay_and_report(self):
        # Exercise all stages on one generated day. Separate tests above require
        # actual learner updates; tiny integration slots intentionally may stop
        # after initial validation rather than exceed their allocated time.
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            overrides = {
                "dqn": dict(buffer_size=48, learning_starts=0, batch_size=8),
                "ppo_mlp": dict(n_steps=24, batch_size=8, n_epochs=1),
                "ppo_transformer": dict(n_steps=24, batch_size=8, n_epochs=1),
            }
            with contextlib.redirect_stdout(io.StringIO()), patch.object(
                train, "preflight", return_value={}
            ), patch.object(train, "learning_plots"):
                train.run(
                    self.data.folder,
                    self.responses,
                    output,
                    "cpu",
                    0.00001,
                    limit_days=1,
                    overrides=overrides,
                )
            self.assertEqual(read_json(output / "status.json")["stage"], "complete")
            selection = read_json(output / "selection.json")
            self.assertTrue(selection["frozen_before_test"])
            self.assertEqual(selection["representative_seed"], 42)
            self.assertEqual(
                len(read_json(output / "fit_summaries.json")), len(RUN_ORDER)
            )
            self.assertFalse((output / "active_resume.pkl").exists())
            self.assertTrue((output / "REPORT.md").exists())
            self.assertTrue((output / "test_metrics.json").exists())
            with patch.object(
                train, "preflight", side_effect=AssertionError("Already finished")
            ):
                train.run(
                    self.data.folder,
                    self.responses,
                    output,
                    "cpu",
                    0.00001,
                    limit_days=1,
                    overrides=overrides,
                )
            with self.assertRaises(ValueError):
                changed = dict(
                    selection, fixed_action=(selection["fixed_action"] + 1) % 21
                )
                save_json(output / "selection.json", changed)
                train.final_evaluation(
                    self.data,
                    self.responses,
                    output,
                    self.scenarios,
                    self.cache,
                    None,
                    "changed",
                    selection,
                    "cpu",
                    {},
                    {},
                    1,
                )


if __name__ == "__main__":
    unittest.main()

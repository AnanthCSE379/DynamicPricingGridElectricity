"""Verify observer neutrality, reward accounting, recovery and dashboard controls."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.test_pipeline import fixture
from norway_energy_crisis.dqn.environment import EpisodeProvider, PriceEnv
from norway_energy_crisis.capacity.environment import CapacityPriceEnv
from norway_energy_crisis.capacity.scenarios import calibrate
from norway_energy_crisis.capacity.policies import SlotClock, make_agent
from norway_energy_crisis.capacity.train import save_resume, load_resume
from .recorder import EpisodeRecorder
from .store import DATABASE, diagnostics, episodes, load_episode
from .details import household_history, household_timeline


class FixtureEngine:
    """Cheap deterministic dynamics for testing recording, not consumer research."""

    signature = "generated-fixture-engine"
    is_price_free = False

    def __init__(self, scales):
        self.scales = scales

    def predict(self, history, price, calendar):
        return (
            history[:, -1, 0] * 0.97 + 0.03 + np.asarray(price) * 0.003
        ) * self.scales


class InspectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture_temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, prepared = fixture(cls.root)
        cls.data = PreparedData(prepared)
        cls.provider = EpisodeProvider(cls.data, "train")
        cls.engine = FixtureEngine(cls.data.preprocessor.scales)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def base(self, capacity=False):
        if capacity:
            return CapacityPriceEnv(
                self.provider, self.engine, calibrate(self.data), self.folder / "cache"
            )
        return PriceEnv(self.provider, self.engine, self.folder / "cache", 2.0)

    def recorder(self, capacity=False, name="recorded"):
        return EpisodeRecorder(
            self.base(capacity), self.folder / name, {"fixture": True}
        )

    @staticmethod
    def day(env, index=0):
        env.reset(seed=42, options={"episode_index": index, "scenario_index": 0})
        rewards = []
        for hour in range(24):
            result = env.step((hour * 3) % 21)
            rewards.append(result[1])
        return rewards, result[-1]

    def test_observations_rewards_and_rng_unchanged(self):
        baseline, recorded = self.base(), self.recorder()
        for env in (baseline, recorded):
            env.reset(seed=37, options={"episode_index": 0})
        rng = torch.get_rng_state().clone()
        for hour in range(24):
            expected = baseline.step((hour * 5) % 21)
            actual = recorded.step((hour * 5) % 21)
            self.assertEqual(expected[1:4], actual[1:4])
            for key in expected[0]:
                np.testing.assert_array_equal(expected[0][key], actual[0][key])
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(
            baseline.np_random.bit_generator.state,
            recorded.np_random.bit_generator.state,
        )
        saved = load_episode(recorded.store.path, 1)
        self.assertEqual(
            saved["arrays"]["requested"].shape, (24, self.provider.households)
        )
        self.assertAlmostEqual(saved["reward"], -100 * actual[-1]["cost"])
        self.assertFalse(any(level == "error" for level, _ in diagnostics(saved)))
        recorded.close()

    def test_hourly_and_terminal_components_are_not_redistributed(self):
        for capacity in (False, True):
            with self.subTest(capacity=capacity):
                env = self.recorder(capacity, str(capacity))
                rewards, info = self.day(env)
                saved = load_episode(env.store.path, 1)
                frame = saved["steps"]
                np.testing.assert_allclose(
                    frame.filter(regex="^reward_").sum(axis=1), rewards, atol=1e-6
                )
                if capacity:
                    self.assertEqual(sum(rewards[:-1]), 0)
                    self.assertAlmostEqual(sum(rewards), -100 * info["loss"])
                else:
                    self.assertEqual(frame.reward_energy.iloc[:-1].sum(), 0)
                    self.assertEqual(frame.reward_charge.iloc[:-1].sum(), 0)
                env.close()

    def test_partial_episode_is_visible_and_recovery_flags_lost_work(self):
        env = self.recorder()
        self.day(env)
        env.reset(options={"episode_index": 0})
        snapshot = env.snapshot()
        env.step(0)
        partial = load_episode(env.store.path, 2)
        self.assertEqual(partial["status"], "active")
        self.assertEqual(len(partial["steps"]), 1)
        self.assertEqual(len(episodes(env.store.path)), 2)
        env.restore(snapshot)
        self.assertEqual(episodes(env.store.path).discarded.tolist(), [0, 1])
        self.day(env)
        self.assertEqual(episodes(env.store.path).id.tolist(), [1, 2, 3])
        env.close()

    def test_model_roundtrip_with_recorded_dqn_and_masked_ppo(self):
        for configuration in ("dqn", "ppo_mlp", "ppo_transformer"):
            with self.subTest(configuration=configuration):
                env = self.recorder(True, configuration)
                clock = SlotClock(300, configuration)
                overrides = (
                    {"buffer_size": 64, "learning_starts": 0, "batch_size": 8}
                    if configuration == "dqn"
                    else {"n_steps": 24, "batch_size": 12, "n_epochs": 1}
                )
                model = make_agent(env, configuration, 42, "cpu", clock, overrides)
                env.bind(model, clock)
                model.learn(24)
                self.assertGreater(model._n_updates, 0)
                self.assertEqual(env.hour, 0)
                state = {"signature": {"fixture": configuration}}
                path = self.folder / f"{configuration}.pkl"
                save_resume(path, model, env, clock, state, "cpu")
                recovered, restored_clock, _ = load_resume(
                    path, env, 300, state["signature"], configuration, "cpu"
                )
                env.bind(recovered, restored_clock)
                recovered.learn(24, reset_num_timesteps=False)
                self.assertEqual(recovered.num_timesteps, 48)
                self.assertEqual(episodes(env.store.path).steps.tolist(), [24, 24])
                env.close()

    def test_dashboard_empty_legacy_and_interactive_recording(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install optional dashboard dependencies to test the UI.")

        app_path = str(Path(__file__).with_name("app.py"))
        with patch.dict(os.environ, {"NORWAY_DASHBOARD_RUNS": str(self.folder)}):
            empty = AppTest.from_file(app_path).run(timeout=20)
            self.assertFalse(empty.exception)
            env = self.recorder()
            self.day(env)
            self.day(env, 1)
            app = AppTest.from_file(app_path).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.tabs), 4)
            chooser = next(
                widget for widget in app.selectbox if widget.label == "Episode"
            )
            chooser.select(1).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(
                any(m.label == "Highest episode reward" for m in app.metric)
            )
            self.assertTrue(app.dataframe)
            env.close()
        legacy = self.folder / "legacy"
        legacy.mkdir()
        (legacy / "best.zip").touch()
        (legacy / "history.json").write_text(
            json.dumps([{"seconds": 1, "mean_cost": 0.1}])
        )
        with patch.dict(os.environ, {"NORWAY_DASHBOARD_RUNS": str(legacy)}):
            app = AppTest.from_file(app_path).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)

    def test_provenance_mismatch_rejected(self):
        env = self.recorder()
        env.close()
        with self.assertRaises(ValueError):
            EpisodeRecorder(self.base(), self.folder / "recorded", {"fixture": False})
        self.assertTrue((self.folder / "recorded" / DATABASE).exists())

    def test_household_accounts_and_visible_history_match_the_selected_step(self):
        env = self.recorder(True)
        self.day(env)
        saved = load_episode(env.store.path, 1)
        household = saved["context"]["household_ids"][0]
        timeline = household_timeline(saved, household)
        np.testing.assert_array_equal(
            timeline.requested_kwh, saved["arrays"]["requested"][:, 0]
        )
        np.testing.assert_allclose(
            timeline.household_charge_nok, timeline.delivered_kwh * saved["steps"].price
        )
        self.assertAlmostEqual(
            timeline.cumulative_requested_kwh.iloc[7],
            float(saved["arrays"]["requested"][:8, 0].astype(np.float64).sum()),
        )
        initial = household_history(saved, household, 1)
        np.testing.assert_array_equal(
            initial["Demand kWh"], saved["context"]["initial_history"][0]
        )
        incoming = household_history(saved, household, 8)
        self.assertEqual(len(incoming), 24)
        np.testing.assert_array_equal(
            incoming["Demand kWh"].to_numpy()[-7:], saved["arrays"]["requested"][:7, 0]
        )
        # This action's outcome and later outcomes cannot enter its incoming history.
        saved["arrays"]["requested"][7:, 0] = 1e6
        pd.testing.assert_frame_equal(incoming, household_history(saved, household, 8))
        for hour in (0, 25):
            with self.assertRaises(ValueError):
                household_history(saved, household, hour)
        with self.assertRaises(ValueError):
            household_timeline(saved, "missing")
        env.close()

    def test_individual_household_and_hour_navigation(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install optional dashboard dependencies to test the UI.")

        def widget(app, collection, label):
            return next(w for w in getattr(app, collection) if w.label == label)

        env = self.recorder()
        self.day(env)
        self.day(env, 1)
        with patch.dict(os.environ, {"NORWAY_DASHBOARD_RUNS": str(self.folder)}):
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(
                timeout=20
            )
            self.assertFalse(app.exception)
            first_household = widget(app, "selectbox", "Household").value
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 1)
            self.assertTrue(widget(app, "button", "Previous hour").disabled)
            widget(app, "button", "Next hour").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 2)
            widget(app, "button", "Next household").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertNotEqual(
                widget(app, "selectbox", "Household").value, first_household
            )
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 2)
            widget(app, "button", "Previous household").click().run(timeout=20)
            self.assertEqual(
                widget(app, "selectbox", "Household").value, first_household
            )
            widget(app, "button", "Previous hour").click().run(timeout=20)
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 1)
            widget(app, "button", "Last recorded hour").click().run(timeout=20)
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 24)
            self.assertTrue(widget(app, "button", "Next hour").disabled)
            widget(app, "number_input", "Inspect hour").set_value(8).run(timeout=20)
            expected = household_timeline(
                load_episode(env.store.path, 2), first_household
            ).iloc[7]
            self.assertEqual(
                widget(app, "metric", "Household power (kW)").value,
                f"{expected.requested_kw:.4f}",
            )
            widget(app, "selectbox", "Episode").select(1).run(timeout=20)
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 1)
            widget(app, "selectbox", "Episode").select(2).run(timeout=20)
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 8)
            widget(app, "button", "Refresh now").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(widget(app, "number_input", "Inspect hour").value, 8)
        env.close()

    def test_refresh_discovers_runs_and_reads_new_steps_while_writer_is_open(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install optional dashboard dependencies to test the UI.")

        def widget(app, collection, label):
            return next(w for w in getattr(app, collection) if w.label == label)

        with patch.dict(os.environ, {"NORWAY_DASHBOARD_RUNS": str(self.folder)}):
            app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(
                timeout=20
            )
            self.assertFalse(app.exception)
            self.assertEqual(widget(app, "selectbox", "Refresh interval").value, 30)
            env = self.recorder()
            env.reset(options={"episode_index": 0})
            env.step(1)
            widget(app, "button", "Refresh now").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any("episode #1, hour 1" in c.value for c in app.caption))
            widget(app, "selectbox", "Refresh interval").select(60).run(timeout=20)
            widget(app, "checkbox", "Automatic refresh").uncheck().run(timeout=20)
            self.assertFalse(app.exception)
            env.step(2)
            widget(app, "button", "Refresh now").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertTrue(any("episode #1, hour 2" in c.value for c in app.caption))
            widget(app, "checkbox", "Follow newest episode").check().run(timeout=20)
            self.assertTrue(widget(app, "selectbox", "Episode").disabled)
            for _ in range(22):
                env.step(1)
            env.reset(options={"episode_index": 1})
            env.step(3)
            widget(app, "button", "Refresh now").click().run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(widget(app, "selectbox", "Episode").value, 2)
            env.close()


if __name__ == "__main__":
    unittest.main()

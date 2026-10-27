"""Generated-fixture checks for costs, information boundaries and learner recovery."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from norway_energy_crisis.clean_data import digest
from norway_energy_crisis.test_pipeline import fixture
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.capacity.scenarios import calibrate
from norway_energy_crisis.inspection.recorder import EpisodeRecorder
from norway_energy_crisis.inspection.store import load_episode, episodes
from norway_energy_crisis.dqn.train import (
    SlotClock,
    make_agent,
    save_resume,
    load_resume,
)
from norway_energy_crisis.train_response import (
    ResponseModel,
    checkpoint,
    atomic_torch_save,
)
from norway_energy_crisis.response_v2.data import ConsumerData
from norway_energy_crisis.response_v2.model import (
    ConsumerModel,
    ConsumerSimulator,
    bundle,
    atomic_save,
)
from .engine import ResponseEngine
from .environment import EpisodeProvider, StressPriceEnv
from .evaluation import evaluate, baselines, paired_intervals
from .objective import state_weights, stress_cost, household_costs, bounded
from .io import save_json
from . import train
from .audit import simulator_audit
from .report import write_report, value_audit, continuation_audit


class FixtureEngine:
    """Deterministic household dynamics; fixture tests do not claim real response."""

    family = "old"
    signature = "fixture-stress48"
    is_price_free = False

    def __init__(self, data, price_free=False):
        self.scales = data.preprocessor.scales
        self.is_price_free = price_free

    def predict(self, history, price, calendar):
        history = np.asarray(history)
        p = 0 if self.is_price_free else np.asarray(price) * 0.002
        return (history[..., -1, 0] * 0.98 + 0.02 + p) * self.scales

    def candidates(self, history, prices, calendar):
        return self.predict(
            np.broadcast_to(history, (len(prices), *history.shape)),
            np.asarray(prices)[:, None],
            calendar,
        )


class PricingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture_temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.fixture_temp.name)
        with redirect_stdout(io.StringIO()):
            _, cls.cleaned, prepared = fixture(cls.root)
        cls.data = PreparedData(prepared)
        cls.provider = EpisodeProvider(cls.data, "train")
        cls.validation = EpisodeProvider(cls.data, "validation")
        cls.scenarios = calibrate(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.fixture_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.engine = FixtureEngine(self.data)

    def tearDown(self):
        self.temp.cleanup()

    def env(self, provider=None, engine=None, folder="cache"):
        return StressPriceEnv(
            provider or self.provider,
            engine or self.engine,
            self.scenarios,
            self.folder / folder,
        )

    def day(self, env, action):
        env.reset(seed=42, options={"episode_index": 0, "scenario_index": 2})
        rewards = []
        for hour in range(48):
            result = env.step(action(hour) if callable(action) else action)
            rewards.append(result[1])
            self.assertEqual(result[2], hour == 47)
            if hour < 47:
                self.assertEqual(result[-1]["reward_components"]["energy"], 0)
                self.assertEqual(result[-1]["reward_components"]["charge"], 0)
        self.assertAlmostEqual(sum(rewards), -100 * result[-1]["cost"], places=7)
        return result[-1]

    def test_weights_cost_bound_tolerances_and_negative_references(self):
        np.testing.assert_allclose(state_weights([0.8, 0.9, 1]), [1, 3, 5])
        grid = np.linspace(0, 100, 10001)
        self.assertTrue((np.diff(stress_cost(grid)) >= 0).all())
        self.assertTrue(((stress_cost(grid) >= 0) & (stress_cost(grid) <= 1)).all())
        self.assertEqual(stress_cost(0.9), 0)
        self.assertAlmostEqual(float(stress_cost(1)), 0.125)
        same = household_costs([1, 0], [1, 0], [1, 0], [1, 0], 0.5)
        self.assertEqual(same, {"energy": 0, "charge": 0})
        within = household_costs([1.04], [1.04], [1], [1], 0.5)
        self.assertEqual(within, {"energy": 0, "charge": 0})
        for energy in (0.5, 1.5):
            self.assertGreater(
                household_costs([energy], [1], [1], [1], 0.5)["energy"], 0
            )
        self.assertGreater(
            household_costs([2, 0], [1, 1], [1, 1], [1, 1], 0.5)["energy"], 0
        )
        guarded = household_costs([1, 0], [0, 1], [1, 0], [-1, 0], 0.5)
        self.assertTrue(all(np.isfinite(v) for v in guarded.values()))
        with self.assertRaises(ValueError):
            StressPriceEnv(
                self.provider,
                self.engine,
                [replace(self.scenarios[0], capacity_kw=(0, 1))],
                self.folder,
            )
        with self.assertRaises(ValueError):
            bounded(-1)

    def test_daily_reveal_has_no_price_reference_or_account_leakage(self):
        changed = EpisodeProvider(self.data, "train")
        changed._price_days[0] = (changed._price_days[0][0], np.full((2, 24), 99.0))
        a, b = self.env(), self.env(changed, folder="changed")
        x, _ = a.reset(options={"episode_index": 0, "scenario_index": 0})
        y, _ = b.reset(options={"episode_index": 0, "scenario_index": 0})
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])
        self.assertTrue((x["known_mask"][24:] == 0).all())
        self.assertTrue((x["known_prices"].reshape(2, 48)[:, 24:] == 0).all())
        for hour in range(24):
            x, *_ = a.step(3)
            y, *_ = b.step(3)
            if hour < 23:
                for key in x:
                    np.testing.assert_array_equal(x[key], y[key])
        self.assertTrue((x["known_mask"] == 1).all())
        self.assertFalse(np.array_equal(x["known_prices"], y["known_prices"]))
        self.assertEqual(a.known_hours, 48)

    def test_provider_reads_history_only_and_partitions_are_isolated(self):
        class HistoryOnly:
            def __init__(self, array):
                self.array = array

            def __getitem__(self, key):
                times = np.asarray(key[1])
                if times.size != 24 or np.ptp(times) != 23:
                    raise AssertionError("Future measurement window accessed")
                return self.array[key]

        original_demand, original_temperature = self.data.demand, self.data.temperature
        try:
            self.data.demand = HistoryOnly(original_demand)
            self.data.temperature = HistoryOnly(original_temperature)
            provider = EpisodeProvider(self.data, "validation")
            env = self.env(provider)
            self.day(env, 0)
        finally:
            self.data.demand, self.data.temperature = (
                original_demand,
                original_temperature,
            )
        for code, split in enumerate(("train", "validation")):
            for episode in EpisodeProvider(self.data, split).episodes:
                self.assertTrue(
                    (
                        self.data.partitions[episode.origin - 24 : episode.origin + 48]
                        == code
                    ).all()
                )
        with self.assertRaises(ValueError):
            EpisodeProvider(self.data, "test")

    def test_exact_rewards_no_allocation_and_pre_action_weights(self):
        env = self.env()
        result = self.day(env, lambda t: (t * 7) % 21)
        initial = env.regional_power(env.episode.demand_history[:, -1]) / np.asarray(
            env.scenario.capacity_kw
        )
        np.testing.assert_allclose(env.weights[0], state_weights(initial))
        for hour in range(1, 48):
            previous = env.regional_power(env.demands[hour - 1]) / np.asarray(
                env.scenario.capacity_kw
            )
            np.testing.assert_array_equal(env.weights[hour], state_weights(previous))
        self.assertFalse(env.allocation_enabled)
        self.assertTrue(0 <= result["cost"] <= 1)
        self.assertEqual(len(env.demands), 48)
        self.assertTrue(env.action_masks().all())
        with self.assertRaises(ValueError):
            env.step(0)

    def test_reference_independence_and_price_free_physical_invariance(self):
        engine = FixtureEngine(self.data, True)
        a, b = self.env(engine=engine), self.env(engine=engine)
        self.day(a, 0)
        self.day(b, 20)
        np.testing.assert_array_equal(a.demands, b.demands)
        np.testing.assert_array_equal(a.reference, b.reference)
        np.testing.assert_array_equal(a.demands, a.reference)
        self.assertEqual(
            a.daily_result()["terms"]["grid"], b.daily_result()["terms"]["grid"]
        )
        self.assertEqual(a.daily_result()["terms"]["energy"], 0)

    def test_explicit_telemetry_and_reference_refresh(self):
        env = EpisodeRecorder(self.env(), self.folder / "recording", {"fixture": True})
        env.reset(options={"episode_index": 0, "scenario_index": 0})
        env.step(2)
        partial = load_episode(env.store.path, 1)
        self.assertFalse(partial["metadata"]["allocation_enabled"])
        self.assertTrue(np.isnan(partial["context"]["reference"][24:]).all())
        for _ in range(47):
            env.step(2)
        saved = load_episode(env.store.path, 1)
        self.assertEqual(saved["steps"].shape[0], 48)
        self.assertTrue(np.isfinite(saved["context"]["reference"]).all())
        np.testing.assert_array_equal(
            saved["arrays"]["requested"], saved["arrays"]["delivered"]
        )
        np.testing.assert_allclose(
            saved["steps"].filter(regex="^reward_").sum(1), saved["steps"].reward
        )
        self.assertIn("grid_weight_0", saved["steps"])
        env.close()

    def test_old_new_adapter_checkpoint_contracts_and_raw_api(self):
        consumer_data = ConsumerData(self.cleaned, self.data.folder)
        old_root, new_root = self.folder / "old", self.folder / "new"
        price_paths, control_paths = [], []
        for seed in (42, 7, 2026):
            torch.manual_seed(seed)
            old = ResponseModel("Transformer", 24, self.data.static.shape[1], True)
            old_folder = old_root / f"transformer_24h_seed{seed}_price"
            old_folder.mkdir(parents=True)
            atomic_torch_save(
                checkpoint(old, self.data, {}, 6, 1.0), old_folder / "best.pt"
            )
            for use_price in (True, False):
                model = ConsumerModel(self.data.static.shape[1], use_price=use_price)
                torch.nn.init.normal_(model.head[-1].weight, std=0.005)
                path = (
                    new_root
                    / f"C_seed{seed}_{'price' if use_price else 'no_price'}/best.pt"
                )
                path.parent.mkdir(parents=True)
                atomic_save(bundle(model, consumer_data), path)
                (price_paths if use_price else control_paths).append(
                    str(path.resolve())
                )
        save_json(
            new_root / "selection.json",
            dict(
                price_checkpoints=price_paths,
                control_checkpoints=control_paths,
                checkpoint_sha256={p: digest(p) for p in price_paths + control_paths},
            ),
        )
        first = self.provider.episodes[0]
        new = ResponseEngine(self.data, "new", "cpu", old_root, new_root)
        old = ResponseEngine(self.data, "old", "cpu", old_root, new_root)
        self.assertTrue(
            all(
                not p.requires_grad
                for e in (old, new)
                for m in e.models
                for p in m.parameters()
            )
        )
        origin = first.origin
        past = np.arange(origin - 24, origin)
        sim = ConsumerSimulator(price_paths[:2], "cpu")
        expected = sim.forecast(
            first.demand_history,
            self.data.prices[self.data.area_index[:, None], past],
            self.data.temperature[self.data.area_index[:, None], past],
            self.data.times[past],
            self.data.preprocessor.ids,
            np.full((self.data.households, 1), 0.7),
        )[:, 0]
        np.testing.assert_allclose(
            new.predict(first.history, 0.7, first.future_calendar[0]),
            expected,
            rtol=1e-5,
            atol=1e-6,
        )
        control = ResponseEngine(self.data, "control", "cpu", old_root, new_root)
        np.testing.assert_array_equal(
            control.predict(first.history, 0.05, first.future_calendar[0]),
            control.predict(first.history, 2, first.future_calendar[0]),
        )

    def test_update_recovery_discarded_logs_and_incompatible_fingerprint(self):
        env = EpisodeRecorder(self.env(), self.folder / "fit", {"fixture": True})
        clock = SlotClock(100)
        model = make_agent(
            env, "cpu", 42, clock, dict(buffer_size=64, learning_starts=0, batch_size=8)
        )
        env.bind(model, clock)
        model.learn(48)
        state = {"signature": {"fixture": 1}}
        path = self.folder / "resume.pkl"
        save_resume(path, model, env, clock, state, "cpu")
        model.learn(48, reset_num_timesteps=False)
        restored, restored_clock, _ = load_resume(
            path, env, 100, state["signature"], "cpu"
        )
        env.bind(restored, restored_clock)
        restored.learn(48, reset_num_timesteps=False)
        self.assertEqual(restored.num_timesteps, 96)
        self.assertEqual(episodes(env.store.path).discarded.tolist(), [0, 1, 0])
        with self.assertRaises(ValueError):
            load_resume(path, env, 100, {"fixture": 2}, "cpu")
        env.close()

    def test_small_baseline_comparison_and_bounded_fit(self):
        provider = EpisodeProvider(self.data, "validation")
        provider.episodes = provider.episodes[:1]
        env = self.env(provider)
        result = baselines(env, self.folder / "baselines", self.folder / "constants")
        self.assertEqual(
            set(result["results"]),
            {
                "minimum",
                "maximum",
                "fixed",
                "constant_oracle",
                "market",
                "capacity_rule",
                "greedy",
            },
        )
        self.assertEqual(result["results"]["fixed"]["cases"], 3)
        fixed = evaluate(env, lambda e: 3)
        paired = paired_intervals([fixed], [fixed], resamples=20)
        self.assertEqual(paired["7"]["upper"], 0)
        train_provider = EpisodeProvider(self.data, "train")
        train_provider.episodes = train_provider.episodes[:1]
        fit = train.fit(
            train_provider,
            provider,
            self.engine,
            self.scenarios[:1],
            self.folder / "cache",
            self.folder / "tiny_fit",
            42,
            2.0,
            "fixture",
            "cpu",
            validation_seconds=100,
            checkpoint_seconds=0.5,
            overrides=dict(buffer_size=64, learning_starts=0, batch_size=8),
        )
        self.assertGreater(fit["updates"], 0)
        self.assertEqual(fit["decisions"] % 48, 0)
        self.assertTrue((self.folder / "tiny_fit/best.zip").exists())

    def test_audits_and_report_complete_on_generated_data(self):
        provider = EpisodeProvider(self.data, "validation")
        provider.episodes = provider.episodes[:1]
        env = self.env(provider)
        baseline = baselines(env, self.folder / "baselines", self.folder / "constants")
        audit = simulator_audit(
            self.data,
            {"validation": provider},
            {"old": self.engine},
            self.scenarios,
            self.folder / "cache",
            self.folder,
        )
        self.assertIn("price_support", audit)
        clock = SlotClock(60)
        model = make_agent(env, "cpu", 42, clock)
        q = value_audit(env, model, self.folder / "q_values.json")
        self.assertEqual(q["states"], 4)
        self.assertTrue(np.isfinite(q["mean_regret"]))
        continuation = continuation_audit(
            self.data, env, {"fixture": model}, self.folder / "continuation.json"
        )
        self.assertTrue(continuation["rows"])
        fits = []
        policies = {}
        for family in ("old", "new"):
            for seed in (42, 7, 2026):
                name = f"{family}_seed{seed}"
                policies[name] = baseline["results"]["market"]
                fits.append(
                    dict(
                        family=family,
                        seed=seed,
                        decisions=48,
                        updates=0,
                        best_decisions=0,
                        elapsed_seconds=1.0,
                        initial_untrained=True,
                    )
                )
                save_json(
                    self.folder / name / "history.json",
                    [dict(seconds=1.0, decisions=48, mean_cost=0.1)],
                )
        save_json(self.folder / "fit_summaries.json", fits)
        save_json(
            self.folder / "settings.json",
            dict(
                scenarios=[
                    dict(name=s.name, capacity_kw=list(s.capacity_kw))
                    for s in self.scenarios
                ]
            ),
        )
        paired = paired_intervals(
            [policies["new_seed42"]], [policies["old_seed42"]], resamples=20
        )
        comparisons = {
            family: dict(new_minus_old=paired) for family in ("new", "reserved")
        }
        results = {"new": dict(policies=policies, baselines=baseline["results"])}
        write_report(self.folder, {}, comparisons, results)
        self.assertIn("did not establish", (self.folder / "REPORT.md").read_text())
        self.assertEqual(len(list(self.folder.glob("*.png"))), 5)
        self.assertTrue((self.folder / "comparison.csv").exists())

    def test_dashboard_stress_episode_and_daily_reference_reveal(self):
        try:
            from streamlit.testing.v1 import AppTest
        except ImportError:
            self.skipTest("Install optional dashboard dependencies to test the UI.")
        env = EpisodeRecorder(self.env(), self.folder / "recorded", {"fixture": True})
        env.reset(options={"episode_index": 0, "scenario_index": 0})
        env.step(1)
        app_path = Path(__file__).parents[1] / "inspection/app.py"
        with patch.dict(os.environ, {"NORWAY_DASHBOARD_RUNS": str(self.folder)}):
            app = AppTest.from_file(str(app_path)).run(timeout=30)
            self.assertFalse(app.exception)
            for hour in range(47):
                env.step(hour % 21)
            next(w for w in app.button if w.label == "Refresh now").click().run(
                timeout=30
            )
            self.assertFalse(app.exception)
            inspector = next(w for w in app.number_input if w.label == "Inspect hour")
            inspector.set_value(48).run(timeout=30)
            self.assertFalse(app.exception)
            self.assertTrue(any("grid weight" in m.value.lower() for m in app.caption))
            self.assertFalse(
                any("essential power" in m.label.lower() for m in app.metric)
            )
        env.close()


if __name__ == "__main__":
    unittest.main()

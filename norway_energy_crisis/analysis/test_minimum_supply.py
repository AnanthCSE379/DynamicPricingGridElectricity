"""Supply reservations, safe feedback and curtailment stress diagnostics."""

import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from norway_energy_crisis.clean_data import digest
from norway_energy_crisis.minimum_supply import MinimumSupply
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.dqn.environment import EpisodeProvider
from norway_energy_crisis.capacity.environment import CapacityPriceEnv
from norway_energy_crisis.capacity.scenarios import CapacityScenario
from norway_energy_crisis.analysis.stress_minimum_supply import intervention_rollout
from norway_energy_crisis.test_pipeline import fixture
from norway_energy_crisis.analysis.test_load_support import ScaledPersistence
from norway_energy_crisis.analysis import test_load_support


class MinimumSupplyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, path = fixture(cls.root)
        cls.data = PreparedData(path)
        cls.baseline = MinimumSupply.fit(cls.data)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_training_only_fit_and_positive_quantile(self):
        changed = copy.copy(self.data)
        changed.demand = np.array(self.data.demand)
        changed.demand[:, self.data.partitions != 0] = 10000
        np.testing.assert_array_equal(
            MinimumSupply.fit(changed).floor_kw, self.baseline.floor_kw
        )
        for i, floor in enumerate(self.baseline.floor_kw):
            train = self.data.demand[i, self.data.partitions == 0]
            self.assertEqual(
                floor, np.quantile(train[train > 0], 0.01, method="linear")
            )
        changed.demand[:] = 0
        self.assertEqual(MinimumSupply.fit(changed).floor_kw, (0, 0))

    def test_demand_cap_zero_requests_and_units(self):
        baseline = MinimumSupply((1, 2, 3), (0.1, 0.1, 0.1), "data", "test")
        result = baseline.allocate([0, 0.03, 1], [0, 0, 0], 0.13)
        np.testing.assert_allclose(result.delivered, [0, 0.03, 0.1])
        self.assertAlmostEqual(result.delivered.sum(), 0.13)
        units = MinimumSupply((1, 2), (0.2, 0.2), "data", "test")
        result = units.allocate([1, 1], [0, 0], 0.2, interval_hours=0.5)
        np.testing.assert_allclose(result.delivered, [0.1, 0.1])

    def test_fair_priority_includes_current_reserved_service(self):
        baseline = MinimumSupply((1, 2), (0.5, 0.5), "data", "test")
        result = baseline.allocate([4, 4], [0, 0], 4, [4, 4], [4, 0])
        np.testing.assert_allclose(result.delivered, [0.5, 3.5])
        result = baseline.allocate([4, 4], [1, 1], 4)
        np.testing.assert_allclose(result.essential_delivered, [1, 1])
        np.testing.assert_allclose(result.flexible_delivered, [1, 1])

    def test_infeasible_and_invalid_reservations_fail(self):
        baseline = MinimumSupply((1, 2), (0.1, 0.1), "data", "test")
        for budget in (0, 0.19):
            with self.assertRaisesRegex(ValueError, "Capacity cannot meet"):
                baseline.allocate([1, 1], [0, 0], budget)
        for floors in ((-1, 0), (float("nan"), 0)):
            with self.assertRaises(ValueError):
                MinimumSupply((1, 2), floors, "data", "test")
        with self.assertRaises(ValueError):
            baseline.allocate([1, 1], [0, 0], 1, [1, 1], [2, 0])

    def test_persistence_and_provenance(self):
        path = self.root / "minimum_supply.json"
        self.baseline.save(path)
        self.assertEqual(MinimumSupply.load(path), self.baseline)
        state = json.loads(path.read_text())
        state["floor_kw"][0] += 1
        path.write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, "changed"):
            MinimumSupply.load(path)
        provider = EpisodeProvider(self.data, "train")
        self.baseline.validate_provider(provider)
        provider.manifest_hash = "different"
        with self.assertRaises(ValueError):
            self.baseline.validate_provider(provider)

    def test_environment_feedback_and_snapshot_identity(self):
        provider = EpisodeProvider(self.data, "train")
        scales = self.data.preprocessor.scales
        engine = SimpleNamespace(
            signature="persistence_test",
            is_price_free=False,
            predict=lambda history, price, calendar: history[:, -1, 0] * scales,
        )
        scenario = CapacityScenario("small", 0.9, (0.65, 0.75), alpha=0)
        baseline = MinimumSupply.fixed(self.data)
        env = CapacityPriceEnv(
            provider, engine, [scenario], self.root / "cache", minimum_supply=baseline
        )
        env.reset(options=dict(episode_index=0, scenario_index=0))
        requested = env.history[:, -1, 0] * scales
        _, reward, terminated, _, _ = env.step(0)
        np.testing.assert_allclose(env.history[:, -1, 0] * scales, requested)
        self.assertTrue((env.delivered_energy < env.requested_energy).all())
        self.assertEqual(reward, 0)
        self.assertFalse(terminated)
        state = env.snapshot()
        restored = CapacityPriceEnv(
            provider, engine, [scenario], self.root / "cache", minimum_supply=baseline
        )
        restored.restore(state)
        np.testing.assert_array_equal(restored.history, env.history)
        old = CapacityPriceEnv(provider, engine, [scenario], self.root / "cache")
        with self.assertRaisesRegex(ValueError, "baseline mismatch"):
            old.restore(state)
        total = reward
        for _ in range(23):
            _, reward, _, _, info = env.step(0)
            total += reward
        self.assertEqual(total, -100 * info["loss"])
        self.assertEqual(info["minimum_supply_signature"], baseline.signature)
        self.assertEqual(
            baseline.prepared_manifest_sha256,
            digest(self.data.folder / "manifest.json"),
        )

    def test_intervention_is_explicit_and_releases_after_cut(self):
        data = test_load_support.LoadSupportTests().data()
        from norway_energy_crisis.analysis.audit_load_support import inputs

        values = inputs(data, np.array([0]), np.array([24]), 72)
        predicted, feedback = intervention_rollout(
            [ScaledPersistence(1)], values, 0.01, "cpu", cut_hours=6
        )
        self.assertEqual(predicted[0, 0], 1)
        np.testing.assert_allclose(feedback[0, :6], 0.01)
        np.testing.assert_allclose(predicted[0, 1:], 0.01)
        np.testing.assert_array_equal(predicted[:, 6:], feedback[:, 6:])
        np.testing.assert_array_equal(values["history"][:, -1, 0], [0.5])


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()

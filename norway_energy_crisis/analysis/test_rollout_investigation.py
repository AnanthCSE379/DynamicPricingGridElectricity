"""Verify explicit recursive and measurement-refreshed sliding-window semantics."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from norway_energy_crisis.analysis.investigate_rollout import (
    advance,
    metrics,
    sliding_forecast,
)
from norway_energy_crisis.response_v2.data import ConsumerData
from norway_energy_crisis.response_v2.model import ConsumerModel, rollout
from norway_energy_crisis.test_pipeline import fixture


class PreviousLoadModel(torch.nn.Module):
    def encode(self, history):
        return history[:, -1, 0]

    def predict_encoded(self, encoded, history, static, price, calendar):
        return encoded + 0.1


class SlidingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        with redirect_stdout(io.StringIO()):
            _, cleaned, prepared = fixture(cls.root)
        cls.data = ConsumerData(cleaned, prepared)
        cls.hh = cls.data.all_households
        cls.inputs = cls.data.inputs(cls.hh, np.full(3, 48), 72)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self, price=True):
        model = ConsumerModel(
            self.data.features.static.shape[1], use_price=price
        ).eval()
        torch.nn.init.normal_(model.head[-1].weight, std=0.02)
        return model.requires_grad_(False)

    def test_explicit_loop_agrees_with_production_recursion(self):
        models = [self.model(), self.model()]
        before = self.inputs.history.clone()
        independent = sliding_forecast(models, self.inputs)["prediction"]
        with torch.inference_mode():
            expected = rollout(models, self.inputs).numpy()
        np.testing.assert_allclose(independent, expected, atol=1e-6)
        torch.testing.assert_close(before, self.inputs.history)

    def test_each_step_drops_oldest_and_appends_latest_prediction(self):
        result = sliding_forecast([PreviousLoadModel()], self.inputs, trace=True)
        q = result["prediction"]
        self.assertEqual(len(result["trace"]), 72)
        for hour in (1, 2, 24, 48, 71):
            history = result["trace"][hour]
            np.testing.assert_allclose(
                history[:, -1, 0], q[:, hour - 1] / self.inputs.scales.numpy()
            )
            np.testing.assert_allclose(
                history[:, 0, 0], result["trace"][hour - 1][:, 1, 0]
            )

    def test_measured_value_enters_only_after_forecasting_its_hour(self):
        observed = self.data.targets(self.hh, np.full(3, 48), 72)
        changed = observed.clone()
        changed[:, 20] += 8
        model = PreviousLoadModel()
        baseline = sliding_forecast([model], self.inputs, observed)["prediction"]
        modified = sliding_forecast([model], self.inputs, changed)["prediction"]
        np.testing.assert_array_equal(modified[:, :21], baseline[:, :21])
        self.assertTrue(np.all(modified[:, 21] > baseline[:, 21]))

    def test_refreshed_history_is_real_load_but_temperature_is_origin_value(self):
        measured = self.data.targets(self.hh, np.full(3, 48), 72)
        result = sliding_forecast(
            [PreviousLoadModel()], self.inputs, measured, trace=True
        )
        for hour in (1, 24, 71):
            np.testing.assert_allclose(
                result["trace"][hour][:, -1, 0],
                measured[:, hour - 1].numpy() / self.inputs.scales.numpy(),
            )
            np.testing.assert_array_equal(
                result["trace"][hour][:, -1, 2], self.inputs.history[:, -1, 2].numpy()
            )

    def test_price_free_probes_and_controlled_prices_are_invariant(self):
        model = self.model(False)
        observed = self.data.targets(self.hh, np.full(3, 48), 72)
        result = sliding_forecast([model], self.inputs, observed, True, 0.1, (-1, 2))
        self.assertEqual(float(np.max(np.abs(result["local_change"]))), 0)
        self.assertEqual(float(np.max(np.abs(result["endpoint_change"]))), 0)
        altered = replace(self.inputs, prices=self.inputs.prices + 10)
        np.testing.assert_array_equal(
            sliding_forecast([model], self.inputs)["prediction"],
            sliding_forecast([model], altered)["prediction"],
        )

    def test_price_probe_context_is_shared_and_weights_stay_frozen(self):
        model = self.model()
        originals = [p.clone() for p in model.parameters()]
        observed = self.data.targets(self.hh, np.full(3, 48), 72)
        ordinary = sliding_forecast([model], self.inputs, observed)["prediction"]
        probed = sliding_forecast([model], self.inputs, observed, True, 0.1, (-1, 2))
        np.testing.assert_array_equal(ordinary, probed["prediction"])
        self.assertTrue(
            all(torch.equal(p, v) for p, v in zip(model.parameters(), originals))
        )

    def test_invalid_measurements_are_rejected(self):
        with self.assertRaises(ValueError):
            sliding_forecast([PreviousLoadModel()], self.inputs, torch.zeros(3, 1))
        measured = torch.zeros(3, 72)
        measured[:, 0] = float("nan")
        with self.assertRaises(ValueError):
            sliding_forecast([PreviousLoadModel()], self.inputs, measured)

    def test_daily_metrics_do_not_mix_forecast_days(self):
        actual = np.ones((2, 3, 72), np.float32)
        predicted = actual.copy()
        predicted[:, :, 24:48] = 2
        predicted[:, :, 48:] = 3
        result = metrics(predicted, actual, np.array([True, True, True]))
        self.assertEqual(result["days"]["1"]["rmse"], 0)
        self.assertEqual(result["days"]["2"]["rmse"], 1)
        self.assertEqual(result["days"]["3"]["rmse"], 2)
        self.assertEqual(result["days"]["3"]["aggregate_energy_ratio"], 3)

    def test_independent_derived_features_match_raw_reconstruction(self):
        inputs = self.inputs
        demand = torch.tensor([0.05, 0.8, 1.4])
        updated = advance(
            inputs.history,
            demand,
            inputs.prices[:, 0],
            inputs.calendar[:, 0],
            inputs.history[:, -1, 2],
        )
        past = np.arange(25, 49)
        q = np.asarray(self.data.demand[self.hh[:, None], past]).copy()
        q[:, -1] = demand.numpy() * self.inputs.scales.numpy()
        prices = self.data.prices[self.data.area_index[self.hh, None], past]
        temp = np.asarray(
            self.data.temperature[self.data.area_index[self.hh, None], past]
        ).copy()
        temp[:, -1] = self.data.temperature[self.data.area_index[self.hh], 47]
        expected = self.data.features.history(
            q, prices, temp, self.data.times[past], self.hh
        )
        np.testing.assert_allclose(updated, expected, atol=2e-6)


if __name__ == "__main__":
    unittest.main()

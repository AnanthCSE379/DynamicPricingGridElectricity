"""Check information boundaries and shared recursion in load diagnostics."""

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from norway_energy_crisis.analysis.audit_load_support import inputs, predict
from norway_energy_crisis.preprocess import Preprocessor


class ScaledPersistence(torch.nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.temperatures = []

    def forward(self, history, static, next_price, target_calendar):
        self.temperatures.append(history[:, -1, 2].clone())
        return history[:, -1, 0] * self.factor


class LoadSupportTests(unittest.TestCase):
    def data(self):
        state = dict(
            household_ids=[1],
            demand_scales=[2],
            static_encoded=[[1]],
            price_mean=0,
            price_std=1,
            temperature_mean=0,
            temperature_std=1,
        )
        return SimpleNamespace(
            preprocessor=Preprocessor(state),
            static=np.ones((1, 1), np.float32),
            area_index=np.array([0]),
            demand=np.ones((1, 96), np.float32),
            prices=np.ones((1, 96), np.float32),
            temperature=np.arange(96, dtype=np.float32)[None],
            times=np.datetime64("2022-01-01", "ns").astype("int64")
            + np.arange(96, dtype=np.int64) * 3600 * 10**9,
        )

    def test_future_measurements_cannot_change_inputs(self):
        data = self.data()
        original = inputs(data, np.array([0]), np.array([24]), 72)
        data.demand[:, 24:] = 999
        data.temperature[:, 24:] = -999
        changed = inputs(data, np.array([0]), np.array([24]), 72)
        for key in original:
            np.testing.assert_array_equal(original[key], changed[key])

    def test_recursive_ensemble_uses_shared_mean_and_persisted_weather(self):
        data = self.data()
        values = inputs(data, np.array([0]), np.array([24]), 72)
        models = [ScaledPersistence(0.5), ScaledPersistence(1.5)]
        result = predict(models, values, "cpu")
        np.testing.assert_allclose(result, np.ones((1, 72)))
        for model in models:
            for temperature in model.temperatures:
                np.testing.assert_array_equal(temperature.numpy(), [23])

    def test_nonfinite_predictions_are_rejected(self):
        values = inputs(self.data(), np.array([0]), np.array([24]), 1)
        with self.assertRaises(ValueError):
            predict([ScaledPersistence(float("nan"))], values, "cpu")


if __name__ == "__main__":
    unittest.main()

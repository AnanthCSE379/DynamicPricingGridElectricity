"""Check sequence boundaries, training-only fills, joint inputs and prediction."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .data import batch_inputs, fit_scales, make_features, window_targets
from .model import HouseholdGRU
from .__main__ import predict


class HouseholdHistoryTests(unittest.TestCase):
    def test_entire_history_and_target_stay_in_one_partition(self):
        partitions = np.repeat([0, 1, 2], 15)
        demand = np.ones((2, 45), dtype=np.float32)
        windows = window_targets(partitions, demand)
        used = {}
        for name, targets in windows.items():
            self.assertEqual(len(targets), 5)
            used[name] = set((targets[:, None] + np.arange(-10, 1)).ravel())
        self.assertTrue(used["train"].isdisjoint(used["validation"]))
        self.assertTrue(used["train"].isdisjoint(used["test"]))
        self.assertTrue(used["validation"].isdisjoint(used["test"]))

    def test_training_only_mean_fills_inputs_with_observed_indicator(self):
        demand = np.array([[1., 3., np.nan, 999.], [0., 0., 0., 888.]], dtype=np.float32)
        means, maxima = fit_scales(demand, np.array([True, True, True, False]))
        np.testing.assert_allclose(means, [2., 0.])
        np.testing.assert_allclose(maxima, [3., 1.])
        features = make_features(demand, means, maxima)
        torch.testing.assert_close(features[2], torch.tensor([2 / 3, 0., 0., 1.]))
        self.assertTrue(np.isnan(demand[0, 2]))  # The actual target is never filled.

    def test_ten_steps_all_households_and_next_price_are_used(self):
        torch.manual_seed(123)
        model = HouseholdGRU(households=3, width=4)
        history = torch.rand(2, 10, 6, requires_grad=True)
        price = torch.tensor([0., 1.], requires_grad=True)
        output = model(history, price)
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertTrue(torch.all(output > 0))
        output.sum().backward()
        self.assertGreater(float(history.grad[:, 0, :3].abs().sum()), 0.)
        self.assertTrue(torch.all(history.grad[:, :, :3].abs().sum(dim=(0, 1)) > 0))
        self.assertGreater(float(price.grad.abs().sum()), 0.)

    def test_exact_window_alignment_and_inference_round_trip(self):
        demand = np.arange(36, dtype=np.float32).reshape(3, 12)
        means, maxima = fit_scales(demand, np.ones(12, dtype=bool))
        features = make_features(demand, means, maxima)
        history, price = batch_inputs(features, torch.arange(12, dtype=torch.float32), torch.tensor([10]))
        torch.testing.assert_close(history[0], features[:10])
        self.assertEqual(float(price[0]), 10.)
        model = HouseholdGRU(3, 4).eval()
        household_ids = ["a", "b", "c"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            torch.save({"weights": model.state_dict(), "household_ids": household_ids, "width": 4,
                        "household_means": torch.from_numpy(means), "household_maxima": torch.from_numpy(maxima)}, path)
            request = {"household_ids": household_ids, "demand_history_kwh": demand[:, :10].T.tolist(),
                       "next_price_pence_per_kwh": 3.99}
            result = predict(request, path)
            with torch.no_grad():
                expected = model(history, torch.tensor([0.]))[0].numpy() * maxima
            np.testing.assert_allclose(result["predicted_next_demand_kwh"], expected, rtol=1e-5)
            request["household_ids"] = household_ids[::-1]
            with self.assertRaisesRegex(ValueError, "household order"):
                predict(request, path)

    def test_metrics_exclude_missing_targets_and_match_baseline_pairs(self):
        from scripts.train_household_response import evaluate
        torch.manual_seed(321)
        demand = np.ones((3, 22), dtype=np.float32)
        demand[0, 10] = np.nan
        demand[1, 12] = np.nan
        means, maxima = fit_scales(demand, np.ones(22, dtype=bool))
        features = make_features(demand, means, maxima)
        values = torch.from_numpy(np.nan_to_num(demand, nan=0.).T.copy())
        known = torch.from_numpy(np.isfinite(demand).T.copy())
        targets = torch.arange(10, 22)
        groups = torch.arange(22) % 12
        prices = torch.zeros(22)
        model = HouseholdGRU(3, 4).eval()
        scores = evaluate(model, features, prices, values, known, torch.tensor(maxima), targets, groups)
        with torch.no_grad():
            predicted = model(*batch_inputs(features, prices, targets)) * torch.tensor(maxima)[None, :]
        observed_targets = known[targets]
        paired_targets = observed_targets & known[targets - 1]
        errors = predicted - values[targets]
        expected = float(torch.sqrt(errors[observed_targets].double().square().mean()))
        self.assertAlmostEqual(scores["gru_all_observed_targets"]["rmse_kwh"], expected, places=6)
        self.assertEqual(scores["gru_all_observed_targets"]["count"], int(observed_targets.sum()))
        self.assertEqual(scores["gru_persistence_comparable"]["count"], int(paired_targets.sum()))
        self.assertEqual(scores["persistence"]["count"], int(paired_targets.sum()))


if __name__ == "__main__":
    unittest.main()

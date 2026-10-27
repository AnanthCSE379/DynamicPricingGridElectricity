"""Exact shock timing, request bounds and information-boundary checks."""

import unittest

import numpy as np
import torch

from norway_energy_crisis.analysis.consumer_recovery import (
    HOURS,
    WARMUP,
    cases,
    permitted_inputs,
    run_sequence,
)
from norway_energy_crisis.analysis.test_load_support import ScaledPersistence
from norway_energy_crisis.analysis import test_load_support


class CalendarOnly(torch.nn.Module):
    def forward(self, history, static, next_price, target_calendar):
        return 0.5 + 0.5 * target_calendar[:, 0]


class ConsumerRecoveryTests(unittest.TestCase):
    def data(self):
        data = test_load_support.LoadSupportTests().data()
        data.households = 1
        data.demand = np.ones((1, 160), np.float32)
        data.prices = np.ones((1, 160), np.float32)
        data.temperature = np.arange(160, dtype=np.float32)[None]
        data.times = data.times[0] + np.arange(160, dtype=np.int64) * 3600 * 10**9
        return data

    def test_no_future_measurements_and_no_mutation(self):
        data = self.data()
        values = permitted_inputs(data, np.array([24]), 1)
        data.demand[:, 24:] = 99
        data.temperature[:, 24:] = -999
        data.prices[:, 24:] = 888
        changed = permitted_inputs(data, np.array([24]), 1)
        for key in values:
            np.testing.assert_array_equal(values[key], changed[key])
        history = values["history"].copy()
        run_sequence([ScaledPersistence(1)], values, "cpu", 0.1, WARMUP, 6)
        np.testing.assert_array_equal(values["history"], history)

    def test_exact_cut_timing_and_one_hour_prediction_lag(self):
        values = permitted_inputs(self.data(), np.array([24]), 1)
        request, consumed, _ = run_sequence(
            [ScaledPersistence(1)], values, "cpu", 0.05, WARMUP, 6
        )
        np.testing.assert_allclose(request[0, : WARMUP + 1], 1)
        np.testing.assert_allclose(request[0, WARMUP + 1 :], 0.05)
        np.testing.assert_allclose(consumed[0, WARMUP : WARMUP + 6], 0.05)
        np.testing.assert_array_equal(
            consumed[:, WARMUP + 6 :], request[:, WARMUP + 6 :]
        )
        self.assertEqual(request.shape, (1, HOURS))

    def test_caps_never_increase_consumption(self):
        values = permitted_inputs(self.data(), np.array([24]), 1)
        request, consumed, _ = run_sequence(
            [ScaledPersistence(1)], values, "cpu", 1.25, WARMUP, 1
        )
        np.testing.assert_array_equal(request, consumed)
        np.testing.assert_allclose(request, 1)

    def test_history_independent_model_recovers_immediately(self):
        values = permitted_inputs(self.data(), np.array([24]), 1)
        control, _, _ = run_sequence([CalendarOnly()], values, "cpu")
        request, consumed, _ = run_sequence(
            [CalendarOnly()], values, "cpu", 0.1, WARMUP, 6
        )
        np.testing.assert_array_equal(control, request)
        np.testing.assert_array_equal(
            consumed[:, WARMUP + 6 :], control[:, WARMUP + 6 :]
        )

    def test_shared_ensemble_and_warmup_fork_match_full_sequence(self):
        values = permitted_inputs(self.data(), np.array([24]), 1)
        models = [ScaledPersistence(0.5), ScaledPersistence(1.5)]
        full, _, _ = run_sequence(models, values, "cpu", 0.1, WARMUP, 1)
        from norway_energy_crisis.analysis.consumer_recovery import subset_hours

        warmup, _, history = run_sequence(
            models, subset_hours(values, 0, WARMUP), "cpu"
        )
        tail, _, _ = run_sequence(
            models, subset_hours(values, WARMUP, HOURS), "cpu", 0.1, 0, 1, history
        )
        np.testing.assert_array_equal(full, np.concatenate((warmup, tail), axis=1))

    def test_invalid_caps_and_nonfinite_predictions_fail(self):
        values = permitted_inputs(self.data(), np.array([24]), 1)
        for cap in (-1, float("nan")):
            with self.assertRaises(ValueError):
                run_sequence([ScaledPersistence(1)], values, "cpu", cap)
        with self.assertRaises(ValueError):
            run_sequence([ScaledPersistence(float("nan"))], values, "cpu")
        names = [case.name for case in cases()]
        self.assertEqual(len(names), len(set(names)))

    def test_recovery_metrics_distinguish_immediate_and_missing_recovery(self):
        from norway_energy_crisis.analysis.report_consumer_recovery import (
            household_results,
        )

        control = np.ones((1, 1, HOURS))
        request = control.copy()
        consumed = control.copy()
        consumed[:, :, WARMUP] = 0.1
        case = next(case for case in cases() if case.name == "cap0.1_normal1h")
        args = (np.array([0]), np.array([2]), np.array([1]))
        recovered = household_results(request, consumed, control, case, *args)
        self.assertTrue(recovered["affected"][0, 0])
        self.assertEqual(recovered["first_recovery_hours"][0, 0], 3)
        self.assertEqual(recovered["day3_ratio"][0, 0], 1)
        request[:, :, WARMUP + 1 :] = 0.1
        consumed[:, :, WARMUP + 1 :] = 0.1
        stuck = household_results(request, consumed, control, case, *args)
        self.assertFalse(stuck["recovered_by_72h"][0, 0])
        self.assertTrue(np.isnan(stuck["first_recovery_hours"][0, 0]))
        self.assertAlmostEqual(stuck["day3_ratio"][0, 0], 0.1)


if __name__ == "__main__":
    unittest.main()

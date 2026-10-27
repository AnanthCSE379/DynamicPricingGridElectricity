"""Consumer-model data boundaries, edge cases, gradients and experiment recovery."""

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from norway_energy_crisis.test_pipeline import fixture
from norway_energy_crisis.response_v2.data import (
    ConsumerData,
    FEATURES,
    ForecastInputs,
    TransitionSampler,
    origins_for,
)
from norway_energy_crisis.response_v2.model import (
    ConsumerModel,
    ConsumerSimulator,
    append_prediction,
    atomic_save,
    bundle,
    load_model,
    rollout,
)
from norway_energy_crisis.response_v2.evaluate import (
    actuals,
    baseline_predictions,
    paired_blocks,
    price_curves,
    sequence,
    summarize,
    verify_trained_raw_api,
)
from norway_energy_crisis.response_v2 import train


class GuardedHistory:
    """Fail if an inference helper attempts to read future measured values."""

    def __init__(self, array, origin):
        self.array, self.origin = array, origin

    def __getitem__(self, indices):
        times = np.asarray(indices[1])
        if times.max() >= self.origin:
            raise AssertionError("Future measurement access")
        return self.array[indices]


class AffineConsumer(torch.nn.Module):
    def __init__(self, multiplier, offset):
        super().__init__()
        self.multiplier, self.offset = multiplier, offset

    def forward(self, history, static, price, target_calendar, price_enabled=True):
        return history[:, -1, 0] * self.multiplier + self.offset


class ResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        with redirect_stdout(io.StringIO()):
            _, cls.cleaned, cls.prepared = fixture(cls.root)
        cls.data = ConsumerData(cls.cleaned, cls.prepared, cls.root / "prepared_v2")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def inputs(self, horizon=6):
        return self.data.inputs(
            self.data.primary, np.full(len(self.data.primary), 48), horizon
        )

    def model(self, **kwargs):
        return ConsumerModel(self.data.features.static.shape[1], **kwargs).eval()

    def test_all_households_keep_primary_order_and_training_scales(self):
        self.assertEqual(len(self.data.ids), 3)
        self.assertEqual(self.data.primary.tolist(), [0, 1])
        old = json.loads((self.prepared / "preprocessing.json").read_text())
        np.testing.assert_array_equal(
            self.data.features.scales[:2], old["demand_scales"]
        )
        self.assertEqual(self.data.state["price_mean"], old["price_mean"])
        self.assertEqual(self.data.state["history_features"], FEATURES)
        self.assertEqual(len(self.data.state["static_encoded"]), 3)

    def test_raw_units_negative_prices_and_zero_are_preserved(self):
        indices = np.arange(24)
        hh = np.array([0])
        demand = np.asarray(self.data.demand[hh[:, None], indices]).copy()
        price = np.asarray(self.data.prices[np.array([0])[:, None], indices]).copy()
        temp = np.asarray(self.data.temperature[np.array([0])[:, None], indices])
        values = self.data.features.history(
            demand, price, temp, self.data.times[indices], hh
        )
        self.assertEqual(float(values[0, 0, 0]), 0)
        recovered = (
            values[0, 0, 1] * self.data.state["price_std"]
            + self.data.state["price_mean"]
        )
        self.assertAlmostEqual(float(recovered), -0.01972, places=6)
        self.assertTrue(np.isfinite(values).all())

    def test_derived_features_match_raw_reconstruction_after_append(self):
        origin = 48
        inputs = self.inputs(1)
        scaled = torch.tensor([0.05, 0.8])
        appended = append_prediction(
            inputs.history,
            scaled,
            inputs.prices[:, 0],
            inputs.history[:, -1, 2],
            inputs.calendar[:, 0],
        )
        h = self.data.primary
        past = np.arange(origin - 23, origin + 1)
        q = np.asarray(self.data.demand[h[:, None], past]).copy()
        q[:, -1] = scaled.numpy() * self.data.features.scales[h]
        price = np.asarray(self.data.prices[self.data.area_index[h, None], past])
        temp = np.asarray(
            self.data.temperature[self.data.area_index[h, None], past]
        ).copy()
        temp[:, -1] = self.data.temperature[self.data.area_index[h], origin - 1]
        expected = self.data.features.history(q, price, temp, self.data.times[past], h)
        np.testing.assert_allclose(appended, expected, atol=2e-6)

    def test_forecast_input_type_has_no_targets_or_future_weather(self):
        inputs = self.inputs()
        self.assertIsInstance(inputs, ForecastInputs)
        self.assertEqual(
            set(vars(inputs)), {"history", "static", "scales", "prices", "calendar"}
        )
        original_q, original_temp = self.data.demand, self.data.temperature
        try:
            self.data.demand = GuardedHistory(original_q, 48)
            self.data.temperature = GuardedHistory(original_temp, 48)
            safe = self.inputs()
            torch.testing.assert_close(safe.history, inputs.history)
        finally:
            self.data.demand, self.data.temperature = original_q, original_temp

    def test_partition_windows_and_common_origins(self):
        for number, split in enumerate(("train", "validation", "test")):
            for t in self.data.origins[split]:
                self.assertTrue(np.all(self.data.partitions[t - 24 : t + 72] == number))
        boundary = int(np.flatnonzero(self.data.partitions == 1)[0])
        with self.assertRaises(ValueError):
            self.data.inputs([0], [boundary], 48)
        for horizon in (1, 24, 48, 72):
            other = origins_for(self.data.partitions, 1, horizon)
            self.assertTrue(set(self.data.origins["validation"]).issubset(other))

    def test_missing_values_shapes_ids_and_times_rejected(self):
        builder = self.data.features
        hh = [self.data.ids[0]]
        q = np.ones((1, 24), np.float32)
        p = np.ones_like(q)
        times = self.data.times[24:48]
        builder.inputs(q, p, p, times, hh, np.zeros((1, 3)))
        for bad in (np.nan, np.inf, -1):
            invalid = q.copy()
            invalid[0, -1] = bad
            with self.assertRaises(ValueError):
                builder.inputs(invalid, p, p, times, hh, np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            builder.inputs(q, p, p, times, [999999], np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            builder.inputs(q, p, p, times[::-1], hh, np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            builder.inputs(q, p, p, times, hh, np.zeros(3))

    def test_direct_head_can_leave_zero_without_a_floor(self):
        model = self.model()
        inputs = self.inputs()
        zero = inputs.history.clone()
        zero[:, :, 0], zero[:, :, 9] = 0, 0
        predicted = model(
            zero, inputs.static, inputs.prices[:, 0], inputs.calendar[:, 0]
        )
        self.assertTrue((predicted > 0.1).all())
        legacy = self.model(head="persistence")
        old = legacy(zero, inputs.static, inputs.prices[:, 0], inputs.calendar[:, 0])
        self.assertTrue((old < 1e-5).all())

    def test_finite_nonnegative_outputs_and_encoder_gradients(self):
        for head in ("direct", "persistence"):
            model = self.model(head=head)
            torch.nn.init.normal_(model.head[-1].weight, std=0.01)
            predicted = rollout(model, self.inputs(48))
            self.assertTrue(torch.isfinite(predicted).all())
            self.assertTrue((predicted >= 0).all())
            predicted.square().mean().backward()
            self.assertTrue(
                all(
                    p.grad is not None and torch.isfinite(p.grad).all()
                    for p in model.parameters()
                )
            )
            self.assertGreater(float(model.projection.weight.grad.abs().sum()), 0)

    def test_price_free_masks_all_derived_price_features(self):
        inputs = self.inputs()
        changed = inputs.history.clone()
        changed[:, :, 1] += 40
        changed[:, :, 10] -= 50
        altered = replace(inputs, history=changed, prices=inputs.prices + 100)
        for permanent in (True, False):
            model = self.model(use_price=not permanent)
            torch.nn.init.normal_(model.head[-1].weight, std=0.03)
            kwargs = {} if permanent else {"price_enabled": False}
            with torch.inference_mode():
                torch.testing.assert_close(
                    rollout(model, inputs, **kwargs),
                    rollout(model, altered, **kwargs),
                    rtol=0,
                    atol=0,
                )

    def test_ensemble_advances_average_not_separate_histories(self):
        inputs = self.inputs(2)
        history = inputs.history.clone()
        history[:, :, 0] = 2
        inputs = replace(inputs, history=history, scales=torch.ones(2))
        predicted = rollout([AffineConsumer(1, 1), AffineConsumer(2, 0)], inputs)
        torch.testing.assert_close(predicted, torch.tensor([[3.5, 5.75], [3.5, 5.75]]))

    def test_origin_temperature_persists_recursively(self):
        seen = []

        class Capture(AffineConsumer):
            def forward(self, history, *args, **kwargs):
                seen.append(history[:, -1, 2].clone())
                return super().forward(history, *args, **kwargs)

        inputs = self.inputs(6)
        rollout(Capture(1, 0), inputs)
        self.assertTrue(all(torch.equal(x, seen[0]) for x in seen))

    def test_diagnostic_cut_never_raises_demand_and_only_changes_later_requests(self):
        inputs = self.inputs(36)
        model = AffineConsumer(1, 0.01)
        q0, _ = sequence([model], inputs, 0, 1)
        q, use = sequence([model], inputs, 0, 1, cap=0.05, cut_hours=1)
        torch.testing.assert_close(q[:, :25], q0[:, :25])
        self.assertTrue(torch.equal(q[:, :24], use[:, :24]))
        self.assertTrue(torch.equal(q[:, 25:], use[:, 25:]))
        self.assertTrue((use[:, 24] <= 0.05).all())
        self.assertTrue((q[:, 25] < q0[:, 25]).all())

    def test_importance_sampler_has_correct_target_distribution(self):
        data = deepcopy(self.data)
        demand = np.array(data.demand)
        demand[0, 100], demand[0, 101] = 0, 2
        demand[1, 200], demand[1, 201] = 0.05, 2
        data.demand = demand
        sampler = TransitionSampler(data, data.primary, 42)
        h, t, w = sampler.sample(100000)
        self.assertGreater(int((t == 101).sum()), 1000)
        self.assertAlmostEqual(float(w.mean()), 1, delta=0.02)
        # Importance-weighted expectation of an arbitrary target statistic.
        actual = float((w * torch.tensor((h + 1) * (t % 7))).mean())
        expected = float(
            np.mean((data.primary[:, None] + 1) * (sampler.origins[None] % 7))
        )
        self.assertAlmostEqual(actual, expected, delta=0.08)
        self.assertTrue(np.all(data.partitions[t] == 0))

    def test_price_curves_report_variety_without_forcing_it(self):
        model = self.model(use_price=False)
        curve = price_curves([model], self.data, [48], "cpu")
        self.assertEqual(len(curve["grid"]), 21)
        self.assertEqual(curve["flat_response_fraction"], 1)
        self.assertEqual(max(curve["household_relative_span_quantiles"]), 0)

    def test_checkpoint_simulator_roundtrip_and_schema_rejection(self):
        model = self.model()
        path = self.root / "roundtrip.pt"
        atomic_save(bundle(model, self.data), path)
        restored, saved = load_model(path)
        self.assertTrue(all(not p.requires_grad for p in restored.parameters()))
        with torch.inference_mode():
            torch.testing.assert_close(
                rollout(model, self.inputs()), rollout(restored, self.inputs())
            )
        sim = ConsumerSimulator([path])
        h = self.data.primary
        times = np.arange(24, 48)
        pred = sim.forecast(
            self.data.demand[h[:, None], times],
            self.data.prices[self.data.area_index[h, None], times],
            self.data.temperature[self.data.area_index[h, None], times],
            self.data.times[times],
            [self.data.ids[i] for i in h],
            self.data.prices[self.data.area_index[h, None], np.arange(48, 54)],
        )
        with torch.inference_mode():
            np.testing.assert_allclose(pred, rollout(restored, self.inputs()).numpy())
        saved["feature_order"] = ["bad"]
        atomic_save(saved, path)
        with self.assertRaises(ValueError):
            load_model(path)

    def test_baselines_do_not_use_future_demand_and_bootstrap_is_paired(self):
        days = self.data.origins["validation"][:2]
        baselines = baseline_predictions(self.data, days, 72)
        np.testing.assert_array_equal(
            baselines["previous_day"][:, :, :24], baselines["previous_day"][:, :, 24:48]
        )
        truth = actuals(self.data, days, 72)
        result = summarize(baselines["previous_day"], truth, self.data, days)
        self.assertEqual(len(result["rmse_by_horizon"]), 72)
        paired = paired_blocks([1, 2, 3], [1, 2, 3])
        self.assertEqual(paired["ci95"], [0, 0])

    def test_completed_update_recovery_and_changed_config_rejection(self):
        folder = self.root / "resume"
        reference = {
            "rmse_24h": 1000,
            "regions": {
                r: {"aggregate_rmse_24h": 1000} for r in self.data.state["areas"]
            },
        }
        original = train.rollout
        calls = [0]

        def interrupted(model, inputs, *args, **kwargs):
            if model.training:
                calls[0] += 1
                if calls[0] == 3:
                    raise RuntimeError("interrupted after completed updates")
            return original(model, inputs, *args, **kwargs)

        with redirect_stdout(io.StringIO()):
            with patch.object(train, "rollout", interrupted):
                with self.assertRaises(RuntimeError):
                    train.train_run(
                        self.data,
                        folder,
                        "B",
                        42,
                        True,
                        "cpu",
                        1000,
                        0.3,
                        reference,
                        reference,
                        batch_size=2,
                        smoke_updates=4,
                    )
            state = torch.load(folder / "active.pt", weights_only=False)
            self.assertEqual(state["updates"], 2)
            result = train.train_run(
                self.data,
                folder,
                "B",
                42,
                True,
                "cpu",
                1000,
                0.3,
                reference,
                reference,
                batch_size=2,
                smoke_updates=4,
            )
        self.assertEqual(result["updates"], 4)
        self.assertFalse((folder / "active.pt").exists())
        changed = self.root / "changed"
        changed.mkdir()
        (changed / "config.json").write_text("{}")
        with self.assertRaises(ValueError):
            train.train_run(
                self.data,
                changed,
                "B",
                42,
                True,
                "cpu",
                1000,
                0.3,
                reference,
                reference,
                batch_size=2,
                smoke_updates=2,
            )

    def test_generated_comparison_freezes_selection_and_writes_report(self):
        data = deepcopy(self.data)
        data.origins = {name: values[:2] for name, values in data.origins.items()}
        output = self.root / "generated_comparison"
        output.mkdir()
        reference = {
            "rmse_24h": 1000,
            "regions": {r: {"aggregate_rmse_24h": 1000} for r in data.state["areas"]},
        }
        preflight = {
            "initial_mean": 0.3,
            "incumbent_subset": reference,
            "incumbent_validation": reference,
        }
        real_train = train.train_run

        def short_fit(*args, **kwargs):
            kwargs.update(batch_size=2, smoke_updates=2)
            return real_train(*args, **kwargs)

        with redirect_stdout(io.StringIO()):
            with patch.object(train, "train_run", short_fit):
                train.train_experiment(data, output, "cpu", 1000, preflight, None)
            selection = json.loads((output / "selection.json").read_text())
            self.assertTrue(selection["frozen_before_followup"])
            with patch.object(train, "incumbents", return_value=[AffineConsumer(1, 0)]):
                with patch.object(train, "sequence_suite", return_value={}):
                    with patch.object(train, "verify_trained_raw_api", return_value={}):
                        train.evaluate_experiment(data, output, "cpu", self.root)
        self.assertEqual(len(list(output.glob("*/result.json"))), 8)
        self.assertTrue((output / "REPORT.md").is_file())
        metrics = json.loads((output / "evaluation.json").read_text())
        self.assertTrue(metrics["acceptance"]["price_free_invariance"])
        self.assertFalse(metrics["acceptance"]["causal_validation"])
        altered = json.loads((output / "selection.json").read_text())
        self.assertEqual(selection, altered)

    def test_final_raw_api_check_reconstructs_full_sequence(self):
        model = self.model()
        folder = self.root / "raw_check"
        folder.mkdir()
        checkpoint = folder / "model.pt"
        atomic_save(bundle(model, self.data), checkpoint)
        origin = int(self.data.origins["validation"][0])
        inputs = self.data.inputs(self.data.primary, np.full(2, origin), 120)
        normal = 0.5
        standardized = float(self.data.features.base.price(normal))
        prediction, _ = sequence([model], inputs, standardized, standardized)
        np.savez_compressed(folder / "normal.npz", requested=prediction.numpy()[None])
        (folder / "metrics.json").write_text(
            json.dumps(
                {
                    "origins": [origin],
                    "normal_price_nok_kwh": normal,
                }
            )
        )
        checked = verify_trained_raw_api([checkpoint], self.data, folder)
        self.assertTrue(checked["raw_api_consistent"])
        self.assertEqual(checked["hours"], 120)

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available")
    def test_mps_output_and_gradient_parity(self):
        model = self.model()
        torch.nn.init.normal_(model.head[-1].weight, std=0.01)
        accelerator = deepcopy(model).to("mps")
        inputs = self.inputs(6)
        cpu = rollout(model, inputs)
        mps = rollout(accelerator, inputs.to("mps"))
        torch.testing.assert_close(mps.cpu(), cpu, rtol=3e-3, atol=3e-4)
        cpu.mean().backward()
        mps.mean().backward()
        for a, b in zip(model.parameters(), accelerator.parameters()):
            torch.testing.assert_close(a.grad, b.grad.cpu(), rtol=3e-2, atol=3e-3)


if __name__ == "__main__":
    unittest.main()

"""Paired sudden-curtailment probes of frozen consumer predictors.

Run a normal-price warmup, restrict consumption, and release the restriction.
Feedback of restricted consumption is intentionally diagnostic, not an assertion
that the predictors learned outage behavior. No model is fitted or modified.
"""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.preprocess import PreparedData, calendar
from norway_energy_crisis.train_response import load_model, Predictor


WARMUP = 24
RECOVERY = 72
HOURS = WARMUP + 6 + RECOVERY
CAPS = (0.05, 0.1, 0.75, 1.25)
STARTS = (
    "2022-01-08T00",
    "2022-01-22T18",
    "2022-02-09T00",
    "2022-02-23T18",
    "2022-03-09T00",
    "2022-03-23T18",
)


@dataclass(frozen=True)
class Case:
    name: str
    cap_kw: float | None
    cut_hours: int
    high_price: bool = False

    @property
    def control(self):
        return "price_only6" if self.high_price else "control"


def cases():
    result = [Case("control", None, 0), Case("price_only6", None, 6, True)]
    for duration, high_price in ((1, False), (6, False), (6, True)):
        for cap in CAPS:
            name = f"cap{cap:g}_{'high' if high_price else 'normal'}{duration}h"
            result.append(Case(name, cap, duration, high_price))
    return result


def permitted_inputs(data, origins, normal_price, households=None):
    """Copy historical measurements and deterministic future calendar only."""
    selected = (
        np.arange(data.households) if households is None else np.asarray(households)
    )
    household_indices = np.tile(selected, len(origins))
    repeated = np.repeat(origins, len(selected))
    history_indices = repeated[:, None] + np.arange(-24, 0)
    areas = data.area_index[household_indices]
    return dict(
        history=data.preprocessor.history_features(
            data.demand[household_indices[:, None], history_indices],
            data.prices[areas[:, None], history_indices],
            data.temperature[areas[:, None], history_indices],
            data.times[history_indices],
            household_indices,
        ),
        static=np.asarray(data.static[household_indices]),
        scales=data.preprocessor.scales[household_indices],
        prices=np.full(
            (len(repeated), HOURS), data.preprocessor.price(normal_price), np.float32
        ),
        calendar=calendar(data.times[repeated[:, None] + np.arange(HOURS)]),
    )


@torch.inference_mode()
def run_sequence(
    models, values, device, cap_kw=None, cut_start=0, cut_hours=0, initial_history=None
):
    """Return requested and consumed kWh, plus the final normalized history."""
    require(cap_kw is None or np.isfinite(cap_kw) and cap_kw >= 0, "Invalid cut cap.")
    require(cut_start >= 0 and cut_hours >= 0, "Invalid cut interval.")
    tensors = {
        key: torch.as_tensor(np.array(value, copy=True), device=device)
        for key, value in values.items()
    }
    history = tensors["history"] if initial_history is None else initial_history.clone()
    temperature = history[:, -1, 2:3]
    requested, consumed = [], []
    for hour in range(tensors["prices"].shape[1]):
        price, target_calendar = (
            tensors["prices"][:, hour],
            tensors["calendar"][:, hour],
        )
        scaled = torch.stack(
            [
                model(history, tensors["static"], price, target_calendar)
                for model in models
            ]
        ).mean(0)
        request = scaled * tensors["scales"]
        use = (
            request.clamp_max(cap_kw)
            if cap_kw is not None and cut_start <= hour < cut_start + cut_hours
            else request
        )
        requested.append(request)
        consumed.append(use)
        row = torch.cat(
            (
                use[:, None] / tensors["scales"][:, None],
                price[:, None],
                temperature,
                target_calendar,
            ),
            dim=1,
        )
        history = torch.cat((history[:, 1:], row[:, None]), dim=1)
    request = torch.stack(requested, dim=1).cpu().numpy()
    use = torch.stack(consumed, dim=1).cpu().numpy()
    require(
        np.isfinite(request).all()
        and (request >= 0).all()
        and np.isfinite(use).all()
        and (use >= 0).all(),
        "Invalid trajectory.",
    )
    return request, use, history


def subset_hours(values, start, stop):
    return {
        key: value[:, start:stop] if key in ("prices", "calendar") else value
        for key, value in values.items()
    }


def load_models(data, responses, device):
    models, hashes = {}, {}
    for seed, kind in (
        (42, "price"),
        (7, "price"),
        (2026, "price"),
        (42, "no_price"),
        (7, "no_price"),
    ):
        path = responses / f"transformer_24h_seed{seed}_{kind}/best.pt"
        model, bundle = load_model(path)
        require(
            bundle["preprocessing"] == data.state
            and bundle["prepared_manifest_sha256"]
            == digest(data.folder / "manifest.json"),
            "Response model/data mismatch.",
        )
        models[seed, kind] = model.to(device).eval().requires_grad_(False)
        hashes[str(path)] = digest(path)
    return (
        dict(
            ensemble=[models[42, "price"], models[7, "price"]],
            reserved=[models[2026, "price"]],
            s42=[models[42, "price"]],
            s7=[models[7, "price"]],
            price_free=[models[42, "no_price"], models[7, "no_price"]],
        ),
        hashes,
    )


def execute(data, origins, engines, normal_price, device, output, days_per_batch):
    all_cases = cases()
    plans = {
        "ensemble": (np.arange(len(origins)), all_cases),
        "reserved": (np.arange(len(origins)), all_cases),
        "s42": (
            np.array([0, 2, 4]),
            [
                case
                for case in all_cases
                if case.name == "control" or case.cut_hours == 1
            ],
        ),
        "s7": (
            np.array([0, 2, 4]),
            [
                case
                for case in all_cases
                if case.name == "control" or case.cut_hours == 1
            ],
        ),
        "price_free": (
            np.array([0, 2, 4]),
            [
                case
                for case in all_cases
                if case.name
                in ("control", "price_only6", "cap0.1_normal6h", "cap0.1_high6h")
            ],
        ),
    }
    manifest = dict(cases=[asdict(case) for case in all_cases], engines={})
    for engine, (indices, engine_cases) in plans.items():
        folder = output / engine
        folder.mkdir()
        selected_origins = origins[indices]
        requests = {case.name: [] for case in engine_cases}
        consumptions = {case.name: [] for case in engine_cases}
        for start in range(0, len(indices), days_per_batch):
            chosen = selected_origins[start : start + days_per_batch]
            values = permitted_inputs(data, chosen, normal_price)
            warm_request, warm_use, warm_history = run_sequence(
                engines[engine], subset_hours(values, 0, WARMUP), device
            )
            for case in engine_cases:
                tail = subset_hours(values, WARMUP, HOURS)
                tail["prices"] = tail["prices"].copy()
                if case.high_price:
                    tail["prices"][:, : case.cut_hours] = data.preprocessor.price(2.0)
                request, use, _ = run_sequence(
                    engines[engine],
                    tail,
                    device,
                    case.cap_kw,
                    0,
                    case.cut_hours,
                    warm_history,
                )
                shape = (len(chosen), data.households, HOURS)
                requests[case.name].append(
                    np.concatenate((warm_request, request), axis=1).reshape(shape)
                )
                consumptions[case.name].append(
                    np.concatenate((warm_use, use), axis=1).reshape(shape)
                )
            save_json(
                output / "status.json",
                dict(
                    stage="simulation",
                    engine=engine,
                    days_complete=min(start + days_per_batch, len(indices)),
                    days=len(indices),
                ),
            )
            print(
                f"{engine}: {min(start + days_per_batch, len(indices))}"
                f"/{len(indices)} starting days",
                flush=True,
            )
        for case in engine_cases:
            np.savez_compressed(
                folder / f"{case.name}.npz",
                origins=selected_origins,
                requested=np.concatenate(requests[case.name]),
                consumed=np.concatenate(consumptions[case.name]),
            )
        manifest["engines"][engine] = dict(
            origin_indices=indices.tolist(), cases=[case.name for case in engine_cases]
        )
        save_json(output / "trajectories.json", manifest)
    return manifest


def independent_cpu_sequence(
    data, origin, household_indices, normal_price, case, responses
):
    """Rebuild raw histories through Predictor, independently of tensor recursion."""
    predictors = [
        Predictor(responses / f"transformer_24h_seed{seed}_price/best.pt")
        for seed in (42, 7)
    ]
    times = np.asarray(data.times[origin - 24 : origin]).astype("datetime64[ns]")
    areas = data.area_index[household_indices]
    demand = np.array(data.demand[household_indices, origin - 24 : origin], copy=True)
    price_history = np.array(data.prices[areas, origin - 24 : origin], copy=True)
    temperature = np.array(data.temperature[areas, origin - 24 : origin], copy=True)
    ids = [data.preprocessor.ids[i] for i in household_indices]
    predictions, consumptions = [], []
    for hour in range(HOURS):
        cut = WARMUP <= hour < WARMUP + case.cut_hours
        price = 2.0 if case.high_price and cut else normal_price
        request = np.mean(
            [
                predictor.predict(demand, price_history, temperature, times, ids, price)
                for predictor in predictors
            ],
            axis=0,
        )
        use = (
            np.minimum(request, case.cap_kw)
            if case.cap_kw is not None and cut
            else request
        )
        predictions.append(request)
        consumptions.append(use)
        demand = np.column_stack((demand[:, 1:], use))
        price_history = np.column_stack(
            (price_history[:, 1:], np.full(len(ids), price, np.float32))
        )
        temperature = np.column_stack((temperature[:, 1:], temperature[:, -1]))
        times = np.r_[times[1:], times[-1] + np.timedelta64(1, "h")]
    return np.stack(predictions, axis=1), np.stack(consumptions, axis=1)


def verify(data, origins, engines, normal_price, device, output, responses, manifest):
    results = dict(semantics={}, repeat={}, independent_cpu={})
    for engine, plan in manifest["engines"].items():
        folder = output / engine
        for case in cases():
            if case.name not in plan["cases"]:
                continue
            saved = np.load(folder / f"{case.name}.npz")
            request, use = saved["requested"], saved["consumed"]
            require(
                np.isfinite(request).all()
                and (request >= 0).all()
                and (use <= request).all()
                and (use >= 0).all(),
                "Invalid saved trajectory.",
            )
            np.testing.assert_array_equal(request[:, :, :WARMUP], use[:, :, :WARMUP])
            end = WARMUP + case.cut_hours
            np.testing.assert_array_equal(request[:, :, end:], use[:, :, end:])
            if case.cap_kw is not None:
                expected = np.minimum(request[:, :, WARMUP:end], case.cap_kw)
                np.testing.assert_array_equal(use[:, :, WARMUP:end], expected)
                control = np.load(folder / f"{case.control}.npz")["requested"]
                np.testing.assert_array_equal(
                    request[:, :, : WARMUP + 1], control[:, :, : WARMUP + 1]
                )
                affected = np.any(
                    use[:, :, WARMUP:end] < request[:, :, WARMUP:end], axis=2
                )
                np.testing.assert_allclose(
                    request[~affected], control[~affected], atol=1e-5, rtol=1e-5
                )
            results["semantics"][f"{engine}/{case.name}"] = True
    # Repeat all 717 households on every primary starting day using smaller batches.
    case = next(case for case in cases() if case.name == "cap0.05_normal1h")
    saved = np.load(output / "ensemble" / f"{case.name}.npz")["requested"]
    for attempt in (1, 2):
        differences = []
        for day, origin in enumerate(origins):
            values = permitted_inputs(data, np.array([origin]), normal_price)
            request, _, _ = run_sequence(
                engines["ensemble"], values, device, case.cap_kw, WARMUP, case.cut_hours
            )
            difference = np.abs(request - saved[day])
            np.testing.assert_allclose(request, saved[day], atol=0.0002, rtol=0.0002)
            differences.append(float(difference.max()))
        results["repeat"][str(attempt)] = dict(
            maximum_absolute_difference_kwh=max(differences),
            all_households=True,
            all_origins=True,
            passed=True,
        )
        print(f"Full-cohort repeat {attempt} passed", flush=True)
    # Twelve households span low/high training loads and both regional pools.
    order = np.argsort(data.preprocessor.scales)
    selected = np.unique(
        np.r_[
            order[np.linspace(0, len(order) - 1, 10).astype(int)],
            np.flatnonzero(data.area_index == 0)[0],
            np.flatnonzero(data.area_index == 1)[0],
        ]
    )
    for name in ("control", "cap0.05_normal1h", "cap1.25_high6h"):
        case = next(case for case in cases() if case.name == name)
        reference, consumed = independent_cpu_sequence(
            data, int(origins[0]), selected, normal_price, case, responses
        )
        saved = np.load(output / "ensemble" / f"{case.name}.npz")
        expected = saved["requested"][0, selected]
        np.testing.assert_allclose(reference, expected, atol=0.0002, rtol=0.0002)
        np.testing.assert_allclose(
            consumed, saved["consumed"][0, selected], atol=0.0002, rtol=0.0002
        )
        results["independent_cpu"][name] = dict(
            households=len(selected),
            maximum_absolute_difference_kwh=float(np.abs(reference - expected).max()),
            passed=True,
        )
        print(f"Independent raw-input CPU verification {name} passed", flush=True)
    # The price-free predictor must be unchanged by the candidate price pulse.
    control = np.load(output / "price_free/control.npz")["requested"]
    price_only = np.load(output / "price_free/price_only6.npz")["requested"]
    low = np.load(output / "price_free/cap0.1_normal6h.npz")["requested"]
    high = np.load(output / "price_free/cap0.1_high6h.npz")["requested"]
    np.testing.assert_array_equal(control, price_only)
    np.testing.assert_array_equal(low, high)
    results["price_free_invariance"] = True
    results["passed"] = True
    save_json(output / "verification.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument("--responses", type=Path, default=ROOT / "runs/comparison_v1")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "runs/consumer_recovery_v1"
    )
    parser.add_argument("--device", choices=("mps", "cpu"), default="mps")
    parser.add_argument("--days-per-batch", type=int, default=2)
    args = parser.parse_args()
    require(not args.output.exists(), "Choose a new output directory.")
    require(args.days_per_batch > 0, "Batch size must be positive.")
    if args.device == "mps":
        require(
            torch.backends.mps.is_available(),
            "MPS unavailable; no silent CPU fallback.",
        )
    torch.set_num_threads(4)
    data = PreparedData(args.data)
    origins = np.searchsorted(
        data.times, np.array(STARTS, dtype="datetime64[ns]").astype("int64")
    )
    for origin in origins:
        require(
            np.all(
                data.partitions[origin - 24 : origin + HOURS] == data.partitions[origin]
            ),
            "History and complete experiment must stay within one partition.",
        )
    weights = np.bincount(data.area_index)
    normal_price = float(
        np.median(np.repeat(data.prices[:, data.partitions == 0], weights, axis=0))
    )
    args.output.mkdir(parents=True)
    started = time.monotonic()
    engines, hashes = load_models(data, args.responses, args.device)
    source_paths = [Path(__file__), ROOT / "preprocess.py", ROOT / "train_response.py"]
    provenance = dict(
        source_hashes={str(path): digest(path) for path in source_paths},
        response_hashes=hashes,
        prepared_manifest=digest(data.folder / "manifest.json"),
        starts=STARTS,
        origin_indices=origins.tolist(),
        households=data.households,
        household_ids=data.preprocessor.ids,
        normal_price_nok_kwh=normal_price,
        high_price_nok_kwh=2.0,
        warmup_hours=WARMUP,
        recovery_hours=RECOVERY,
        total_hours=HOURS,
        caps_kw=CAPS,
        device=args.device,
        normal_definition="Cohort-weighted training median regional spot price.",
        temperature="Persist last origin temperature; no future measured weather.",
        scope="Retrospective frozen-model sensitivity, not causal outage validation.",
        versions=dict(torch=torch.__version__, numpy=np.__version__),
    )
    save_json(args.output / "provenance.json", provenance)
    manifest = execute(
        data,
        origins,
        engines,
        normal_price,
        args.device,
        args.output,
        args.days_per_batch,
    )
    save_json(args.output / "status.json", dict(stage="independent_verification"))
    verify(
        data,
        origins,
        engines,
        normal_price,
        args.device,
        args.output,
        args.responses,
        manifest,
    )
    for path, expected in {**provenance["source_hashes"], **hashes}.items():
        require(
            digest(path) == expected, "Source or frozen model changed during testing."
        )
    save_json(
        args.output / "status.json",
        dict(stage="verified", elapsed_seconds=time.monotonic() - started),
    )
    print(f"Verified simulations saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()

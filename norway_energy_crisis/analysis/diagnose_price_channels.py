"""Separate model price-level, price-change and historical-price channels."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from norway_energy_crisis.clean_data import ROOT, digest, require, save_json
from norway_energy_crisis.analysis.investigate_rollout import advance, selected_origins
from norway_energy_crisis.response_v2.data import ConsumerData
from norway_energy_crisis.response_v2.model import ConsumerSimulator, load_model


@torch.inference_mode()
def decompose(models, inputs, actual, normal, high):
    """Clamp loads to the same measured history; perturb model price inputs only.

    Intermediate feature combinations are algebraic ablations, not physically
    consistent price policies. The ordering is absolute price, change, history.
    """
    require(
        all(m.config["head"] == "direct" and m.config["use_price"] for m in models),
        "This decomposition requires the selected direct price-aware head.",
    )
    histories = [inputs.history.clone(), inputs.history.clone()]
    temperature = inputs.history[:, -1, 2]
    rows = {k: [] for k in ("normal", "high", "absolute", "change", "history", "total")}
    for hour in range(72):
        p0 = torch.full_like(inputs.prices[:, hour], normal)
        p1 = torch.full_like(p0, high)
        calendar = inputs.calendar[:, hour]
        difference0 = p0 - histories[0][:, -1, 1]
        difference1 = p1 - histories[1][:, -1, 1]
        values = []
        for model in models:
            z0 = model.encode(histories[0])
            z1 = model.encode(histories[1])

            def head(z, p, change):
                x = torch.cat(
                    (z, inputs.static, p[:, None], change[:, None], calendar), -1
                )
                return F.softplus(model.head(x)[:, 0]) * inputs.scales

            a = head(z0, p0, difference0)
            b = head(z0, p1, difference0)
            c = head(z0, p1, difference1)
            d = head(z1, p1, difference1)
            torch.testing.assert_close(
                a,
                model.predict_encoded(z0, histories[0], inputs.static, p0, calendar)
                * inputs.scales,
            )
            torch.testing.assert_close(
                d,
                model.predict_encoded(z1, histories[1], inputs.static, p1, calendar)
                * inputs.scales,
            )
            values.append(torch.stack((a, b, c, d)))
        a, b, c, d = torch.stack(values).mean(0)
        for key, value in {
            "normal": a,
            "high": d,
            "absolute": b - a,
            "change": c - b,
            "history": d - c,
            "total": d - a,
        }.items():
            rows[key].append(value.cpu().numpy())
        for i, p in enumerate((p0, p1)):
            histories[i] = advance(
                histories[i], actual[:, hour] / inputs.scales, p, calendar, temperature
            )
        torch.testing.assert_close(
            histories[0][:, :, 0], histories[1][:, :, 0], rtol=0, atol=0
        )
    result = {key: np.stack(value, -1) for key, value in rows.items()}
    np.testing.assert_allclose(
        result["absolute"] + result["change"] + result["history"],
        result["total"],
        atol=3e-6,
    )
    return result


def main(output, responses):
    data = ConsumerData(ROOT / "cleaned", ROOT / "prepared")
    selection = json.loads((responses / "selection.json").read_text())
    origins = selected_origins(np.r_[data.origins["validation"], data.origins["test"]])
    normal_raw = float(
        np.median(
            data.prices[
                data.area_index[data.primary, None],
                np.flatnonzero(data.partitions == 0),
            ]
        )
    )
    normal = float(data.features.base.price(normal_raw))
    high = float(data.features.base.price(2.0))
    require(not output.exists(), "Choose a new channel-diagnostic output.")
    output.mkdir(parents=True)
    torch.set_num_threads(2)
    results = {}
    for name, paths in {
        "ensemble": selection["price_checkpoints"][:2],
        "seed2026": [selection["price_checkpoints"][2]],
    }.items():
        models = []
        for path in paths:
            require(
                digest(path) == selection["checkpoint_sha256"][path], "Weights changed."
            )
            model, saved = load_model(path, "cpu")
            require(saved["preprocessing"] == data.state, "Feature mismatch.")
            models.append(model)
        values = {}
        for number, origin in enumerate(origins):
            day = {}
            for start in range(0, len(data.ids), 256):
                hh = data.all_households[start : start + 256]
                inputs = data.inputs(hh, np.full(len(hh), origin), 72)
                actual = data.targets(hh, np.full(len(hh), origin), 72)
                part = decompose(models, inputs, actual, normal, high)
                for key, value in part.items():
                    day.setdefault(key, []).append(value)
            for key, value in day.items():
                values.setdefault(key, []).append(np.concatenate(value))
            print(f"{name}: price-channel origin {number+1}/{len(origins)}", flush=True)
        values = {key: np.stack(value) for key, value in values.items()}
        np.savez_compressed(
            output / f"{name}.npz", origins=origins, household_ids=data.ids, **values
        )
        results[name] = {}
        for day in range(3):
            sl = slice(day * 24, (day + 1) * 24)
            baseline = values["normal"][:, data.primary, sl].sum()
            results[name][str(day + 1)] = {
                key: float(values[key][:, data.primary, sl].sum() / baseline)
                for key in ("absolute", "change", "history", "total")
            }
        results[name]["immediate"] = {
            key: float(
                values[key][:, data.primary, 0].sum()
                / values["normal"][:, data.primary, 0].sum()
            )
            for key in ("absolute", "change", "history", "total")
        }
    # Independently check the complete constant-price recursive paths via raw CPU API.
    hh = np.r_[
        data.primary[data.area_index[data.primary] == 0][::81][:6],
        data.primary[data.area_index[data.primary] == 1][::38][:6],
    ]
    origin = origins[0]
    past = np.arange(origin - 24, origin)
    sim = ConsumerSimulator(selection["price_checkpoints"][:2], "cpu")
    raw_checks = {}
    for label, price in (
        ("constant_low", 0.05),
        ("constant_normal", normal_raw),
        ("constant_high", 2.0),
    ):
        q = sim.forecast(
            data.demand[hh[:, None], past],
            data.prices[data.area_index[hh, None], past],
            data.temperature[data.area_index[hh, None], past],
            data.times[past],
            [data.ids[h] for h in hh],
            np.full((len(hh), 72), price),
        )
        path = output.parent / "price_aware" / f"{label}.npz"
        expected = np.load(path)["prediction"][0, hh]
        np.testing.assert_allclose(q, expected, rtol=3e-3, atol=3e-4)
        raw_checks[label] = float(np.max(np.abs(q - expected)))
    save_json(
        output / "summary.json",
        {
            "normal_price": normal_raw,
            "high_price": 2.0,
            "origins": origins.tolist(),
            "contributions": results,
            "raw_cpu_constant_path_max_difference": raw_checks,
            "scope": (
                "Model-channel attribution with fixed measured loads; "
                "not a physical response experiment."
            ),
            "order": (
                "Absolute candidate price, candidate price change, "
                "encoded historical prices."
            ),
            "source_sha256": digest(__file__),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "runs/sliding_72h_investigation_v1/price_channels",
    )
    parser.add_argument(
        "--responses", type=Path, default=ROOT / "runs/response_refinement_v2"
    )
    args = parser.parse_args()
    main(args.output, args.responses)

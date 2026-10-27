"""Frozen old/upgraded predictors behind one raw-consumption interface."""

import json
from pathlib import Path

import numpy as np
import torch

from norway_energy_crisis.clean_data import ROOT, digest, require
from norway_energy_crisis.dqn.environment import fingerprint
from norway_energy_crisis.train_response import load_model as load_old
from norway_energy_crisis.response_v2.model import load_model as load_new


class ResponseEngine:
    """Predict kWh from the same shared history; never maintain hidden histories.

    - Old checkpoints consume nine saved base features.
    - Upgraded checkpoints additionally receive log demand and price changes.
    - Ensemble members see identical inputs; average before environment feedback.
    """

    def __init__(
        self,
        data,
        family,
        device="mps",
        old_root=None,
        new_root=None,
        selection_file=None,
    ):
        require(family in ("old", "new", "reserved", "control"), "Unknown engine.")
        require(device in ("cpu", "mps"), "Choose cpu or mps explicitly.")
        if device == "mps":
            require(torch.backends.mps.is_available(), "MPS unavailable; no fallback.")
        if family == "reserved":
            require(
                selection_file is not None and Path(selection_file).exists(),
                "Freeze policy selection first.",
            )
            require(
                json.loads(Path(selection_file).read_text()).get("frozen_before_test")
                is True,
                "Selection not frozen.",
            )
        old_root = Path(old_root or ROOT / "runs/comparison_v1")
        new_root = Path(new_root or ROOT / "runs/response_refinement_v2")
        self.family, self.device = family, device
        self.preprocessor = data.preprocessor
        self.scales = self.preprocessor.scales
        self.is_price_free = family == "control"
        self.models, self.paths = [], []
        if family == "old":
            paths = [
                old_root / f"transformer_24h_seed{s}_price/best.pt" for s in (42, 7)
            ]
        else:
            selected = json.loads((new_root / "selection.json").read_text())
            key = "control_checkpoints" if family == "control" else "price_checkpoints"
            paths = (
                [Path(p) for p in selected[key]][2:3]
                if family == "reserved"
                else [Path(p) for p in selected[key]][:2]
            )
            for path in paths:
                require(
                    digest(path) == selected["checkpoint_sha256"][str(path)],
                    "Response weights changed.",
                )
        for path in paths:
            model, saved = load_old(path) if family == "old" else load_new(path, device)
            state = saved["preprocessing"]
            if family == "old":
                require(state == data.state, "Old preprocessing differs.")
                require(
                    saved["prepared_manifest_sha256"]
                    == digest(data.folder / "manifest.json"),
                    "Old data differs.",
                )
            else:
                positions = [
                    state["household_ids"].index(i) for i in self.preprocessor.ids
                ]
                require(
                    np.array_equal(
                        np.asarray(state["demand_scales"])[positions], self.scales
                    ),
                    "Demand scales differ.",
                )
                require(
                    np.array_equal(
                        np.asarray(state["static_encoded"])[positions],
                        self.preprocessor.static,
                    ),
                    "Static descriptors differ.",
                )
                for name in (
                    "price_mean",
                    "price_std",
                    "temperature_mean",
                    "temperature_std",
                ):
                    require(
                        state[name] == data.state[name],
                        f"Feature conversion differs: {name}",
                    )
                require(
                    saved["data_fingerprint"]["prepared"]
                    == digest(data.folder / "manifest.json"),
                    "Upgraded data differs.",
                )
            model.to(device).eval().requires_grad_(False)
            self.models.append(model)
            self.paths.append(str(path.resolve()))
        self.static = torch.as_tensor(
            self.preprocessor.static, dtype=torch.float32, device=device
        )
        self.signature = fingerprint(
            dict(
                family=family,
                weights=[digest(p) for p in paths],
                data=digest(data.folder / "manifest.json"),
                source=digest(__file__),
                recursion="shared_mean_base9_v1",
            )
        )

    @torch.inference_mode()
    def predict(self, history, raw_prices, calendar):
        """Return household or batch×household kWh, with float32 accelerator inputs."""
        history = np.asarray(history, dtype=np.float32)
        single = history.ndim == 3
        if single:
            history = history[None]
        batch, households = history.shape[:2]
        require(
            households == len(self.scales) and history.shape[2:] == (24, 9),
            "History shape mismatch.",
        )
        price = self.preprocessor.price(
            np.broadcast_to(raw_prices, (batch, households))
        ).reshape(-1)
        calendar = np.asarray(calendar, dtype=np.float32)
        if calendar.shape == (6,):
            calendar = np.broadcast_to(calendar, (batch, 6))
        calendar = np.broadcast_to(calendar[:, None], (batch, households, 6)).reshape(
            -1, 6
        )
        features = history.reshape(-1, 24, 9)
        if self.family != "old":
            changes = np.zeros(features.shape[:2], dtype=np.float32)
            changes[:, 1:] = np.diff(features[:, :, 1], axis=1)
            available = np.ones_like(changes)
            available[:, 0] = 0
            features = np.concatenate(
                (
                    features,
                    np.log1p(features[:, :, :1]),
                    changes[:, :, None],
                    available[:, :, None],
                ),
                axis=-1,
            )
        static = self.static.repeat(batch, 1)
        values = []
        chunk = 1024 if self.device == "mps" else 256
        for start in range(0, len(features), chunk):
            stop = start + chunk
            x = torch.as_tensor(features[start:stop], device=self.device)
            p = torch.as_tensor(
                price[start:stop], dtype=torch.float32, device=self.device
            )
            c = torch.as_tensor(
                calendar[start:stop].copy(), dtype=torch.float32, device=self.device
            )
            prediction = torch.stack(
                [m(x, static[start:stop], p, c) for m in self.models]
            ).mean(0)
            values.append(prediction.cpu().numpy())
        demand = np.concatenate(values).reshape(batch, households) * self.scales
        require(
            np.isfinite(demand).all() and (demand >= 0).all(),
            "Invalid consumer output.",
        )
        return demand[0] if single else demand

    def candidates(self, history, prices, calendar):
        return self.predict(
            np.broadcast_to(history, (len(prices), *history.shape)),
            np.asarray(prices)[:, None],
            calendar,
        )

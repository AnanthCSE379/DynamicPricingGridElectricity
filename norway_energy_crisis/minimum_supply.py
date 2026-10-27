"""Training-supported supply reservations, not identified consumer requirements.

A baseline protects requested consumption before flexible allocation. It never
raises demand or rewrites the response model's history to manufacture accuracy.
"""

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from norway_energy_crisis.clean_data import digest, require, save_json
from norway_energy_crisis.dqn.environment import fingerprint
from norway_energy_crisis.capacity.allocation import AllocationResult, allocate


@dataclass(frozen=True)
class MinimumSupply:
    household_ids: tuple
    floor_kw: tuple[float, ...]
    prepared_manifest_sha256: str
    rule: str
    quantile: float | None = None

    def __post_init__(self):
        values = np.asarray(self.floor_kw, dtype=float)
        require(
            values.shape == (len(self.household_ids),)
            and len(set(self.household_ids)) == len(self.household_ids)
            and np.isfinite(values).all()
            and (values >= 0).all(),
            "Invalid household minimum-supply baseline.",
        )
        require(
            self.quantile is None or 0 <= self.quantile <= 1,
            "Invalid training-tail quantile.",
        )

    @property
    def signature(self):
        return fingerprint(asdict(self))

    def validate_provider(self, provider):
        require(
            tuple(provider.preprocessor.ids) == self.household_ids,
            "Minimum-supply household order mismatch.",
        )
        require(
            provider.manifest_hash == self.prepared_manifest_sha256,
            "Minimum-supply data fingerprint mismatch.",
        )

    def save(self, path):
        save_json(
            path,
            dict(
                format="minimum_supply_v1",
                **asdict(self),
                signature=self.signature,
                interpretation=(
                    "Declared supply reservation, not a validated behavioral "
                    "or essential-load threshold."
                ),
            ),
        )

    @classmethod
    def load(cls, path):
        state = json.loads(Path(path).read_text())
        require(state["format"] == "minimum_supply_v1", "Unknown baseline format.")
        result = cls(
            tuple(state["household_ids"]),
            tuple(state["floor_kw"]),
            state["prepared_manifest_sha256"],
            state["rule"],
            state["quantile"],
        )
        require(result.signature == state["signature"], "Baseline file changed.")
        return result

    @classmethod
    def fit(cls, data, quantile=0.01):
        """Reserve each household's positive-load training 1st percentile.

        One percent is a declared marginal-support rule, not an accuracy optimum.
        Positive observations avoid adopting zero as the supply reservation;
        genuine zero requests remain zero. Entirely inactive households reserve
        zero. Validation and follow-up observations never fit this baseline.
        """
        require(0 <= quantile <= 1, "Invalid training-tail quantile.")
        training = np.asarray(data.demand[:, data.partitions == 0])
        require(
            training.shape[1] > 0
            and np.isfinite(training).all()
            and (training >= 0).all(),
            "Invalid training consumption.",
        )
        floors = []
        for values in training:
            positive = values[values > 0]
            floors.append(
                float(np.quantile(positive, quantile, method="linear"))
                if len(positive)
                else 0.0
            )
        return cls(
            tuple(data.preprocessor.ids),
            tuple(floors),
            digest(data.folder / "manifest.json"),
            "positive_training_quantile_linear",
            float(quantile),
        )

    @classmethod
    def fixed(cls, data, floor_kw=0.1):
        return cls(
            tuple(data.preprocessor.ids),
            (float(floor_kw),) * data.households,
            digest(data.folder / "manifest.json"),
            "fixed_available_supply",
        )

    def allocate(
        self,
        requested,
        essential,
        budget_kwh,
        requested_before=None,
        served_before=None,
        method="fair",
        order=None,
        *,
        household_indices=None,
        interval_hours=1.0
    ):
        """Protect min(request, reservation) while preserving essential accounting.

        Essential requirements and baseline reservations must jointly fit. Fail
        explicitly if they do not: an infeasible capacity cannot guarantee supply.
        Flexible priority includes baseline energy already served this hour.
        """
        requested, essential = [
            np.asarray(values, dtype=np.float64) for values in (requested, essential)
        ]
        require(
            requested.ndim == 1
            and essential.shape == requested.shape
            and np.isfinite(requested).all()
            and np.isfinite(essential).all()
            and (requested >= essential).all()
            and (essential >= 0).all(),
            "Invalid requests or essential requirements.",
        )
        require(
            np.isfinite(interval_hours) and interval_hours > 0,
            "Invalid allocation interval.",
        )
        indices = (
            np.arange(len(self.floor_kw))
            if household_indices is None
            else np.asarray(household_indices, dtype=int)
        )
        require(
            indices.shape == requested.shape
            and (indices >= 0).all()
            and (indices < len(self.floor_kw)).all(),
            "Invalid baseline indices.",
        )
        reserve = np.minimum(
            requested, np.asarray(self.floor_kw)[indices] * interval_hours
        )
        protected = np.maximum(essential, reserve)
        require(
            np.isfinite(budget_kwh) and budget_kwh >= 0,
            "Invalid regional energy budget.",
        )
        require(
            protected.sum() <= budget_kwh,
            "Capacity cannot meet essential demand and the demand-capped "
            "minimum supply.",
        )
        prior_requests = (
            np.zeros_like(requested)
            if requested_before is None
            else np.asarray(requested_before, dtype=float)
        )
        prior_served = (
            np.zeros_like(requested)
            if served_before is None
            else np.asarray(served_before, dtype=float)
        )
        require(
            prior_requests.shape == requested.shape
            and prior_served.shape == requested.shape
            and np.isfinite(prior_requests).all()
            and np.isfinite(prior_served).all()
            and (prior_requests >= 0).all()
            and (prior_served >= 0).all()
            and (prior_served <= prior_requests + 1e-8).all(),
            "Invalid service history.",
        )
        reserved_flexible = protected - essential
        result = allocate(
            requested,
            protected,
            budget_kwh,
            prior_requests + reserved_flexible,
            prior_served + reserved_flexible,
            method,
            order,
        )
        return AllocationResult(
            delivered=result.delivered,
            essential_delivered=essential.copy(),
            flexible_delivered=result.delivered - essential,
            essential_unserved=np.zeros_like(essential),
            flexible_unserved=np.maximum(requested - result.delivered, 0),
            remaining_kwh=result.remaining_kwh,
            essential_infeasible=False,
        )

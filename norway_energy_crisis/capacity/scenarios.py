"""Training-calibrated stress assumptions and experiment provenance."""

from dataclasses import asdict, dataclass
import importlib.metadata
from pathlib import Path
import platform

import numpy as np

from norway_energy_crisis.clean_data import ROOT, digest, require
from norway_energy_crisis.dqn.environment import fingerprint

LEVELS = (("mild", 0.95), ("moderate", 0.90), ("severe", 0.75))
SEEDS = (42, 7, 2026)
CONFIGURATIONS = ("dqn", "ppo_mlp", "ppo_transformer")
RUN_ORDER = tuple((name, seed) for seed in SEEDS for name in CONFIGURATIONS)
OBJECTIVE = "terminal_reference_household_service_loss_v1"


@dataclass(frozen=True)
class CapacityScenario:
    """Constructed hourly average-power limits, never measured feeder ratings."""

    name: str
    quantile: float
    capacity_kw: tuple[float, float]
    alpha: float = 0.5
    interval_hours: float = 1.0
    calibration_provenance: str = (
        "Declared stress assumption; not a measured infrastructure rating."
    )

    def __post_init__(self):
        require(len(self.capacity_kw) == 2, "Two independent regional pools required.")
        require(
            np.isfinite(self.capacity_kw).all() and min(self.capacity_kw) >= 0,
            "Capacity must be finite and nonnegative.",
        )
        require(
            0 <= self.alpha <= 1 and self.interval_hours > 0,
            "Invalid essential fraction or interval length.",
        )

    @property
    def budget_kwh(self):
        return np.asarray(self.capacity_kw, dtype=np.float64) * self.interval_hours

    @property
    def signature(self):
        return fingerprint(asdict(self))


def calibrate(data, alpha=0.5):
    """Fit regional limits exclusively from measured training consumption."""
    training = data.partitions == 0
    regional = [
        np.asarray(
            data.demand[data.area_index == area][:, training], dtype=np.float64
        ).sum(0)
        for area in range(2)
    ]
    require(
        all(len(values) and np.isfinite(values).all() for values in regional),
        "Invalid regional training demand.",
    )
    provenance = (
        f"prepared_manifest={digest(data.folder / 'manifest.json')}; "
        "measured training regional hourly aggregates; NumPy linear quantile"
    )
    return tuple(
        CapacityScenario(
            name,
            quantile,
            tuple(
                float(np.quantile(values, quantile, method="linear"))
                for values in regional
            ),
            alpha,
            calibration_provenance=provenance,
        )
        for name, quantile in LEVELS
    )


def sources():
    """Include shared inference code in cache/resume identities, not just new files."""
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [
        ROOT / name
        for name in (
            "clean_data.py",
            "preprocess.py",
            "train_response.py",
            "minimum_supply.py",
        )
    ]
    paths += list((ROOT / "dqn").glob("*.py"))
    paths += [ROOT / "inspection" / name for name in ("recorder.py", "store.py")]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def dependencies():
    names = (
        "numpy",
        "pandas",
        "torch",
        "gymnasium",
        "stable-baselines3",
        "sb3-contrib",
    )
    return dict(
        python=platform.python_version(),
        platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in names},
    )


def evidence(data, scenarios):
    return dict(
        prepared_manifest_sha256=digest(data.folder / "manifest.json"),
        cleaned_manifest_sha256=data.manifest["cleaned_manifest_sha256"],
        household_ids=data.preprocessor.ids,
        areas=data.state["areas"],
        scenarios=[asdict(scenario) for scenario in scenarios],
        capacity_calibration=(
            "Regional measured training hourly demand; linear quantiles."
        ),
        sources=sources(),
        dependencies=dependencies(),
        objective=OBJECTIVE,
        evidence={
            "measured": "Historical household consumption and regional spot prices.",
            "survey": (
                "Later reported contracts/descriptors, assumed stable retrospectively."
            ),
            "reanalysis": "Regional ERA5-Land temperature; persisted within forecasts.",
            "constructed": (
                "Regional capacity pools, essential fractions and FCFS arrival order."
            ),
            "modelled": (
                "Counterfactual price response, delivered service and energy charges."
            ),
        },
        limitations=[
            "Recorded day-ahead regional prices are assumed known at midnight.",
            (
                "Response-model development and earlier experiments already examined"
                " the follow-up test months."
            ),
            (
                "The pools are not measured feeders; no inter-region energy transfer is"
                " allowed."
            ),
            (
                "Limits apply to hourly average power, not voltage, line flow or"
                " instantaneous peaks."
            ),
            (
                "Essential floors are scenario assumptions, not measured appliance"
                " requirements."
            ),
            (
                "Curtailment is permanent; no queues, recovery, fatigue or rebound are"
                " simulated."
            ),
            "Future requests do not respond to allocation or curtailment.",
            (
                "Alternative-price response is observational-model prediction, not"
                " causal validation."
            ),
            (
                "February-March and response seed 2026 were previously examined;"
                " results are retrospective."
            ),
            (
                "Energy charges omit complete retail bills, supplier costs and consumer"
                " welfare measurements."
            ),
            (
                "All bounded prices are valid; this study cannot attribute a benefit to"
                " action masking."
            ),
        ],
    )

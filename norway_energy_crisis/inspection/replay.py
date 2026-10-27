"""Export explicitly labelled deterministic checkpoint replays for the viewer."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import DQN

from norway_energy_crisis.clean_data import ROOT, digest
from norway_energy_crisis.preprocess import PreparedData
from norway_energy_crisis.dqn.environment import EpisodeProvider, PriceEnv, fingerprint
from norway_energy_crisis.dqn.train import make_engine, checkpoint_paths
from .recorder import EpisodeRecorder


def replay(
    run,
    output=None,
    data_folder=ROOT / "prepared",
    response_root=ROOT / "runs/comparison_v1",
    days=3,
    device="cpu",
):
    run = Path(run).resolve()
    checkpoint = run / "best.zip"
    best = json.loads((run / "best.json").read_text())
    signature = best["signature"]
    capacity = "configuration" in signature
    output = (
        Path(output)
        if output
        else ROOT / "runs/inspection_replays" / f"{run.parent.name}_{run.name}"
    )
    if output.exists():
        raise ValueError(
            "Choose a new replay output; existing recordings are preserved."
        )
    if not 1 <= days <= 30:
        raise ValueError("Choose 1–30 validation days.")
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable; choose cpu explicitly or enable GPU access."
        )
    torch.set_num_threads(4)
    data = PreparedData(data_folder)
    provider = EpisodeProvider(data, "validation")
    engine = make_engine(response_root, "ensemble", data, device)
    expected_engine = None if capacity else signature.get("engine")
    if capacity:
        evidence = json.loads((run.parent / "evidence.json").read_text())
        expected_weights = {
            entry["sha256"] for entry in evidence["response_weights"]["ensemble"]
        }
        for path in checkpoint_paths(response_root, "ensemble"):
            if digest(path) not in expected_weights:
                raise ValueError(
                    "Response weights differ from the original experiment."
                )
        if (
            digest(data.folder / "manifest.json")
            != evidence["prepared_manifest_sha256"]
        ):
            raise ValueError("Prepared data differs from the original experiment.")
    if expected_engine is not None:
        # Older runs predate source formatting changes. Verify weights/data with
        # the recorded source identity, while disclosing the current code identity.
        original_identity = fingerprint(
            dict(
                models=[digest(p) for p in checkpoint_paths(response_root, "ensemble")],
                data=digest(data.folder / "manifest.json"),
                recursion="shared_mean_v1",
                temperature="persist_origin",
                source=signature["sources"]["environment.py"],
            )
        )
        if original_identity != expected_engine:
            raise ValueError("Checkpoint response weights or prepared data differ.")
    if capacity:
        from norway_energy_crisis.capacity.environment import CapacityPriceEnv
        from norway_energy_crisis.capacity.policies import algorithm
        from norway_energy_crisis.capacity.scenarios import calibrate

        base = CapacityPriceEnv(
            provider, engine, calibrate(data), output / "reference_cache"
        )
        model = algorithm(signature["configuration"]).load(checkpoint, device=device)
    else:
        base = PriceEnv(provider, engine, output / "reference_cache", signature["cap"])
        model = DQN.load(checkpoint, device=device)
    env = EpisodeRecorder(
        base,
        output,
        {
            "source_run": str(run),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": digest(checkpoint),
            "response_engine": engine.signature,
            "original_response_engine": expected_engine,
            "current_environment_sha256": digest(ROOT / "dqn/environment.py"),
            "split": "validation",
            "deterministic": True,
            "note": "New checkpoint replay, not historical exploratory training.",
        },
        kind="checkpoint_replay",
    )
    # Spread selected days across validation; never read test-period measurements.
    indices = np.linspace(
        0, len(provider.episodes) - 1, min(days, len(provider.episodes)), dtype=int
    )
    try:
        for index in indices:
            for scenario in range(len(base.scenarios) if capacity else 1):
                observation, _ = env.reset(
                    options={"episode_index": int(index), "scenario_index": scenario}
                )
                done = False
                while not done:
                    kwargs = (
                        {"action_masks": env.action_masks()}
                        if capacity and signature["configuration"] != "dqn"
                        else {}
                    )
                    action, _ = model.predict(observation, deterministic=True, **kwargs)
                    observation, _, terminated, truncated, _ = env.step(int(action))
                    done = terminated or truncated
                print(
                    f"Replayed {provider.episodes[index].timestamp}, "
                    f"scenario {scenario}",
                    flush=True,
                )
    finally:
        env.close()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "prepared")
    parser.add_argument(
        "--response-root", type=Path, default=ROOT / "runs/comparison_v1"
    )
    parser.add_argument("--days", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    args = parser.parse_args()
    print(
        replay(
            args.run, args.output, args.data, args.response_root, args.days, args.device
        )
    )


if __name__ == "__main__":
    main()

"""Training-only scaling and strictly separated ten-step input windows."""

import numpy as np
import torch


HISTORY_STEPS = 10  # Five hours of half-hour demand readings.
PRICES = np.array([3.99, 11.76, 67.20], dtype=np.float32)


def window_targets(partitions, demand):
    """Every historical and target timestamp must remain in one partition."""
    targets = {}
    has_readings = np.isfinite(demand).any(axis=0)
    for partition_id, name in enumerate(("train", "validation", "test")):
        selected = []
        for target in range(HISTORY_STEPS, len(partitions)):
            interval = slice(target - HISTORY_STEPS, target + 1)
            if np.all(partitions[interval] == partition_id) and has_readings[interval].all():
                selected.append(target)
        targets[name] = np.array(selected, dtype=np.int64)
    return targets


def fit_scales(demand, training_timestamps):
    """Fit household means/maxima using raw training observations only."""
    measured = demand[:, training_timestamps]
    known = np.isfinite(measured)
    counts = known.sum(axis=1)
    if (counts == 0).any():
        raise ValueError("A household has no training observations; its output cannot be trained.")
    means = np.nansum(measured, axis=1, dtype=np.float64) / counts
    maxima = np.nanmax(measured, axis=1)
    # A household with all-zero training consumption still needs a nonzero divisor.
    maxima = np.where(maxima > 0, maxima, 1.)
    return means.astype(np.float32), maxima.astype(np.float32)


def make_features(demand, means, maxima):
    """Use training means for missing inputs and retain explicit missingness flags."""
    known = np.isfinite(demand)
    filled = np.where(known, demand, means[:, None])
    scaled = filled / maxima[:, None]
    # Shape is time × (household demands + observed flags), not household samples.
    return torch.from_numpy(np.concatenate((scaled.T, known.T.astype(np.float32)), axis=1).copy())


def batch_inputs(features, prices, targets):
    offsets = torch.arange(-HISTORY_STEPS, 0)  # Oldest history first, ending at t-1.
    history = features[targets[:, None] + offsets[None, :]]
    return history, prices[targets]

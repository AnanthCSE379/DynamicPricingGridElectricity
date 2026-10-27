"""Atomic output helpers; all experiment artifacts stay in their own directory."""

from pathlib import Path

from norway_energy_crisis.clean_data import save_json as replace_json


def save_json(path, value):
    """Create the output parent, then atomically replace finite JSON data."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    replace_json(path, value)

"""The synthetic acquisition of ``data/synthetic``, loaded for the end-to-end tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SYNTHETIC = Path(__file__).resolve().parents[1] / "data" / "synthetic"


FEATURES = ("area", "intensity", "eccentricity")


FIELD_KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The committed tables, as a user reads them, and the truth."""
    objects = pd.read_csv(SYNTHETIC / "synthetic_cells.csv")
    fovs = pd.read_csv(SYNTHETIC / "synthetic_fovs.csv")
    return objects, fovs, json.loads((SYNTHETIC / "expected.json").read_text())

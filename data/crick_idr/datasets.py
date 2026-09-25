"""The registered Crick/IDR datasets and their roles (development or held out).

One place that says which data the analyses were built on and which they
were not. ``registry()`` declares:

- ``idr0028_screenB``: the committed 12 wells x 28 fields on 3 plates,
  **development**: every threshold, preset and definition of new in
  ``docs/preliminary_findings.md`` was chosen while looking at it;
- ``idr0139``: 10 wells x 4 fields on one plate (the JCS case study),
  **held out**: it has never been used to choose a setting. It is shallow (a
  reservoir of 2 fields judges 1 or 2 stream fields), so it can refute a
  development conclusion but cannot establish a protocol.

``BenchmarkRegistry.check_tuning`` raises when a run declares it was tuned on
the held-out dataset. The loaders return MorphoBricks-style linked tables
with each paper's readout added (``YAPTAZ_Ratio``, ``Fascin_Ratio``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from idr0028_data import RATIO as YAPTAZ_RATIO
from idr0028_data import load as load_idr0028

from morphocoverage.benchmark import BenchmarkDataset, BenchmarkRegistry

HERE = Path(__file__).resolve().parent
FASCIN_RATIO = "Fascin_Ratio"
GROUPS = ("condition", "repeat", "well")


def load_idr0139() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The idr0139 observation and FOV tables with ``Fascin_Ratio``, and the features."""
    objects = pd.read_csv(HERE / "idr0139_nuclei.csv.gz", dtype={"repeat": str})
    fovs = pd.read_csv(HERE / "idr0139_fovs.csv.gz", dtype={"repeat": str})
    features = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    nuclear = objects["Nuclei_Intensity_MeanIntensity_Fascin"]
    cytoplasm = objects["Cytoplasm_Intensity_MeanIntensity_Fascin"]
    return objects.assign(**{FASCIN_RATIO: nuclear / (nuclear + cytoplasm)}), fovs, features


def registry() -> BenchmarkRegistry:
    """The two datasets with their declared roles."""
    return BenchmarkRegistry(
        [
            BenchmarkDataset(
                name="idr0028_screenB",
                role="development",
                load=lambda: load_idr0028()[:2],
                feature_cols=(YAPTAZ_RATIO,),
                acquisition_order_col="acquisition_order",
                group_cols=GROUPS,
                observation_unit="nucleus",
                description=(
                    "12 wells x 28 fields on 3 plates; every preset and definition of new "
                    "was chosen on it (docs/preliminary_findings.md)"
                ),
            ),
            BenchmarkDataset(
                name="idr0139",
                role="held_out",
                load=lambda: load_idr0139()[:2],
                feature_cols=(FASCIN_RATIO,),
                acquisition_order_col="acquisition_order",
                group_cols=GROUPS,
                observation_unit="nucleus",
                description=(
                    "10 wells x 4 fields on one plate (JCS case study); never used to choose "
                    "a setting; too shallow to set a protocol"
                ),
            ),
        ]
    )


__all__ = ["FASCIN_RATIO", "GROUPS", "YAPTAZ_RATIO", "load_idr0028", "load_idr0139", "registry"]

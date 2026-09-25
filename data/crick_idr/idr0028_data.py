"""The committed idr0028 screen B subset, as the analysis scripts next to it read it.

One place for what every idr0028 script shares: the table paths, the YAP/TAZ
nuclear ratio of Marcotti et al. (nuclear mean / (nuclear + cytoplasmic mean
intensity)), the 30 CellProfiler intensity features, and the helpers the
semi-synthetic controls use to rebuild FOVs from blocks of cells. It is a
helper for these scripts, not part of the package.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from morphocoverage import validate_input_tables
from morphocoverage.schema import ValidatedInput

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RATIO = "YAPTAZ_Ratio"


def load() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """The observation and FOV tables with ``RATIO`` added, and the 30 intensity features.

    ``repeat`` (the plate) is read as text. The features are the columns named
    ``Nuclei_*`` or ``Cytoplasm_*``; the ratio is derived from two of them and
    is not one of them.
    """
    objects = pd.read_csv(HERE / "idr0028_screenB_nuclei.csv.gz", dtype={"repeat": str})
    fovs = pd.read_csv(HERE / "idr0028_screenB_fovs.csv.gz", dtype={"repeat": str})
    features = [c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_"))]
    nuclear = objects["Nuclei_Intensity_MeanIntensity_YAPTAZ"]
    cytoplasm = objects["Cytoplasm_Intensity_MeanIntensity_YAPTAZ"]
    return objects.assign(**{RATIO: nuclear / (nuclear + cytoplasm)}), fovs, features


def fov_cells(
    objects: pd.DataFrame, fovs: pd.DataFrame, plate: str, well: str, features: list[str]
) -> list[np.ndarray]:
    """Feature matrix of each FOV of one well, in FOV-table order (empty FOVs included)."""
    names = fovs.loc[(fovs["repeat"] == plate) & (fovs["well"] == well), "image_file"]
    by_file = dict(tuple(objects.groupby("image_file")))
    empty = np.empty((0, len(features)))
    return [by_file[n][features].to_numpy(float) if n in by_file else empty for n in names]


def blocks_table(blocks: list[np.ndarray], features: list[str], prefix: str) -> ValidatedInput:
    """Validated input with one FOV per block of cells, named ``prefix0``, ``prefix1``, ..."""
    names = [f"{prefix}{j}" for j in range(len(blocks))]
    kept = [(name, block) for name, block in zip(names, blocks, strict=True) if len(block)]
    # One frame for all blocks: building and concatenating a frame per block
    # dominated the runtime of the controls.
    objects = pd.DataFrame(np.vstack([block for _, block in kept]), columns=features)
    objects["image_file"] = np.repeat([name for name, _ in kept], [len(b) for _, b in kept])
    return validate_input_tables(
        objects, fovs=pd.DataFrame({"image_file": names}), feature_cols=features
    )

"""The Crick/IDR datasets, ready to use: ``load(name)`` returns everything an analysis needs.

Four datasets are stored next to this file (``README.md`` gives provenance,
licence and columns). Each is a pair of MorphoBricks-style linked tables, one
row per acquired field (``*_fovs.csv.gz``, fields without nuclei included) and
one row per nucleus (``*_nuclei.csv.gz``), with the papers' readout stored as
a column:

| name | wells | fields/well | features | readout |
|---|---|---|---|---|
| ``idr0028_screenB`` | the paper's 12, 3 plates | 28 | 30 intensity | ``YAPTAZ_Ratio`` |
| ``idr0139`` | the paper's 10, 1 plate | 4 | 60 intensity | ``Fascin_Ratio`` |
| ``idr0028_screenB_plates`` | all 1,144, 3 plates | 28 | readout only | ``YAPTAZ_Ratio`` |
| ``idr0139_plate`` | all 308, 1 plate | 4 | readout only | ``Fascin_Ratio`` |

``load(name)`` returns a ``Dataset``: the two tables (``condition``,
``biological_repeat`` and ``technical_repeat``, the well, on both), the explicit feature
columns, the readout,
the negative-control condition, the grouping the analyses use (one group per
well), the declared observation unit and the dataset's role. The readout is
derived from two features (nuclear mean / (nuclear mean + cytoplasmic mean)
intensity) and is not one of them.

Roles (``registry()`` declares the first two for the benchmark;
``BenchmarkRegistry.check_tuning`` raises when a run declares it was tuned on
held-out data):

- ``idr0028_screenB`` is **development**: every threshold, preset and
  definition of new in ``docs/preliminary_findings.md`` was chosen while
  looking at it. ``idr0028_screenB_plates`` contains its wells.
- ``idr0139`` is **held out**, its status per choice in AGENTS.md section 9. It
  is shallow (a reservoir of 2 fields judges 1 or 2 stream fields), so it can
  refute a development conclusion but cannot establish a protocol.
  ``idr0139_plate`` contains its wells and is used only for descriptive,
  retrospective checks.

The helpers at the end rebuild FOVs from blocks of nuclei for the
semi-synthetic controls. This is data-preparation code, not part of the
package.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from morphocoverage import validate_input_tables
from morphocoverage.benchmark import BenchmarkDataset, BenchmarkRegistry
from morphocoverage.schema import ValidatedInput

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
YAPTAZ_RATIO = "YAPTAZ_Ratio"
FASCIN_RATIO = "Fascin_Ratio"
# The MorphoBricks group columns: the well is the technical repeat within a plate.
GROUPS = ("condition", "biological_repeat", "technical_repeat")
OBSERVATION_UNIT = "nucleus"


@dataclass(frozen=True)
class Dataset:
    """One stored dataset with what its analyses need to know about it."""

    name: str
    objects: pd.DataFrame
    fovs: pd.DataFrame
    features: tuple[str, ...]
    readout: str
    control: str
    groups: tuple[str, ...]
    observation_unit: str
    role: str
    description: str

    @property
    def tables(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``(objects, fovs)``, as every package analysis takes them."""
        return self.objects, self.fovs


_SPECS = {
    "idr0028_screenB": (
        YAPTAZ_RATIO,
        "MOCK",
        "development",
        "the paper's 12 wells x 28 fields on 3 plates (idr0028 screen B), 30 features",
    ),
    "idr0139": (
        FASCIN_RATIO,
        "Untreated",
        "held_out",
        "the paper's 10 wells x 4 fields on one plate (idr0139, the JCS case study), 60 features",
    ),
    "idr0028_screenB_plates": (
        YAPTAZ_RATIO,
        "MOCK",
        "development",
        "every well of the 3 idr0028 screen B plates, 28 fields each, readout only",
    ),
    "idr0139_plate": (
        FASCIN_RATIO,
        "Untreated",
        "held_out",
        "every well of the idr0139 plate, 4 fields each, readout only; descriptive use only",
    ),
}
NAMES = tuple(_SPECS)


def load(name: str) -> Dataset:
    """The stored dataset ``name`` (one of ``NAMES``), ready for the package's analyses."""
    if name not in _SPECS:
        raise ValueError(f"unknown dataset {name!r}; choose one of {list(NAMES)}")
    readout, control, role, description = _SPECS[name]
    objects = pd.read_csv(HERE / f"{name}_nuclei.csv.gz", dtype={"biological_repeat": str})
    fovs = pd.read_csv(HERE / f"{name}_fovs.csv.gz", dtype={"biological_repeat": str})
    # No experiment description comes with these data: their curators confirmed each folder
    # label as its own biological repeat, so it is declared as the repeat group, by which
    # MorphoCoverage pairs repeats (#617). The stored files are unchanged.
    fovs["repeat_group"] = fovs["biological_repeat"]
    if "condition" not in objects.columns:
        # The complete plates keep the metadata in the FOV table; idr0028 reuses
        # file names across plates, so the join is on (biological_repeat, image_file).
        metadata = fovs[["biological_repeat", "image_file", "condition", "technical_repeat"]]
        objects = objects.merge(
            metadata, on=["biological_repeat", "image_file"], how="left", validate="many_to_one"
        )
        first = ["condition", "biological_repeat", "technical_repeat"]
        objects = objects[first + [c for c in objects.columns if c not in first]]
    features = tuple(c for c in objects.columns if c.startswith(("Nuclei_", "Cytoplasm_")))
    return Dataset(
        name=name,
        objects=objects,
        fovs=fovs,
        features=features,
        readout=readout,
        control=control,
        groups=GROUPS,
        observation_unit=OBSERVATION_UNIT,
        role=role,
        description=description,
    )


def registry() -> BenchmarkRegistry:
    """The two benchmark datasets with their declared roles."""
    return BenchmarkRegistry(
        [
            BenchmarkDataset(
                name=name,
                role=_SPECS[name][2],
                load=lambda name=name: load(name).tables,
                feature_cols=(_SPECS[name][0],),
                acquisition_order_col="acquisition_order",
                group_cols=GROUPS,
                observation_unit=OBSERVATION_UNIT,
                description=_SPECS[name][3],
            )
            for name in ("idr0028_screenB", "idr0139")
        ]
    )


def fov_matrices(
    objects: pd.DataFrame, fovs: pd.DataFrame, plate: str, well: str, features: list[str]
) -> list[np.ndarray]:
    """Feature matrix of each FOV of one well, in FOV-table order (empty FOVs included)."""
    names = fovs.loc[
        (fovs["biological_repeat"] == plate) & (fovs["technical_repeat"] == well), "image_file"
    ]
    in_plate = objects[objects["biological_repeat"] == plate]
    by_file = dict(tuple(in_plate.groupby("image_file")))
    empty = np.empty((0, len(features)))
    return [by_file[n][features].to_numpy(float) if n in by_file else empty for n in names]


def blocks_table(blocks: list[np.ndarray], features: list[str], prefix: str) -> ValidatedInput:
    """Validated input with one FOV per block of nuclei, named ``prefix0``, ``prefix1``, ..."""
    names = [f"{prefix}{j}" for j in range(len(blocks))]
    kept = [(name, block) for name, block in zip(names, blocks, strict=True) if len(block)]
    # One frame for all blocks: building and concatenating a frame per block
    # dominated the runtime of the controls.
    objects = pd.DataFrame(np.vstack([block for _, block in kept]), columns=features)
    objects["image_file"] = np.repeat([name for name, _ in kept], [len(b) for _, b in kept])
    return validate_input_tables(
        objects, fovs=pd.DataFrame({"image_file": names}), feature_cols=features
    )


__all__ = [
    "FASCIN_RATIO",
    "GROUPS",
    "NAMES",
    "OBSERVATION_UNIT",
    "RESULTS",
    "YAPTAZ_RATIO",
    "Dataset",
    "blocks_table",
    "fov_matrices",
    "load",
    "registry",
]

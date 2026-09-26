"""A synthetic acquisition whose answers are known before MorphoCoverage sees it.

The tables follow the MorphoBricks format of ``docs/data_format.md``: a FOV
table with every acquired field, including fields where nothing was segmented,
and an observation table with one row per cell. The design is small enough to
run every workflow in seconds and large enough for each answer to be clear:

- 2 conditions (``control``, ``drug``) x 3 biological repeats (``2026_01_05``,
  ``2026_01_12``, ``2026_01_19``) x 5 wells (``technical_repeat``) x 6 fields.
  That makes 180 fields. The same file names (``field_01.tif`` ...) repeat in
  every well, so a field is only identified within its well, as in a real plate.
- Field 4 of every well is empty (the microscope looked and found no cell):
  30 empty fields. The other fields hold 8 to 24 cells, fixed by the seed.
- ``area`` (the readout) is ``100 + repeat + well + field + cell`` with
  normal effects of variance 9, 16, 25 and 100, and ``+15`` in ``drug``
  (1.5 cell standard deviations).
- ``intensity`` and ``eccentricity`` have field and cell variation but no
  effect of condition: null features.
- ``label_id`` numbers the cells of each field from 1.
- ``rare``: 5 cells of a rare state (large, bright) were added to 3 of the 6
  fields of one drug well (repeat ``2026_01_05``, well ``D02``). The label
  column marks them.
- Stage coordinates on a 3 x 2 grid per well and a raster acquisition order.

``expected.json`` holds the truth and the answers derived from it: exact
counts, the variance components, and the effects with how precisely each is
recovered. ``tests/test_end_to_end.py`` checks the whole pipeline against it,
then breaks the tables in the ways people break them and checks that each
mistake is caught and explained.

Usage, from the repository root::

    python data/synthetic/make_synthetic.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SEED = 20260926
CONDITIONS = ("control", "drug")
REPEATS = ("2026_01_05", "2026_01_12", "2026_01_19")
WELLS = {
    "control": ("B02", "B03", "B04", "B05", "B06"),
    "drug": ("D02", "D03", "D04", "D05", "D06"),
}
FIELDS_PER_WELL = 6
EMPTY_FIELD = 4  # field_04.tif of every well holds no cell
VARIANCE = {"repeat": 9.0, "well": 16.0, "field": 25.0, "cell": 100.0}
AREA_BASELINE = 100.0
DRUG_EFFECT = 15.0
RARE = {"repeat": "2026_01_05", "well": "D02", "fields": (1, 2, 3), "cells": 5}
FEATURES = ("area", "intensity", "eccentricity")


def build(seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The observation table and the FOV table, deterministic for ``seed``."""
    rng = np.random.default_rng(seed)
    sd = {name: np.sqrt(value) for name, value in VARIANCE.items()}
    fov_rows, cell_rows = [], []
    order = 0
    for repeat in REPEATS:
        repeat_effect = rng.normal(0, sd["repeat"])
        for condition in CONDITIONS:
            shift = DRUG_EFFECT if condition == "drug" else 0.0
            for well in WELLS[condition]:
                well_effect = rng.normal(0, sd["well"])
                for field in range(1, FIELDS_PER_WELL + 1):
                    order += 1
                    key = {
                        "condition": condition,
                        "repeat": repeat,
                        "technical_repeat": well,
                        "image_file": f"field_{field:02d}.tif",
                    }
                    fov_rows.append(
                        {
                            **key,
                            "acquisition_order": order,
                            "stage_x": float((field - 1) % 3) * 500.0,
                            "stage_y": float((field - 1) // 3) * 500.0,
                        }
                    )
                    n_cells = 0 if field == EMPTY_FIELD else int(rng.integers(8, 25))
                    field_area = rng.normal(0, sd["field"])
                    field_intensity = rng.normal(0, 2.0)
                    area = (
                        AREA_BASELINE
                        + repeat_effect
                        + well_effect
                        + shift
                        + field_area
                        + rng.normal(0, sd["cell"], n_cells)
                    )
                    intensity = 50.0 + field_intensity + rng.normal(0, 5.0, n_cells)
                    eccentricity = np.clip(rng.normal(0.5, 0.1, n_cells), 0.0, 1.0)
                    rare = (
                        repeat == RARE["repeat"]
                        and well == RARE["well"]
                        and field in RARE["fields"]
                    )
                    labels = [False] * n_cells
                    if rare:
                        area = np.concatenate([area, rng.normal(250.0, 5.0, RARE["cells"])])
                        intensity = np.concatenate(
                            [intensity, rng.normal(90.0, 2.0, RARE["cells"])]
                        )
                        eccentricity = np.concatenate([eccentricity, np.full(RARE["cells"], 0.9)])
                        labels += [True] * RARE["cells"]
                    for label_id, values in enumerate(
                        zip(area, intensity, eccentricity, labels, strict=True), start=1
                    ):
                        cell_rows.append(
                            {
                                **key,
                                "label_id": label_id,
                                "area": round(float(values[0]), 3),
                                "intensity": round(float(values[1]), 3),
                                "eccentricity": round(float(values[2]), 4),
                                "rare": bool(values[3]),
                            }
                        )
    return pd.DataFrame(cell_rows), pd.DataFrame(fov_rows)


def expected(objects: pd.DataFrame, fovs: pd.DataFrame) -> dict:
    """The truth and the answers that follow from it (see the module)."""
    key = ["condition", "repeat", "technical_repeat", "image_file"]
    counts = objects.groupby(key).size().reindex(pd.MultiIndex.from_frame(fovs[key]), fill_value=0)
    n = counts.to_numpy(dtype=float)
    filled = n[n > 0]
    # The cluster size m = sum n_j^2 / N that sets the design effect of whole fields.
    cluster = float(np.sum(filled**2) / np.sum(filled))
    within = VARIANCE["field"] + VARIANCE["cell"]
    # A field of n cells has z = (mean_j - mean) / (s / sqrt(n)) with variance about
    # 1 + n v_field / v_cell, so |z| > 1.96 in a share erfc(1.96 / sqrt(2 (1 + n v_f / v_c))).
    spread = 1 + filled.mean() * VARIANCE["field"] / VARIANCE["cell"]
    unusual = math.erfc(1.96 / math.sqrt(2 * spread))
    return {
        "seed": SEED,
        "design": {
            "conditions": list(CONDITIONS),
            "repeats": list(REPEATS),
            "wells_per_condition_and_repeat": len(WELLS["control"]),
            "fields_per_well": FIELDS_PER_WELL,
        },
        "counts": {
            "fields": len(fovs),
            "empty_fields": int((n == 0).sum()),
            "cells": len(objects),
            "technical_repeats": int(fovs.groupby(["repeat", "technical_repeat"]).ngroups),
            "analysis_groups": int(
                fovs.groupby(["condition", "repeat", "technical_repeat"]).ngroups
            ),
            "rare_cells": int(objects["rare"].sum()),
            "rare_fields": len(RARE["fields"]),
        },
        "truth": {
            "variance": VARIANCE,
            "area_baseline": AREA_BASELINE,
            "drug_effect": DRUG_EFFECT,
            # Within a well: the share of the variance between fields.
            "icc_within_well": VARIANCE["field"] / within,
            "cluster_size": cluster,
            "design_effect_within_well": (cluster * VARIANCE["field"] + VARIANCE["cell"]) / within,
            "unusual_fov_share": unusual,
            # Cohen's d on cells within one repeat: the effect over the SD of a cell's area
            # around its condition mean (well, field and cell variance).
            "cohen_d_cells": DRUG_EFFECT
            / np.sqrt(VARIANCE["well"] + VARIANCE["field"] + VARIANCE["cell"]),
            "null_features": ["intensity", "eccentricity"],
            "rare": RARE | {"fields": list(RARE["fields"])},
        },
        "answers": {
            "check_tables": "no error; one warning: the 30 empty fields",
            "fixed_8_fields": "a well of 6 fields never reaches 8: no",
            "fixed_4_fields_live_stop": 4,
            "ranking_top": "area",
            "ranking_unranked": ["intensity", "eccentricity"],
            "contrast_direction": "drug above control in every repeat",
            "rare_state": "clustered in 3 of the 6 fields of one well",
        },
    }


def main() -> None:
    """Write the two tables and ``expected.json`` next to this script."""
    objects, fovs = build()
    objects.to_csv(HERE / "synthetic_cells.csv", index=False)
    fovs.to_csv(HERE / "synthetic_fovs.csv", index=False)
    (HERE / "expected.json").write_text(json.dumps(expected(objects, fovs), indent=2) + "\n")
    print(f"{len(fovs)} fields, {len(objects)} cells written to {HERE}")


if __name__ == "__main__":
    main()

"""Builders shared by the workflow tests: the acquisition, its Setup, wells and report parts."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scenarios import FEATURE_COLS, _fov_table, _objects_for

from morphocoverage import RareTarget
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.retrospective import OutcomeCriterion, dataset_overview, microscopist_report
from morphocoverage.sampling_unit import fov_variability
from morphocoverage.workflows import Setup


def _tables(spec, seed=0, per_field=20, obs_sd=1.0):
    """``spec``: (condition, repeat, technical repeat, fields, well mean, field SD) per well."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean, field_sd in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "technical_repeat": well,
                "position_id": f"{repeat}_{condition}_{well}_f{j}",
            }
            fovs.append(key)
            offset = mean + rng.normal(0, field_sd)
            objects += [{**key, "x": v} for v in offset + rng.normal(0, obs_sd, per_field)]
    return pd.DataFrame(objects), pd.DataFrame(fovs)


def _report_parts(objects, fovs, group_cols):
    overview = dataset_overview(objects, fovs, feature_cols=["x"], group_cols=group_cols)
    variability = fov_variability(objects, fovs, feature="x", group_cols=group_cols)
    document = microscopist_report(overview=overview, variability=variability)
    return {"overview": overview, "variability": variability}, document


_WELLS = [
    ("ctrl", "r1", "A", 6, 10.0, 0.1),
    ("ctrl", "r1", "B", 6, 14.0, 0.1),
    ("ctrl", "r1", "C", 6, 8.0, 0.1),
    ("drug", "r1", "A", 6, 12.0, 0.1),
    ("drug", "r1", "B", 6, 16.0, 0.1),
]


def _wells(objects, fovs):
    fovs = fovs.assign(technical_repeat=[f"w{i % 2}" for i in range(len(fovs))])
    keys = ["condition", "biological_repeat", "image_file", "technical_repeat"]
    return objects.merge(fovs[keys]), fovs


READOUT = FEATURE_COLS[0]


def _acquisition(n_fovs: int = 16, n_objects: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    fov_tables, object_tables = [], []
    for condition, shift in (("control", 0.0), ("drug", 3.0)):
        for repeat in ("R1", "R2"):
            fovs = _fov_table(n_fovs, condition=condition, repeat=repeat, grid_width=4)
            objects = _objects_for(fovs, np.full(n_fovs, n_objects), rng)
            objects.loc[:, list(FEATURE_COLS)] += shift
            fov_tables.append(fovs)
            object_tables.append(objects)
    return pd.concat(object_tables, ignore_index=True), pd.concat(fov_tables, ignore_index=True)


def _setup(**changes) -> Setup:
    settings = {
        "readout": READOUT,
        "features": FEATURE_COLS,
        "control": {"condition": "control"},
        "acquisition_order_col": "acquisition_order",
        "stage_cols": ("stage_x", "stage_y"),
    }
    return Setup(**{**settings, **changes})


def _fingerprint(tolerance: float = 0.5) -> OutcomeCriterion:
    return OutcomeCriterion(
        CovarianceError(FEATURE_COLS, True),
        tolerance,
        "lower_is_better",
        f"the covariance of the features within {tolerance:.0%}",
    )


def _rare() -> RareTarget:
    return RareTarget(
        fov_incidence=0.2, detection_probability=0.95, min_fov_occurrences=1, phenotype="planted"
    )

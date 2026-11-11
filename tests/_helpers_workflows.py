"""Builders shared by the workflow tests: the acquisition, its Setup, wells and report parts."""

from __future__ import annotations

import sys
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scenarios import FEATURE_COLS, _fov_table, _objects_for

from morphocoverage import ObjectiveConfig, RareTarget, SamplingObjective
from morphocoverage.benchmark import AcquisitionOrder
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.retrospective import OutcomeCriterion, dataset_overview, microscopist_report
from morphocoverage.retrospective.frontier import BASELINE, FrontierResult
from morphocoverage.sampling_unit import fov_variability
from morphocoverage.sufficiency import sufficiency_report
from morphocoverage.workflows import Setup, report

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data" / "scenarios"))
import make_scenarios


def _tables(spec, seed=0, per_field=20, obs_sd=1.0):
    """``spec``: (condition, repeat, technical repeat, fields, well mean, field SD) per well."""
    rng = np.random.default_rng(seed)
    objects, fovs = [], []
    for condition, repeat, well, n_fields, mean, field_sd in spec:
        for j in range(n_fields):
            key = {
                "condition": condition,
                "biological_repeat": repeat,
                "repeat_group": repeat,
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


@cache
def _scenario_report(name: str) -> Any:
    """``report`` on a scenario's tables, as a user runs it, its warnings kept quiet."""
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return report(
            loaded.objects, loaded.fovs, Setup(**loaded.setup), feature_tables=loaded.feature_tables
        )


def _sets(layout: dict[str, list[str]], shifts: dict[str, float]) -> tuple[pd.DataFrame, ...]:
    """Conditions imaged in the repeats ``layout`` gives, 3 fields of 20 cells each, two
    lognormal features; ``shifts``: each condition's log shift of both."""
    rng = np.random.default_rng(16)
    rows = []
    for condition, repeats in layout.items():
        for repeat in repeats:
            batch = rng.normal(0, 0.1)
            for field in range(3):
                for label in range(20):
                    row = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "repeat_group": repeat,
                        "image_file": f"{condition}_{repeat}_{field}.tif",
                        "label_id": label,
                    }
                    for feature in ("area", "length"):
                        noise = rng.normal(0, 0.3) + batch + shifts.get(condition, 0.0)
                        row[feature] = 10 * np.exp(noise)
                    rows.append(row)
    objects = pd.DataFrame(rows)
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    return objects, fovs


SCREEN = {
    "ctrl": ["R1", "R2", "R3", "R4", "R5", "R6"],
    "siA": ["R1", "R2", "R3"],
    "siB": ["R4", "R5", "R6"],
}


def _report(name: str):
    loaded = make_scenarios.load(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(
            loaded.objects,
            loaded.fovs,
            Setup(**loaded.setup),
            feature_tables=loaded.feature_tables,
        )
    return loaded, result


def _homogeneous(n_features: int, n_fields: int, per_field: int = 1):
    rng = np.random.default_rng(0)
    names = [f"f{i}" for i in range(n_features)]
    rows = [
        {"condition": "a", "position_id": f"p{i}", **dict(zip(names, rng.normal(size=n_features)))}
        for i in range(n_fields)
        for _ in range(per_field)
    ]
    fovs = pd.DataFrame({"condition": "a", "position_id": [f"p{i}" for i in range(n_fields)]})
    return pd.DataFrame(rows), fovs, names


def _novelty_report(novelty, objects, fovs, names):
    return sufficiency_report(
        objects,
        fovs,
        feature_cols=names,
        strategies=[],
        objective=ObjectiveConfig(SamplingObjective.POPULATION),
        orders=[AcquisitionOrder.permuted(0)],
        novelty=novelty,
        fov_col="position_id",
        group_cols=["condition"],
    )


def _uneven_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for cond, repeat, n in (
        ("ctrl", "R1", 6),
        ("ctrl", "R2", 1),
        ("drug", "R1", 6),
        ("drug", "R2", 6),
    ):
        for i in range(n):
            rows.append(
                {
                    "position_id": f"{cond}{repeat}{i}",
                    "condition": cond,
                    "biological_repeat": repeat,
                    "repeat_group": repeat,
                    "area": 1.0 + i / 10,
                }
            )
    fovs = pd.DataFrame(rows)[["position_id", "condition", "biological_repeat"]]
    return pd.DataFrame(rows), fovs


QUICK = {
    "fov_counts": [2],
    "reservoir_fraction": 0.25,
    "n_splits": 1,
    "draws": 2,
    "permuted_orders": 1,
    "dispersed_orders": 0,
}


KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]


def _well_acquisition(n_fovs: int = 6, n_objects: int = 5):
    return _wells(*_acquisition(n_fovs=n_fovs, n_objects=n_objects))


ROOT = Path(__file__).resolve().parents[1]


DOCS = ROOT / "docs"


def _sictrl_tables(conditions=("siCTRL", "A"), fields=2, cells=3):
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in conditions:
        for field in range(fields):
            image = f"{condition}_{field}.tif"
            key = {
                "condition": condition,
                "biological_repeat": "2026_01_01",
                "repeat_group": "2026_01_01",
                "image_file": image,
            }
            fovs.append(key)
            for label in range(cells):
                rows.append({**key, "label_id": label + 1, "area": float(rng.normal(100, 10))})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


AREA_SETUP = Setup(readout="area", control={"condition": "siCTRL"})


def _flat(text: str) -> str:
    """Text with its wrapping undone, for substrings that may span a wrapped line."""
    return " ".join(text.split())


def _family_score(family, alpha, effort, saved=None, low=None, high=None, setting=0.1, **extra):
    return {
        "family": family,
        "alpha": alpha,
        "setting": setting,
        "effort": effort,
        "risk": 0.04,
        "fields_saved": saved,
        "saved_low": low,
        "saved_high": high,
        "saved_reached": 1.0,
        "adaptivity": 0.31,
        **extra,
    }


def _frontier_rows(rows, units=8):
    settings = {"label": "the mean", "units": units, "unit_cols": ["biological_repeat", "well"]}
    return FrontierResult(pd.DataFrame(), pd.DataFrame(rows), pd.DataFrame(), settings)


def _base(alpha, effort=6.0):
    return _family_score(BASELINE, alpha, effort, setting=effort)

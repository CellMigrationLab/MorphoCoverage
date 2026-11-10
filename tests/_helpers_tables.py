"""Builders shared by the tests of the tables: small acquisitions and the data-check titles."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from morphocoverage import check_tables, integrity_checks, validate_input_tables
from morphocoverage.objectives import ObjectiveConfig, SamplingObjective
from morphocoverage.sampling_unit import fov_variability


def _one_condition_tables(prefix: str = "f", n: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        {
            "condition": "a",
            "biological_repeat": "R1",
            "repeat_group": "R1",
            "position_id": [f"{prefix}{i}" for i in range(n)],
        }
    )
    rows = [
        {
            "condition": "a",
            "biological_repeat": "R1",
            "repeat_group": "R1",
            "position_id": p,
            "x": float(v),
        }
        for p in fovs["position_id"]
        for v in rng.normal(size=8)
    ]
    return pd.DataFrame(rows), fovs


def _variability(objects, fovs):
    return fov_variability(objects, fovs, feature="x")


def _two_condition_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    rows, fovs = [], []
    for condition in ("ctrl", "drug"):
        for repeat in ("R1", "R2", "R3"):
            for f in range(6):
                pid = f"{condition}{repeat}{f}"
                fovs.append(
                    {
                        "position_id": pid,
                        "condition": condition,
                        "biological_repeat": repeat,
                        "repeat_group": repeat,
                    }
                )
                for _ in range(6):
                    rows.append({**fovs[-1], "x": rng.normal(), "y": rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(fovs)


OBJECTS, FOVS = _two_condition_tables()


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _mixed(order: list[int]) -> pd.DataFrame:
    """Four FOVs whose identifiers print alike in pairs: 1 / "1" and 2 / "2"."""
    fovs = pd.DataFrame(
        {
            "position_id": [1, "1", 2, "2"],
            "condition": "c",
            "block": [1, "1", 2, "2"],
            "v": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return fovs.iloc[order].reset_index(drop=True)


def _validated_mixed(order: list[int]):
    fovs = _mixed(order)
    return validate_input_tables(
        objects=fovs, fovs=fovs[["position_id", "condition", "block"]], feature_cols=["v"]
    )


def _labelled_tables(label):
    fovs = pd.DataFrame({"condition": "a", "position_id": [f"f{i}" for i in range(8)]})
    objects = fovs.loc[fovs.index.repeat(3)].reset_index(drop=True)
    rng = np.random.default_rng(0)
    objects = objects.assign(x=rng.normal(size=len(objects)), state=label(len(objects)))
    return objects, fovs


def _cells(**columns) -> pd.DataFrame:
    """Four cells in two fields of one condition and repeat, with ``columns`` added."""
    table = pd.DataFrame(
        {
            "condition": ["a"] * 4,
            "biological_repeat": ["r1"] * 4,
            "repeat_group": ["r1"] * 4,
            "position_id": ["p1", "p1", "p2", "p2"],
            "area": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return table.assign(**columns)


def _fields(objects: pd.DataFrame) -> pd.DataFrame:
    return objects[["condition", "biological_repeat", "position_id"]].drop_duplicates()


def _agree(objects, fovs, **arguments):
    """Run both; ``check_tables`` never raises and accepts exactly what the validator accepts."""
    arguments = {"feature_cols": ["area"], "observation_unit": "cell", **arguments}
    check = check_tables(objects, fovs, **arguments)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            validate_input_tables(objects, fovs=fovs, **arguments)
        accepted = True
    except (ValueError, TypeError):
        accepted = False
    assert check.ok == accepted, check.summary()
    return accepted, check


def _error_titles(check, level="error"):
    return [p.title for p in check.problems if p.level == level]


# Every case where the two rule sets disagreed (A-1 ... A-9, A-19, A-21): the tables and
# the arguments. The same answer is required from both, whether accepted or refused.
_BASE = _cells()


def _six_cell_tables():
    """Six cells in four fields of two conditions, and the FOV table."""
    fovs = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "biological_repeat": ["r1"] * 4,
            "repeat_group": ["r1"] * 4,
            "image_file": ["f1", "f2", "f3", "f4"],
        }
    )
    cells = pd.DataFrame(
        {
            "condition": ["A", "A", "A", "B", "B", "B"],
            "biological_repeat": ["r1"] * 6,
            "repeat_group": ["r1"] * 6,
            "image_file": ["f1", "f1", "f2", "f3", "f3", "f4"],
            "area": [1.0, 2, 3, 4, 5, 6],
        }
    )
    return cells, fovs


def _planes(n_positions=2, n_planes=3, n_cells=4, **columns):
    """Cells measured at every z-plane of each position, one file per plane."""
    rows = [
        {
            "condition": "A",
            "biological_repeat": "r1",
            "repeat_group": "r1",
            "position_id": f"p{p}",
            "image_file": f"p{p}_z{z}.tif",
            "z": z,
            "label_id": k,
            "area": float(10 * p + z + k),
        }
        for p in range(n_positions)
        for z in range(n_planes)
        for k in range(1, n_cells + 1)
    ]
    return pd.DataFrame(rows).assign(**columns)


KEY = ["condition", "biological_repeat", "technical_repeat", "image_file"]


CONTINUOUS_FEATURES = ["area", "intensity", "texture"]


def _continuous_tables(n_fields=40, size=30, seed=0, wells=4):
    """``n_fields`` fields of ``size`` (an int or one per field) continuous observations."""
    rng = np.random.default_rng(seed)
    fovs = pd.DataFrame(
        {
            "condition": "ctrl",
            "biological_repeat": "r1",
            "repeat_group": "r1",
            "technical_repeat": [f"W{i % wells}" for i in range(n_fields)],
            "image_file": [f"f{i:05d}.tif" for i in range(n_fields)],
        }
    )
    sizes = np.broadcast_to(size, n_fields)
    objects = fovs.loc[np.repeat(fovs.index, sizes)].reset_index(drop=True)
    objects["area"] = rng.lognormal(5, 0.3, len(objects))
    objects["intensity"] = rng.gamma(4, 50, len(objects))
    objects["texture"] = rng.normal(0, 1, len(objects))
    return objects, fovs


def _rows(objects, fovs, i):
    return objects[objects["image_file"].eq(fovs["image_file"].iloc[i])].copy()


def _replace(objects, fovs, target, rows):
    """``objects`` with field ``target``'s observations replaced by ``rows``."""
    keep = objects[~objects["image_file"].eq(fovs["image_file"].iloc[target])]
    identity = fovs.iloc[target][KEY].to_dict()
    return pd.concat([keep, rows.assign(**identity)], ignore_index=True)


def _checks(objects, fovs, features=CONTINUOUS_FEATURES, **options):
    return integrity_checks(
        objects, fovs, feature_cols=features, observation_unit="cell", **options
    )


def _found(result):
    return result.findings["check"].tolist()


def _plate_tables(n_repeats=2, wells=("B02", "B03"), fields=3, cells=4, seed=0):
    """Cells in fields of two conditions: condition > repeat > well > field > cell."""
    rng = np.random.default_rng(seed)
    fov_rows, cell_rows = [], []
    for condition, row in (("control", "B"), ("drug", "C")):
        for r in range(n_repeats):
            repeat = f"2026_01_0{r + 1}"
            for well in wells:
                well = row + well[1:]
                for s in range(fields):
                    image = f"{repeat}_{well}_s{s + 1:02d}.tif"
                    ids = {
                        "condition": condition,
                        "biological_repeat": repeat,
                        "repeat_group": repeat,
                        "technical_repeat": well,
                        "image_file": image,
                    }
                    fov_rows.append(ids)
                    for k in range(cells):
                        cell_rows.append({**ids, "label_id": k + 1, "area": rng.normal(100, 10)})
    return pd.DataFrame(cell_rows), pd.DataFrame(fov_rows)


def _warning_titles(check, level="warning"):
    return [p.title for p in check.problems if p.level == level]


POPULATION = ObjectiveConfig(SamplingObjective.POPULATION)


def _four_cell_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    objects = pd.DataFrame(
        {
            "position_id": ["a", "a", "b", "b"],
            "condition": ["A", "A", "B", "B"],
            "biological_repeat": ["r1"] * 4,
            "repeat_group": ["r1"] * 4,
            "label_id": [1, 2, 1, 2],
            "area": [1.0, 2.0, 3.0, 4.5],
            "z": [0, 0, 0, 0],
        }
    )
    fovs = pd.DataFrame(
        {
            "position_id": ["a", "b"],
            "condition": ["A", "B"],
            "biological_repeat": ["r1", "r1"],
            "repeat_group": ["r1", "r1"],
        }
    )
    return objects, fovs

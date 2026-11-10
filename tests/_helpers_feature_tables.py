"""Builders shared by the tests of other feature tables: cells, tables, batches, wide tables."""

import numpy as np
import pandas as pd

from morphocoverage.feature_tables import attach_feature_tables
from morphocoverage.multivariate_outcomes import CovarianceError
from morphocoverage.retrospective.criteria import OutcomeCriterion
from morphocoverage.workflows import Setup


def _tables(cells: int = 2) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Two conditions, two repeats, two fields each, ``cells`` labelled cells a field; an
    embedding of 3 columns in another row order."""
    rows = [
        {
            "condition": c,
            "biological_repeat": r,
            "repeat_group": r,
            "image_file": f"f{f}.tif",
            "label_id": k,
        }
        for c in ("ctrl", "drug")
        for r in ("R1", "R2")
        for f in range(2)
        for k in range(1, cells + 1)
    ]
    objects = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    objects["area"] = rng.normal(size=len(objects))
    objects["shape"] = rng.normal(size=len(objects))
    fovs = objects[["condition", "biological_repeat", "image_file"]].drop_duplicates()
    embedding = objects[["condition", "biological_repeat", "image_file", "label_id"]].copy()
    for j in range(3):
        embedding[f"e{j}"] = rng.normal(size=len(embedding))
    return objects, fovs.reset_index(drop=True), embedding.iloc[::-1].reset_index(drop=True)


KEY = ["condition", "biological_repeat", "image_file", "label_id"]


def _cells(
    fields: int = 4,
    cells: int = 15,
    conditions: tuple[str, ...] = ("ctrl", "a", "b"),
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, np.random.Generator]:
    """Conditions in three repeats, ``fields`` fields of ``cells`` labelled cells each."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": c,
            "biological_repeat": r,
            "repeat_group": r,
            "image_file": f"{c}_{r}_{f}.tif",
            "label_id": k,
        }
        for c in conditions
        for r in ("R1", "R2", "R3")
        for f in range(fields)
        for k in range(1, cells + 1)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    return objects, fovs, rng


def _table(objects: pd.DataFrame, values: np.ndarray, prefix: str) -> pd.DataFrame:
    columns = {f"{prefix}{j}": values[:, j] for j in range(values.shape[1])}
    return pd.concat([objects[KEY].reset_index(drop=True), pd.DataFrame(columns)], axis=1)


def _without(table: pd.DataFrame, condition: str, repeat: str) -> pd.DataFrame:
    """``table`` without the rows of ``condition`` in ``repeat``."""
    gone = (table["condition"] == condition) & (table["biological_repeat"] == repeat)
    return table[~gone].reset_index(drop=True)


def _batch(drop_ctrl_r3: bool = False, effect: float = 1.5, batch: float = 3.0):
    """3 conditions in 3 repeats; DINO of 6 columns with a shift per repeat (a batch) and
    condition ``a`` shifted on the first two columns."""
    objects, fovs, rng = _cells(fields=4, cells=8)
    if drop_ctrl_r3:
        keep = ~((objects["condition"] == "a") & (objects["biological_repeat"] == "R3"))
        objects = objects[keep].reset_index(drop=True)
        fovs = fovs[~((fovs["condition"] == "a") & (fovs["biological_repeat"] == "R3"))]
        fovs = fovs.reset_index(drop=True)
    values = rng.normal(size=(len(objects), 6))
    shifts = {r: rng.normal(size=6) * batch for r in ("R1", "R2", "R3")}
    values += np.vstack([shifts[r] for r in objects["biological_repeat"]])
    values[objects["condition"].eq("a").to_numpy(), :2] += effect
    table = _table(objects, values, "d")
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    merged["x0"] = 1.0
    names = tuple(f"d{j}" for j in range(6))
    return merged, fovs, names


def _setup(names, **kwargs) -> Setup:
    return Setup(
        readout="x0",
        control={"condition": "ctrl"},
        feature_tables={"DINO": names},
        observation_unit="cell",
        **kwargs,
    )


def _wide(fields: int, cells: int, columns: int = 60, seed: int = 0):
    """Explainable area and shape, and a wide DINO table of correlated columns."""
    objects, fovs, rng = _cells(fields=fields, cells=cells, conditions=("ctrl", "a"), seed=seed)
    objects["area"] = rng.normal(size=len(objects))
    objects["shape"] = rng.normal(size=len(objects))
    latent = rng.normal(size=(len(objects), 3))
    values = latent @ rng.normal(size=(3, columns)) + 0.3 * rng.normal(size=(len(objects), columns))
    table = _table(objects, values, "d")
    merged = attach_feature_tables(objects, fovs, {"DINO": table}).objects
    names = tuple(f"d{j}" for j in range(columns))
    return merged, fovs, names


def _two_tables(seed: int = 0) -> tuple[np.ndarray, list[str]]:
    """4 correlated columns (A) and 2 independent ones (B), robustly standardised enough."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(300, 1))
    first = latent + 0.3 * rng.normal(size=(300, 4))
    return np.hstack([first, rng.normal(size=(300, 2))]), ["a0", "a1", "a2", "a3", "b0", "b1"]


COLUMNS = tuple(f"d{j}" for j in range(30))


def _fingerprint_data(pilot: bool, fields: int = 6, cells: int = 25):
    """ctrl and a in 3 repeats, ``fields`` fields of ``cells`` cells, a table of 30 columns
    on 3 latent axes; with ``pilot``, a condition of 4 fields of 3 cells in R1."""
    rng = np.random.default_rng(0)
    load = rng.normal(size=(3, len(COLUMNS)))
    groups = [(c, r, fields, cells) for c in ("ctrl", "a") for r in ("R1", "R2", "R3")]
    groups += [("pilot", "R1", 4, 3)] if pilot else []
    frames = []
    for c, r, n_fields, n_cells in groups:
        for field in range(n_fields):
            values = rng.normal(size=(n_cells, 3)) @ load
            values += 0.3 * rng.normal(size=(n_cells, len(COLUMNS)))
            frame = pd.DataFrame(values, columns=list(COLUMNS))
            frame.insert(0, "label_id", np.arange(1, n_cells + 1))
            frame.insert(0, "image_file", f"{c}{r}{field}")
            frame.insert(0, "biological_repeat", r)
            frame.insert(0, "condition", c)
            frames.append(frame)
    objects = pd.concat(frames, ignore_index=True)
    return objects, objects[KEY[:3]].drop_duplicates().reset_index(drop=True)


def _fingerprint_setup() -> Setup:
    criterion = OutcomeCriterion(
        CovarianceError(COLUMNS, True), 0.15, "lower_is_better", "DINO covariance within 15%"
    )
    return Setup(
        fingerprint=(criterion,), feature_tables={"DINO": COLUMNS}, observation_unit="cell"
    )

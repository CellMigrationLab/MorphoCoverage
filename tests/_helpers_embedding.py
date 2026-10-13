"""Builders shared by the tests of embedding-only and one-image-per-field tables."""

import warnings

import numpy as np
import pandas as pd

from morphocoverage.objectives import RareTarget
from morphocoverage.workflows import Setup, report

KEY = ["condition", "biological_repeat", "image_file", "label_id"]


MITOTIC = RareTarget(
    detection_probability=0.95, min_fov_occurrences=2, object_prevalence=0.2, phenotype="mitotic"
)


def _embedding_only(**fields):
    objects, fovs, table = _cells()
    setup = Setup(control={"condition": "ctrl"}, observation_unit="cell", **fields)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = report(objects, fovs, setup, feature_tables={"DINO": table})
    return objects, fovs, table, result


FEATURES = ("f1", "f2", "f3")


def _one_per_field(seed: int = 0):
    """3 conditions in 3 repeats, 6 fields of one image each; features f1, f2, f3 with
    ``a`` shifted, and a table of 6 other columns."""
    rng = np.random.default_rng(seed)
    rows = [
        {"condition": c, "biological_repeat": f"R{r}", "image_file": f"{c}_{r}_{f}.tif"}
        for c in ("ctrl", "a", "b")
        for r in range(3)
        for f in range(6)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects.copy()
    shift = objects["condition"].eq("a").to_numpy() * 1.0
    for name in FEATURES:
        objects[name] = 10 + rng.normal(size=len(objects)) + shift
    table = objects[["condition", "biological_repeat", "image_file"]].assign(
        **{f"e{j}": rng.normal(size=len(objects)) for j in range(6)}
    )
    return objects, fovs, table


SEVERAL = {"condition": {"cA": ["a1"], "cB": ["b1"]}}


def _several(seed: int = 0):
    """cA, a1, cB, b1 in 3 repeats (cB not imaged in R2), 3 fields of 6 cells; ``area`` in
    the observation table, an explainable table ``fiber`` (fib_len, fib_w) and a table
    ``DINO`` of 4 columns, a1 and b1 shifted."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": c,
            "biological_repeat": f"R{r}",
            "image_file": f"{c}_{r}_{f}.tif",
            "label_id": k,
        }
        for c in ("cA", "a1", "cB", "b1")
        for r in range(3)
        for f in range(3)
        for k in range(1, 7)
        if (c, r) != ("cB", 1)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    shift = objects["condition"].isin(["a1", "b1"]).to_numpy() * 0.8
    objects["area"] = 10 * np.exp(rng.normal(0, 0.2, len(objects)) + shift)
    fiber = objects[KEY].assign(
        fib_len=5 + rng.normal(size=len(objects)) + shift, fib_w=2 + rng.normal(size=len(objects))
    )
    dino = objects[KEY].assign(**{f"d{j}": rng.normal(size=len(objects)) for j in range(4)})
    return objects, fovs, {"fiber": (fiber, "explainable"), "DINO": dino}


def _cells(seed: int = 0, columns: int = 12):
    """3 conditions in 3 repeats, 4 fields of 6 labelled cells; an embedding of ``columns``
    columns with condition ``a`` shifted; a 0/1 label in the observation table."""
    rng = np.random.default_rng(seed)
    rows = [
        {
            "condition": c,
            "biological_repeat": f"R{r}",
            "image_file": f"{c}_{r}_{f}.tif",
            "label_id": k,
        }
        for c in ("ctrl", "a", "b")
        for r in range(3)
        for f in range(4)
        for k in range(1, 7)
    ]
    objects = pd.DataFrame(rows)
    fovs = objects[KEY[:3]].drop_duplicates().reset_index(drop=True)
    values = rng.normal(size=(len(objects), columns))
    values[objects["condition"].eq("a").to_numpy(), :2] += 1.0
    table = pd.concat(
        [objects[KEY], pd.DataFrame(values, columns=[f"e{j}" for j in range(columns)])], axis=1
    )
    objects["mitotic"] = (rng.random(len(objects)) < 0.2).astype(int)
    return objects, fovs, table


EMPTY = {"condition": {"cA": {}, "cB": {}}}


NOTHING_UNDER = "have no condition under them and no parent"

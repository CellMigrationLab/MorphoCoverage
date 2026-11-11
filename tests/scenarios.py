"""Constructed acquisitions with known structure for contract tests (AGENTS.md §9).

These fixtures are not scientific benchmarks. Each one has a property that is
known by construction, so a contract test can check that the framework and
every strategy handle it correctly. Every builder is deterministic for a given
seed and returns linked MorphoBricks-style FOV and object tables.

``true_state`` is ground-truth metadata for tests. It is deliberately a numeric
column that must never reach a strategy as a phenotype feature.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLS = ("feature_a", "feature_b")
# Positive baseline so scalar CV-based strategies are defined, as for whole-cell
# measurements such as area or intensity.
FEATURE_BASELINE = 10.0
ORDER_COL = "acquisition_order"
DEFAULT_SEED = 20260923


@dataclass(frozen=True)
class Scenario:
    """Linked tables plus the facts that hold by construction."""

    name: str
    objects: pd.DataFrame
    fovs: pd.DataFrame
    truth: dict[str, Any] = field(default_factory=dict)


def _fov_table(
    n_fovs: int,
    *,
    condition: str = "control",
    repeat: str = "R1",
    grid_width: int = 10,
    area_um2: np.ndarray | None = None,
) -> pd.DataFrame:
    """One row per acquired FOV, acquired in raster order over a stage grid."""
    index = np.arange(n_fovs)
    return pd.DataFrame(
        {
            "condition": condition,
            "biological_repeat": repeat,
            "repeat_group": repeat,
            "image_file": [f"img_{i:03d}.tif" for i in index],
            ORDER_COL: index + 1,
            "stage_x": (index % grid_width) * 100.0,
            "stage_y": (index // grid_width) * 100.0,
            "area_um2": np.full(n_fovs, 10_000.0) if area_um2 is None else area_um2,
        }
    )


def _objects_for(
    fovs: pd.DataFrame,
    counts: np.ndarray,
    rng: np.random.Generator,
    *,
    means: np.ndarray | None = None,
) -> pd.DataFrame:
    """Draw objects per FOV; ``means`` optionally gives a per-FOV feature offset."""
    rows = []
    for position, (fov, count) in enumerate(zip(fovs.itertuples(index=False), counts)):
        offset = FEATURE_BASELINE + (0.0 if means is None else means[position])
        values = rng.normal(loc=offset, scale=1.0, size=(int(count), len(FEATURE_COLS)))
        for label_id, value in enumerate(values, start=1):
            rows.append(
                {
                    "condition": fov.condition,
                    "biological_repeat": fov.biological_repeat,
                    "repeat_group": fov.biological_repeat,
                    "image_file": fov.image_file,
                    "label_id": label_id,
                    "true_state": 0,
                    **dict(zip(FEATURE_COLS, value)),
                }
            )
    columns = [
        "condition",
        "biological_repeat",
        "image_file",
        "label_id",
        "true_state",
        *FEATURE_COLS,
    ]
    return pd.DataFrame(rows, columns=columns)


def _mark_rare(
    objects: pd.DataFrame,
    fovs_with_rare: dict[str, int],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Turn a chosen number of objects per FOV into a well-separated rare state."""
    objects = objects.copy()
    for image_file, n_rare in fovs_with_rare.items():
        candidates = objects.index[objects["image_file"].eq(image_file)]
        chosen = rng.choice(candidates, size=n_rare, replace=False)
        objects.loc[chosen, "true_state"] = 1
        objects.loc[chosen, list(FEATURE_COLS)] += 12.0
    return objects


def homogeneous(seed: int = DEFAULT_SEED) -> Scenario:
    rng = np.random.default_rng(seed)
    fovs = _fov_table(20)
    return Scenario("homogeneous", _objects_for(fovs, np.full(20, 30), rng), fovs)


def multimodal(seed: int = DEFAULT_SEED) -> Scenario:
    """Two modes separated by 10 SD, mixed half-and-half within every FOV."""
    rng = np.random.default_rng(seed)
    fovs = _fov_table(20)
    objects = _objects_for(fovs, np.full(20, 30), rng)
    second_mode = objects["label_id"] % 2 == 0
    objects.loc[second_mode, "true_state"] = 1
    objects.loc[second_mode, list(FEATURE_COLS)] += 10.0
    return Scenario("multimodal", objects, fovs, {"mode_fraction": 0.5})


def _rare_at_prevalence(name: str, n_rare: int, seed: int) -> Scenario:
    """2000 objects in 40 FOVs, with n_rare rare objects spread one per FOV where possible."""
    rng = np.random.default_rng(seed)
    fovs = _fov_table(40)
    objects = _objects_for(fovs, np.full(40, 50), rng)
    per_fov = np.bincount(np.arange(n_rare) % 40, minlength=40)
    placement = {f: int(k) for f, k in zip(fovs["image_file"], per_fov) if k}
    objects = _mark_rare(objects, placement, rng)
    return Scenario(name, objects, fovs, {"object_prevalence": n_rare / 2000})


def rare_prevalence_10pct(seed: int = DEFAULT_SEED) -> Scenario:
    return _rare_at_prevalence("rare_prevalence_10pct", 200, seed)


def rare_prevalence_1pct(seed: int = DEFAULT_SEED) -> Scenario:
    return _rare_at_prevalence("rare_prevalence_1pct", 20, seed)


def rare_prevalence_0p1pct(seed: int = DEFAULT_SEED) -> Scenario:
    return _rare_at_prevalence("rare_prevalence_0p1pct", 2, seed)


def _rare_one_percent(name: str, placement_fovs: list[int], seed: int) -> Scenario:
    """1% object prevalence (20 of 2000) placed evenly into the given FOV positions."""
    rng = np.random.default_rng(seed)
    fovs = _fov_table(40)
    objects = _objects_for(fovs, np.full(40, 50), rng)
    per_fov = 20 // len(placement_fovs)
    placement = {fovs["image_file"].iloc[i]: per_fov for i in placement_fovs}
    objects = _mark_rare(objects, placement, rng)
    return Scenario(
        name,
        objects,
        fovs,
        {
            "object_prevalence": 0.01,
            "fov_incidence": len(placement_fovs) / 40,
            "first_rare_step": min(placement_fovs) + 1,
        },
    )


def rare_uniform(seed: int = DEFAULT_SEED) -> Scenario:
    return _rare_one_percent("rare_uniform", list(range(0, 40, 2)), seed)


def rare_clustered(seed: int = DEFAULT_SEED) -> Scenario:
    # Late in acquisition order, so early stopping misses it entirely.
    return _rare_one_percent("rare_clustered", [30, 31], seed)


def unequal_objects_per_fov(seed: int = DEFAULT_SEED) -> Scenario:
    rng = np.random.default_rng(seed)
    fovs = _fov_table(30)
    counts = np.where(np.arange(30) % 5 == 0, 200, 5)
    return Scenario("unequal_objects_per_fov", _objects_for(fovs, counts, rng), fovs)


def zero_object_fovs(seed: int = DEFAULT_SEED) -> Scenario:
    rng = np.random.default_rng(seed)
    fovs = _fov_table(20)
    counts = np.where(np.isin(np.arange(20), [0, 7, 8, 19]), 0, 25)
    return Scenario(
        "zero_object_fovs",
        _objects_for(fovs, counts, rng),
        fovs,
        {"empty_fovs": ["img_000.tif", "img_007.tif", "img_008.tif", "img_019.tif"]},
    )


def unequal_fov_area(seed: int = DEFAULT_SEED) -> Scenario:
    """Two FOV sizes with object count proportional to imaged area."""
    rng = np.random.default_rng(seed)
    area = np.where(np.arange(20) % 2 == 0, 10_000.0, 40_000.0)
    fovs = _fov_table(20, area_um2=area)
    counts = (area / 10_000.0 * 10).astype(int)
    return Scenario("unequal_fov_area", _objects_for(fovs, counts, rng), fovs)


def neighbour_correlation(seed: int = DEFAULT_SEED) -> Scenario:
    """Phenotype drifts smoothly with stage position, so adjacent FOVs are similar."""
    rng = np.random.default_rng(seed)
    fovs = _fov_table(40, grid_width=10)
    means = np.sin(fovs["stage_x"].to_numpy() / 300.0) * 4 + fovs["stage_y"].to_numpy() / 100.0
    objects = _objects_for(fovs, np.full(40, 30), rng, means=means)
    return Scenario("neighbour_correlation", objects, fovs, {"fov_means": means})


def duplicated_objects(seed: int = DEFAULT_SEED) -> Scenario:
    """FOV img_000 holds its objects three times over; other FOVs are ordinary."""
    rng = np.random.default_rng(seed)
    fovs = _fov_table(10)
    objects = _objects_for(fovs, np.full(10, 20), rng)
    first = objects[objects["image_file"].eq("img_000.tif")]
    duplicated = pd.concat([objects, first, first], ignore_index=True)
    return Scenario(
        "duplicated_objects",
        duplicated,
        fovs,
        {"duplicated_fov": "img_000.tif", "n_distinct_objects": len(objects)},
    )


def condition_repeat_groups(seed: int = DEFAULT_SEED) -> Scenario:
    """2 conditions x 2 repeats that reuse the same image names in every folder."""
    rng = np.random.default_rng(seed)
    fov_tables, object_tables = [], []
    for condition, shift in (("control", 0.0), ("drug", 5.0)):
        for repeat in ("R1", "R2"):
            fovs = _fov_table(8, condition=condition, repeat=repeat)
            objects = _objects_for(fovs, np.full(8, 15), rng)
            objects.loc[:, list(FEATURE_COLS)] += shift
            fov_tables.append(fovs)
            object_tables.append(objects)
    return Scenario(
        "condition_repeat_groups",
        pd.concat(object_tables, ignore_index=True),
        pd.concat(fov_tables, ignore_index=True),
        {"n_groups": 4, "fovs_per_group": 8},
    )


def batch_shift(seed: int = DEFAULT_SEED) -> Scenario:
    """Same biology in two repeats; repeat R2 carries a technical offset of +3."""
    rng = np.random.default_rng(seed)
    fov_tables, object_tables = [], []
    for repeat, shift in (("R1", 0.0), ("R2", 3.0)):
        fovs = _fov_table(10, repeat=repeat)
        objects = _objects_for(fovs, np.full(10, 20), rng)
        objects.loc[:, list(FEATURE_COLS)] += shift
        fov_tables.append(fovs)
        object_tables.append(objects)
    return Scenario(
        "batch_shift",
        pd.concat(object_tables, ignore_index=True),
        pd.concat(fov_tables, ignore_index=True),
        {"shift": 3.0},
    )


SCENARIOS: dict[str, Callable[..., Scenario]] = {
    builder.__name__: builder
    for builder in (
        homogeneous,
        multimodal,
        rare_prevalence_10pct,
        rare_prevalence_1pct,
        rare_prevalence_0p1pct,
        rare_uniform,
        rare_clustered,
        unequal_objects_per_fov,
        zero_object_fovs,
        unequal_fov_area,
        neighbour_correlation,
        duplicated_objects,
        condition_repeat_groups,
        batch_shift,
    )
}

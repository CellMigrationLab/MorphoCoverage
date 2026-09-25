"""Shared helpers of the sampling-unit checks (private)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..evaluator import iter_analysis_groups
from ..schema import ValidatedInput, fov_positions, validate_input_tables


def validated(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    feature_cols: Sequence[str],
    fov_col: str | None,
    group_cols: Sequence[str] | None,
    observation_unit: str | None,
) -> ValidatedInput:
    """The linked tables, validated as every other analysis validates them."""
    return validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(feature_cols),
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )


def groups(data: ValidatedInput):
    """The independent analysis groups, as ``(group, data)`` pairs."""
    return list(iter_analysis_groups(data))


def fov_values(data: ValidatedInput, values: np.ndarray) -> list[np.ndarray]:
    """``values`` split by acquired FOV, one array per FOV row (empty for zero-observation FOVs).

    ``values`` is aligned with ``data.objects``; missing values (NaN) are dropped.
    """
    positions = fov_positions(data, data.fovs)
    keep = ~np.isnan(values)
    positions, values = positions[keep], values[keep]
    order = np.argsort(positions, kind="stable")
    counts = np.bincount(positions, minlength=len(data.fovs))
    return np.split(values[order], np.cumsum(counts)[:-1])


def check_seed(random_state: object) -> int:
    if isinstance(random_state, bool) or not isinstance(random_state, int):
        raise TypeError("random_state must be an integer seed")
    return random_state


def check_count(value: object, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def matching_groups(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    selector: Mapping[str, Any],
    name: str,
) -> list[dict[str, Any]]:
    """The groups whose FOV metadata match ``selector`` (for example controls)."""
    if not isinstance(selector, Mapping) or not selector:
        raise ValueError(f"{name} must be a non-empty mapping of FOV-table column to value")
    chosen = []
    for group, data in all_groups:
        missing = [c for c in selector if c not in data.fovs.columns]
        if missing:
            raise ValueError(f"{name} names columns absent from the FOV table: {missing}")
        values = data.fovs[list(selector)].drop_duplicates()
        if len(values) == 1 and all(values.iloc[0][c] == v for c, v in selector.items()):
            chosen.append(group)
    if not chosen:
        raise ValueError(f"no analysis group matches {name} {dict(selector)}")
    return chosen

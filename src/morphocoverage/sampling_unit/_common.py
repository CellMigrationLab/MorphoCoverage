"""Shared helpers of the sampling-unit checks (private).

Shared-private: ``validated``, ``groups``, ``matching_groups``,
``no_match_message`` and ``canonical_fields`` are used by the modules of
``sampling_unit``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .._control import control_text, group_matches
from ..data_check import validate_input_tables
from ..evaluator import iter_analysis_groups
from ..schema import ValidatedInput


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


def groups(data: ValidatedInput) -> list[tuple[dict[str, Any], ValidatedInput]]:
    """The independent analysis groups, as ``(group, data)`` pairs."""
    return list(iter_analysis_groups(data))


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
        if group_matches(data.fovs, selector):  # a tuple accepts any of its values
            chosen.append(group)
    if not chosen:
        raise ValueError(no_match_message(all_groups, selector, name))
    return chosen


def no_match_message(
    all_groups: list[tuple[dict[str, Any], ValidatedInput]],
    selector: Mapping[str, Any],
    name: str,
) -> str:
    """Why ``selector`` matched no group, with the values its first column holds."""
    column = next(iter(selector))
    present = sorted(
        {
            str(v)
            for _, data in all_groups
            if column in data.fovs
            for v in data.fovs[column].unique()
        }
    )
    return (
        f"no analysis group matches {name} {control_text(selector)}; {column!r} holds "
        f"{present[:8]} "
        "(values are matched exactly, case and spaces included)"
    )


def canonical_fields(data: ValidatedInput) -> np.ndarray:
    """The rows of ``data.fovs`` in a canonical order: sorted by the FOV key's values as text.

    Resampling that walks the fields in this order draws the same fields for
    the same data, whatever the row order of the FOV table (and whatever the
    order of a categorical column's categories: values, not codes, are sorted).
    """
    key = list(data.fov_key)
    rows = list(zip(*(data.fovs[c].astype(object).map(repr).tolist() for c in key)))
    return np.array(sorted(range(len(rows)), key=rows.__getitem__), dtype=np.int64)

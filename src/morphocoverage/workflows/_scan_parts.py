"""Parts of the design scan (private; split out of ``workflows.scan``): the controls as
declared, the feature tables by tag, the planes per field, the targets' reagents and the
observations per condition and repeat, and the FOV table's group columns attached to lean
observation tables.

Shared-private: used by ``workflows.scan`` (``scan_design`` and its ``_design``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from ..metadata import (
    CONDITION_COLUMN,
    REPEAT_COLUMN,
    TARGET_COLUMN,
    WITHIN_FOV_METADATA_COLUMNS,
    column_has_values,
    sorted_labels,
)
from ..schema import _attach_fov_metadata


def declared_controls(control: Mapping[str, Any] | None) -> tuple[list[Any], Any, Any]:
    """The several controls declared in ``Setup``'s form (none for one control), the control
    ``check_tables`` reads (their pool) and the control value drawn first."""
    control_value = dict(control).get(CONDITION_COLUMN) if control else None
    # Several controls in Setup's form (owner, 2026-10-05): checked as their pool, the first
    # declared drawn first, as the report does (audit 8B).
    controls = list(control_value) if isinstance(control_value, Mapping) else []
    checked_control = {CONDITION_COLUMN: tuple(controls)} if controls else control
    control_value = controls[0] if controls else control_value
    return controls, checked_control, control_value


def with_fov_groups(
    objects: pd.DataFrame, fovs: pd.DataFrame | None, fov_col: str, groups: list[str]
) -> pd.DataFrame:
    """``objects`` with the group columns only the FOV table holds, attached as the validator
    attaches them (lean observation tables, #363); ``objects`` as given when they cannot be
    attached unambiguously, which the data check reports."""
    if fovs is None or all(c in objects.columns for c in groups):
        return objects
    try:
        return _attach_fov_metadata(objects, fovs, fov_col, groups)
    except (ValueError, TypeError, KeyError):  # fallback: the data check names the problem
        return objects


def tagged_tables(tables: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """The explainable tables' columns and the other tables' columns, by name (none without
    feature tables)."""
    explainable = tables.explainable if tables else {}
    others = tables.other if tables else {}
    return explainable, others


def planes(objects: pd.DataFrame, fov_key: list[str]) -> dict[str, tuple[int, int]]:
    """The planes (Z, C or T) per field, fewest and most, of each axis with more than one."""
    found = {}
    for axis in WITHIN_FOV_METADATA_COLUMNS:
        if column_has_values(objects, axis) and fov_key:
            per = objects.groupby(fov_key, sort=False, dropna=False)[axis].nunique()
            if int(per.max()) > 1:
                found[axis] = (int(per.min()), int(per.max()))
    return found


def targets(objects: pd.DataFrame, has_condition: bool) -> dict[Any, list[Any]] | None:
    """The conditions of each ``target`` (``None`` without targets or conditions)."""
    if not (column_has_values(objects, TARGET_COLUMN) and has_condition):
        return None
    by_target = objects.groupby(TARGET_COLUMN, sort=False)[CONDITION_COLUMN].agg(
        lambda values: sorted_labels(values)
    )
    return {t: by_target[t] for t in sorted_labels(by_target.index)}


def observation_layout(
    objects: pd.DataFrame,
    conditions: list[Any],
    repeats: list[Any],
    has_condition: bool,
    has_repeat: bool,
) -> pd.DataFrame:
    """The observations per condition (rows) and biological repeat (columns), or per
    condition alone without repeats; empty without conditions."""
    if has_condition and has_repeat:
        layout = objects.groupby([CONDITION_COLUMN, REPEAT_COLUMN], sort=False).size().unstack()
        return layout.reindex(index=conditions, columns=repeats)
    if has_condition:
        layout = objects.groupby(CONDITION_COLUMN, sort=False).size().reindex(conditions)
        return layout.to_frame("observations")
    return pd.DataFrame()


__all__ = [
    "declared_controls",
    "observation_layout",
    "planes",
    "tagged_tables",
    "targets",
    "with_fov_groups",
]

"""Normalise features to the control condition, repeat by repeat.

The JCB rule and the lab's Plot&Stats notebook express values relative to the
control: each value is divided by the mean of the control condition's objects
of the same repeat, so a treated mean of 1.2 reads as 20% above its own
repeat's control and repeats with different absolute levels become
comparable. ``normalize_to_control`` does exactly that for every feature
column asked for, and nothing else: it neither centres nor scales by a
standard deviation (that is ``transforms.Standardize``, fitted per group on
visible fields), and it never crosses repeats.

It is a retrospective preparation of the tables: it needs the control fields
of every repeat, so a live session, which sees one group at a time, runs on
raw values. A repeat without a control group raises, since normalising some
repeats and not others would mix scales inside one analysis; so does a control
mean of zero or one undefined by missing values, because dividing by it has
no meaning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .data_check import validate_input_tables
from .metadata import MORPHOBRICKS_GROUP_COLUMNS

CONTROL_MEAN = "control_mean"
METHODS = (CONTROL_MEAN,)


def control_means(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> pd.DataFrame:
    """The mean of every feature over the control condition's objects, one row per repeat.

    ``control`` names the control condition by FOV-table value, for example
    ``{"condition": "MOCK"}``. Without a repeat column (or ``repeat_col=None``)
    the whole table is one repeat, labelled ``None``. Raises when a repeat has
    no control objects or a control mean is zero or undefined.
    """
    if not isinstance(control, Mapping) or condition_col not in control:
        raise ValueError(f"control must name the condition column {condition_col!r}")
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(feature_cols),
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    attached = data.objects
    for column in control:
        if column not in attached.columns:
            raise ValueError(f"control names a column absent from the tables: {column!r}")
    if repeat_col is not None and repeat_col not in attached.columns:
        repeat_col = None
    is_control = np.ones(len(attached), dtype=bool)
    for column, value in control.items():
        is_control &= attached[column].to_numpy() == value
    repeats = attached[repeat_col] if repeat_col else pd.Series([None] * len(attached))
    rows = []
    for repeat in dict.fromkeys(repeats.tolist()):
        in_repeat = repeats.to_numpy() == repeat if repeat is not None else np.ones_like(is_control)
        chosen = attached.loc[in_repeat & is_control, list(feature_cols)]
        label = f"{repeat_col}={repeat}" if repeat_col else "the whole table"
        if chosen.empty:
            raise ValueError(
                f"no control objects {dict(control)} in {label}: every repeat needs its control "
                "to be normalised; drop the repeat or turn normalisation off"
            )
        means = chosen.mean(axis=0, skipna=True)
        bad = [c for c in feature_cols if not np.isfinite(means[c]) or means[c] == 0]
        if bad:
            raise ValueError(
                f"the control mean of {bad} in {label} is zero or undefined: these features "
                "cannot be divided by it"
            )
        rows.append({"repeat": repeat, "control_objects": len(chosen), **means.to_dict()})
    return pd.DataFrame(rows)


def normalize_to_control(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = MORPHOBRICKS_GROUP_COLUMNS[1],
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> pd.DataFrame:
    """A copy of ``objects`` with each feature divided by its repeat's control mean.

    Every other column, the row order and the index are kept, so the result
    replaces ``objects`` in any analysis. See ``control_means`` for the
    arguments and the errors.
    """
    means = control_means(
        objects,
        fovs,
        feature_cols=feature_cols,
        control=control,
        condition_col=condition_col,
        repeat_col=repeat_col,
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    data = validate_input_tables(
        objects,
        fovs=fovs,
        feature_cols=list(feature_cols),
        fov_col=fov_col,
        group_cols=group_cols,
        observation_unit=observation_unit,
    )
    attached = data.objects
    if len(attached) != len(objects):
        raise ValueError("validation changed the number of object rows; cannot align the values")
    if repeat_col is not None and repeat_col not in attached.columns:
        repeat_col = None
    normalized = objects.copy()
    if repeat_col is None:
        divisors = means.iloc[[0] * len(objects)][list(feature_cols)].to_numpy()
    else:
        by_repeat = means.set_index("repeat")[list(feature_cols)]
        divisors = by_repeat.loc[attached[repeat_col].tolist()].to_numpy()
    values = objects[list(feature_cols)].to_numpy(dtype=float) / divisors
    normalized[list(feature_cols)] = values
    return normalized


__all__ = ["CONTROL_MEAN", "METHODS", "control_means", "normalize_to_control"]

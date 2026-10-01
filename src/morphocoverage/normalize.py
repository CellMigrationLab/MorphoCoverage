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
mean that is zero (up to rounding), negative or undefined by missing values:
dividing by zero has no meaning, and dividing by a negative mean flips the
sign of every value, so a treated mean above its control would read as below
it (a signed feature such as a velocity or a log-ratio is compared raw). A
repeat column named explicitly but absent from the tables raises too, rather
than pooling the repeats; only the default ``"biological_repeat"`` may be absent.

A signed feature whose control mean happens to be positive is still
normalised, but ``normalize_to_control`` warns (``UserWarning``) whenever the
control objects hold negative values of a feature: such a feature has no
ratio scale, so a value divided by the control mean is not a multiple of the
control, and a control mean near zero inflates every value (a log-ratio with
control mean 0.05 and treated mean 0.3 reads "6 x control"). The same
sentence is in the result's ``attrs["normalization_note"]``, and
``control_means`` lists the features in its ``negative_features`` column
(``negative_values_sentence`` words it), so a report can state it beside
the values.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._former_repeat import one_warning, read_former
from ._wording import listing
from .data_check import validate_input_tables
from .metadata import MORPHOBRICKS_GROUP_COLUMNS, REPEAT_COLUMN
from .statistics import mean_is_zero

CONTROL_MEAN = "control_mean"
METHODS = (CONTROL_MEAN,)


class _DefaultRepeat(str):
    """The default repeat column: the only name that may be absent from the tables."""


# A str subclass, so the default still reads (and compares) as "biological_repeat".
_DEFAULT_REPEAT_COL = _DefaultRepeat(MORPHOBRICKS_GROUP_COLUMNS[1])


def _repeat_column(repeat_col: str | None, columns: pd.Index) -> str | None:
    """The repeat column to use: ``None`` for one implicit repeat.

    The default name may be absent (the whole table is then one repeat); a
    name given explicitly must be a column, since a misspelt one would
    otherwise pool every repeat silently.
    """
    if repeat_col is None or repeat_col in columns:
        return repeat_col
    if isinstance(repeat_col, _DefaultRepeat):
        return None
    raise ValueError(
        f"repeat_col {repeat_col!r} is not a column of the tables: name the column holding "
        "the repeats, or pass repeat_col=None to treat the whole table as one repeat"
    )


def _refused_means(means: pd.Series, chosen: pd.DataFrame, feature_cols: Sequence[str]) -> list:
    """(feature, reason) for every control mean that cannot divide the values."""
    refused = []
    for column in feature_cols:
        mean = means[column]
        if not np.isfinite(mean):
            refused.append((column, "undefined"))
        elif mean_is_zero(mean, chosen[column].to_numpy(dtype=float)):
            refused.append((column, "zero"))
        elif mean < 0:
            refused.append((column, f"negative ({mean:.3g})"))
    return refused


@one_warning
def control_means(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = _DEFAULT_REPEAT_COL,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> pd.DataFrame:
    """The mean of every feature over the control condition's objects, one row per repeat.

    ``control`` names the control condition by FOV-table value, for example
    ``{"condition": "MOCK"}``. Without a repeat column (or ``repeat_col=None``)
    the whole table is one repeat, labelled ``None``; a ``repeat_col`` named
    explicitly must be a column. Raises when a repeat has no control objects
    or a control mean is zero (up to rounding), negative or undefined.

    Each row holds ``biological_repeat``, ``control_objects`` (the number of control
    objects the means rest on: an object missing a feature's value is left
    out of that feature's mean, so with missing values this is the smallest
    count over the features), ``negative_features`` (the features whose
    control objects in that repeat include negative values, as a tuple; see
    the module docstring) and the mean of every feature. The former name
    ``repeat`` (a column, a key of ``control``, ``repeat_col``) is read as
    ``biological_repeat``, with a ``DeprecationWarning``.
    """
    objects, fovs, control, repeat_col, fov_col, group_cols = read_former(
        objects=objects,
        fovs=fovs,
        control=control,
        repeat_col=repeat_col,
        fov_col=fov_col,
        group_cols=group_cols,
    )
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
    repeat_col = _repeat_column(repeat_col, attached.columns)
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
        refused = _refused_means(means, chosen, feature_cols)
        if refused:
            listed = ", ".join(f"{column!r} is {reason}" for column, reason in refused)
            raise ValueError(
                f"the control mean in {label}: {listed}. Dividing by a zero or undefined mean "
                "has no meaning and dividing by a negative one flips the sign of the effect: "
                "leave these features out of normalisation (compare them raw)"
            )
        with_values = int(chosen.notna().sum(axis=0).min())
        negative = tuple(column for column in feature_cols if (chosen[column] < 0).any())
        rows.append(
            {
                REPEAT_COLUMN: repeat,
                "control_objects": with_values,
                "negative_features": negative,
                **means.to_dict(),
            }
        )
    return pd.DataFrame(rows)


def negative_values_sentence(means: pd.DataFrame) -> str | None:
    """The caution for features whose control holds negative values, or ``None`` without any.

    ``means`` is a ``control_means`` table; the sentence names the features
    and says why their normalised values are not multiples of the control.
    """
    features = list(dict.fromkeys(f for row in means["negative_features"] for f in row))
    if not features:
        return None
    return (
        f"the control objects hold negative values of {listing([repr(f) for f in features])}: "
        "a feature that can change sign has no ratio scale, so its value divided by the "
        "control mean is not a multiple of the control, and a control mean near zero "
        "inflates every value; compare such a feature raw"
    )


@one_warning
def normalize_to_control(
    objects: pd.DataFrame,
    fovs: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    control: Mapping[str, Any],
    condition_col: str = MORPHOBRICKS_GROUP_COLUMNS[0],
    repeat_col: str | None = _DEFAULT_REPEAT_COL,
    fov_col: str | None = None,
    group_cols: Sequence[str] | None = None,
    observation_unit: str | None = None,
) -> pd.DataFrame:
    """A copy of ``objects`` with each feature divided by its repeat's control mean.

    Every other column, the row order and the index are kept, so the result
    replaces ``objects`` in any analysis. See ``control_means`` for the
    arguments and the errors. When the control objects hold negative values of
    a feature, the values are still divided but a ``UserWarning`` says why they
    are not multiples of the control, and ``attrs["normalization_note"]``
    keeps the sentence. A ``repeat`` column is returned as ``biological_repeat``
    (see ``control_means``).
    """
    objects, fovs, control, repeat_col, fov_col, group_cols = read_former(
        objects=objects,
        fovs=fovs,
        control=control,
        repeat_col=repeat_col,
        fov_col=fov_col,
        group_cols=group_cols,
    )
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
    repeat_col = _repeat_column(repeat_col, attached.columns)
    normalized = objects.copy()
    if repeat_col is None:
        divisors = means.iloc[[0] * len(objects)][list(feature_cols)].to_numpy()
    else:
        by_repeat = means.set_index(REPEAT_COLUMN)[list(feature_cols)]
        divisors = by_repeat.loc[attached[repeat_col].tolist()].to_numpy()
    values = objects[list(feature_cols)].to_numpy(dtype=float) / divisors
    normalized[list(feature_cols)] = values
    note = negative_values_sentence(means)
    if note is not None:
        # Normalised as asked, but the reader must know the ratios are not "x control".
        normalized.attrs["normalization_note"] = note
        warnings.warn(f"normalize_to_control: {note}.", UserWarning, stacklevel=2)
    return normalized


__all__ = [
    "CONTROL_MEAN",
    "METHODS",
    "control_means",
    "negative_values_sentence",
    "normalize_to_control",
]

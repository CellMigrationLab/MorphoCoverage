"""The feature-column rule of ``data_check.check_tables`` (private).

The features named, distinct, in one role, present in the observation table, numeric and
finite; their missing values reported. Like every rule of ``_data_rules``, it records
problems on a ``Collector`` in plain words and never raises for bad data.

Shared-private: ``_check_features``, used by ``data_check``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from ._data_problems import Collector, _examples, _names
from ._wording import count, percent
from .metadata import repeated_names
from .schema import PLANE_FEATURE_ADVICE, feature_dtype_problem, plane_features


def _check_features(
    c: Collector, objects: pd.DataFrame, feature_cols: Sequence[str], taken: set[str]
) -> None:
    """Feature columns: named, distinct, present, numeric, finite; missing values reported."""
    if not feature_cols:
        c.error(
            "No feature column was named.",
            "feature_cols names no column.",
            "Name the numeric phenotype columns, e.g. feature_cols=['area'] (or Setup.readout).",
        )
        return
    _check_feature_roles(c, feature_cols, taken)
    _check_missing_features(c, objects, feature_cols)
    for feature in dict.fromkeys(feature_cols):
        if feature not in objects.columns:
            continue
        _check_feature_values(c, objects, feature)


def _check_feature_roles(c: Collector, feature_cols: Sequence[str], taken: set[str]) -> None:
    """Each feature named once, never a group or field identifier, never a plane coordinate."""
    # Sorted as text: column names may mix strings and numbers.
    duplicates = sorted(repeated_names(feature_cols), key=str)
    if duplicates:
        c.error(
            "A feature column is named twice.",
            f"feature_cols: {_names(duplicates)}.",
            "Name each feature once.",
        )
    overlap = sorted(set(feature_cols) & taken, key=str)
    if overlap:
        c.error(
            "A column is used both as a feature and as a group or field identifier.",
            f"columns {_names(overlap)}.",
            "Give each column one role: remove it from feature_cols or from group_cols/fov_col.",
        )
    planes = plane_features(feature_cols)
    if planes:
        c.error(
            "A within-field coordinate is named as a feature.",
            f"feature_cols: {_names(planes)}.",
            PLANE_FEATURE_ADVICE[0].upper() + PLANE_FEATURE_ADVICE[1:] + ".",
        )


def _check_missing_features(
    c: Collector, objects: pd.DataFrame, feature_cols: Sequence[str]
) -> None:
    """The features not in the observation table, with a near spelling when it has one."""
    missing = [f for f in dict.fromkeys(feature_cols) if f not in objects.columns]
    if not missing:
        return
    close = {f: _close_column(f, objects.columns) for f in missing}
    hints = [f"{found!r} for {f!r}" for f, found in close.items() if found is not None]
    c.error(
        f"{count(len(missing), 'feature column')} named in feature_cols "
        f"{'is' if len(missing) == 1 else 'are'} not in the observation table.",
        f"missing: {_names(missing)}"
        + (f"; the table has {', '.join(hints)}" if hints else "")
        + ".",
        "Check the spelling (spaces and case count), or add the columns to the observation table.",
    )


def _check_feature_values(c: Collector, objects: pd.DataFrame, feature: Any) -> None:
    """One feature's values: real numbers, finite; its missing values reported."""
    values = objects[feature]
    problem = feature_dtype_problem(values)
    if problem == "complex":
        c.error(
            f"Feature {feature!r} holds complex numbers.",
            f"observation table, column {feature!r}.",
            "Keep one real number per feature: split it into its real and imaginary parts "
            "(or its modulus and phase), each a feature of its own.",
        )
        return
    if problem is not None:
        _report_not_numeric(c, objects, feature)
        return
    numbers = values.to_numpy(dtype=float, na_value=np.nan)
    infinite = pd.Series(np.isinf(numbers), index=objects.index)
    if infinite.any():
        c.error(
            f"Feature {feature!r} contains infinite values.",
            f"observation table, {_examples(objects, infinite, [feature])}.",
            "Replace infinite values by missing values (NaN) or fix their computation, for "
            "example a division by zero.",
        )
    nan = pd.Series(np.isnan(numbers), index=objects.index)
    if nan.any():
        c.warning(
            f"Feature {feature!r} is missing in {percent(nan.mean())} of observations.",
            f"observation table, {_examples(objects, nan, [feature])}.",
            "Missing values are kept and never imputed; check they are expected (for "
            "example, a measurement that failed).",
        )


def _report_not_numeric(c: Collector, objects: pd.DataFrame, feature: Any) -> None:
    """A feature whose values are not numbers: the values that are not, or its type."""
    values = objects[feature]
    as_numbers = pd.to_numeric(values, errors="coerce")
    bad = as_numbers.isna() & values.notna()
    c.error(
        f"Feature {feature!r} is not numeric.",
        "observation table, "
        + (
            _examples(objects, bad, [feature])
            if bad.any()
            else "column type (every value is a number, stored as text or objects)"
        )
        + ".",
        "Convert the column to numbers (a decimal point, not a comma; empty cells for "
        "missing values)"
        + (
            "."
            if bad.any()
            else f", e.g. objects[{feature!r}] = pd.to_numeric(objects[{feature!r}])."
        ),
    )


def _close_column(name: Any, columns: Sequence[Any]) -> Any | None:
    """A column that differs from ``name`` only in case or surrounding whitespace.

    ``None`` for a name that is not text (``7``): only text has case or spaces.
    """
    if not isinstance(name, str):
        return None
    target = name.strip().casefold()
    for column in columns:
        if isinstance(column, str) and column != name and column.strip().casefold() == target:
            return column
    return None
